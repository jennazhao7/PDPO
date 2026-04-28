#!/usr/bin/env python3
"""
Evaluate base-model pairwise accuracy (no adapters) on preference eval sets.

Default datasets:
  - truthy
  - hhrlhf
  - pku

Runs on CPU by default; can use GPU with --device cuda.
"""

from __future__ import annotations

import argparse
import json
from typing import Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_TESTSETS = {
    "truthy": "/users/jzhao7/PDPO/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl",
    "hhrlhf": "/users/jzhao7/PDPO/stage2_debugging/test_pref.jsonl",
    "pku": "/users/jzhao7/PDPO/stage2_debugging/testsets/pku_secure/test_pref.jsonl",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Base-only pairwise accuracy on eval sets (CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B", help="Base model ID/path")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument(
        "--testsets_json",
        default=None,
        help="Optional JSON file mapping dataset name -> jsonl path",
    )
    ap.add_argument(
        "--out_json",
        default=None,
        help="Optional output JSON path for results",
    )
    ap.add_argument(
        "--datasets",
        default="truthy,hhrlhf,pku",
        help="Comma-separated dataset names to evaluate from testset mapping",
    )
    ap.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="cpu",
        help="Compute device for model inference",
    )
    return ap.parse_args()


@torch.no_grad()
def resp_logprob_sum(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    prompt: str,
    response: str,
    max_len: int,
    device: torch.device,
) -> float:
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[-max_len:]
    if len(ids) < 2:
        return 0.0

    p_len = min(len(p_ids), len(ids) - 1)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model(input_ids=x)
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    target = x[:, 1:]
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    start = max(p_len - 1, 0)
    return float(tok_lp[start:].sum().item())


def load_testsets(path: str | None) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_TESTSETS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not data:
        raise ValueError("--testsets_json must be a non-empty JSON object")
    return {str(k): str(v) for k, v in data.items()}


def main() -> int:
    args = parse_args()
    testsets = load_testsets(args.testsets_json)
    selected = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if not selected:
        raise ValueError("--datasets resolved to empty list")
    for ds in selected:
        if ds not in testsets:
            raise KeyError(f"Dataset '{ds}' not found in testsets mapping")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=None,
    ).to(device).eval()

    results = []
    for name in selected:
        path = testsets[name]
        rows = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
        correct = 0
        for r in rows:
            c = resp_logprob_sum(model, tok, r["prompt"], r["chosen"], args.max_len, device)
            j = resp_logprob_sum(model, tok, r["prompt"], r["rejected"], args.max_len, device)
            if c > j:
                correct += 1
        acc = correct / max(1, len(rows))
        rec = {
            "dataset": name,
            "n": len(rows),
            "base_pairwise_accuracy": acc,
            "device": str(device),
        }
        results.append(rec)
        print(f"{name}: n={len(rows)} base_pairwise_accuracy={acc:.4f}")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[saved] {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
