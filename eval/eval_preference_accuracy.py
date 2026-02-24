#!/usr/bin/env python3
"""
Compute clean-label preference accuracy and ECE on test_pref.jsonl.
For each adapter, scores pairs as log P(chosen)-log P(rejected) and measures calibration.
"""
import argparse
import json
import math

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def _resolve_adapter_path(path, manifest_dir):
    import os
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    sub = os.path.join(path, "stage2")
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return sub
    return path


def load_stage2(manifest_path, device):
    import os
    m = json.load(open(manifest_path, "r", encoding="utf-8"))
    base = pick(m, ["base_model", "base_model_id", "base_model_name_or_path"])
    adapters = m.get("adapters", [])
    if not adapters or len(adapters) < 2:
        raise ValueError(f"Manifest needs 2 adapters: {manifest_path}")
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    s1 = _resolve_adapter_path(adapters[0]["path"], manifest_dir)
    s2 = _resolve_adapter_path(adapters[1]["path"], manifest_dir)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, s1)
    model = PeftModel.from_pretrained(model, s2)
    model = model.to(device).eval()
    return model, tok


@torch.no_grad()
def resp_logprob(model, tok, prompt, resp, device, max_len=512):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"]
    ids = p + r
    if len(ids) < 2:
        return 0.0
    ids = ids[-max_len:]
    p_len = min(len(p), len(ids) - 1)
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x)
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    target = x[:, 1:]
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    start = max(p_len - 1, 0)
    return float(tok_lp[start:].sum().item())


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_ece(conf, correct, n_bins=10):
    ece = 0.0
    bins = []
    n = len(conf)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            bins.append({"bin": b, "count": 0, "acc": None, "conf": None})
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        cavg = sum(conf[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - cavg)
        bins.append({"bin": b, "count": len(idx), "acc": acc, "conf": cavg})
    return ece, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.test_jsonl, "r", encoding="utf-8") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_stage2(args.manifest, device)

    margins, confs, corr = [], [], []
    for r in rows:
        m = resp_logprob(model, tok, r["prompt"], r["chosen"], device, args.max_len) - resp_logprob(
            model, tok, r["prompt"], r["rejected"], device, args.max_len
        )
        margins.append(m)
        confs.append(sigmoid(m))
        corr.append(1 if m > 0 else 0)

    acc = sum(corr) / len(corr)
    ece, bins = compute_ece(confs, corr, args.n_bins)

    out = {
        "manifest": args.manifest,
        "n": len(rows),
        "accuracy": acc,
        "ece": ece,
        "mean_margin": sum(margins) / len(margins),
        "bins": bins,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"accuracy": acc, "ece": ece, "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
