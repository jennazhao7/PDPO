#!/usr/bin/env python3
"""E9 track B1 -- generate responses for the utility axis. Generation only; no judging.

WHAT THIS MEASURES
  Stage 1 established that leakage rises with fit. E9 asks the other half of the question: what
  does the model give you in exchange. Three Stage 1 checkpoints spanning the fit range, plus the
  base model as both a generator and the comparison target, all answering ONE fixed held-out
  prompt set.

PROMPT FORMAT MATCHES TRAINING EXACTLY
  Training and evaluation encode with `tokenizer(text, add_special_tokens=False)` and no chat
  template, no role wrapper (see ref_logprob_core.encode_prompt_response). Generation therefore
  feeds the raw prompt the same way. Generating in a different format than the model was trained
  in would make the utility comparison meaningless, so this is not a stylistic choice.

PRECISION
  FP32, matching the recorded project decision. If FP32 does not fit, this script RAISES rather
  than silently falling back to bf16 -- that substitution is a decision, not an implementation
  detail. Headroom is reported before generation starts.

DECODING IS GREEDY, DELIBERATELY
  do_sample=False. Sampling would add per-model variance that the judge cannot distinguish from a
  utility difference, and would make the run irreproducible without pinning RNG state across four
  separate model loads. Greedy removes that confound. Recorded in the manifest.

LENGTH IS A FIRST-CLASS OUTPUT, NOT A DIAGNOSTIC
  PROPS's rubric rates "level of detail", which is a length proxy, and this project has already
  shown the raw preference statistic behaves as a length detector. A win rate without the length
  distribution beside it is uninterpretable, so mean/median/quantiles are computed here and
  carried into the report.

The prompt set is drawn from the HELD-OUT split and its sha256 is recorded in the manifest, so
every model is provably answering the same questions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_debugging"))

from metric_primitives import canonical_json_hash, deterministic_sample  # noqa: E402

PROMPT_NAMESPACE = "e9_utility_prompts"
SEED = 42

# label -> manifest path relative to the stage1 output root. `base` has no adapter.
MODELS = {
    "base": None,
    "pku_e1_p075": "pku_e1_r16_seed42/checkpoint_p075/M2_manifest.json",
    "pku_e2_p050": "pku_e2_r16_seed42/checkpoint_p050/M2_manifest.json",
    "pku_e3_p100": "pku_e3_r16_seed42/checkpoint_p100/M2_manifest.json",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompts(test_pref: Path, n: int) -> List[Dict[str, Any]]:
    rows = [json.loads(x) for x in test_pref.open(encoding="utf-8") if x.strip()]
    sample = deterministic_sample(rows, n, SEED, PROMPT_NAMESPACE)
    return [{"id": r["id"], "prompt": r["prompt"]} for r in sample]


def length_stats(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {}
    ordered = sorted(lengths)

    def q(p: float) -> int:
        return ordered[min(len(ordered) - 1, int(p * len(ordered)))]

    return {
        "n": len(lengths),
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "min": ordered[0], "p25": q(0.25), "p75": q(0.75),
        "p90": q(0.90), "p95": q(0.95), "max": ordered[-1],
        # Saturation at the cap is the thing that silently flattens a length comparison.
        "histogram_32tok_bins": _hist(lengths),
    }


def _hist(lengths: List[int], width: int = 32) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in lengths:
        lo = (value // width) * width
        out[f"{lo}-{lo + width - 1}"] = out.get(f"{lo}-{lo + width - 1}", 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0].split("-")[0])))


def resolve_adapter(manifest_path: Path) -> str:
    """Adapter directory, taken from the manifest rather than assumed from the layout.

    One Stage 1 checkpoint is double-nested (checkpoint_p075/checkpoint_p075/stage2), so deriving
    the path from the directory name would silently load the wrong adapter or none at all."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adapters = manifest.get("adapters", [])
    if len(adapters) != 1:
        raise ValueError(f"expected exactly one adapter in {manifest_path}, got {len(adapters)}")
    declared = Path(adapters[0]["path"])
    here = manifest_path.resolve().parent
    for candidate in (declared, here / declared.name, here / "stage2",
                      here / declared.name / "stage2"):
        if (candidate / "adapter_config.json").is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"no adapter_config.json found for {manifest_path}; declared={declared}, searched under {here}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage1-root", default=str(ROOT / "experiments/stage1_operating_point"))
    ap.add_argument("--test-pref", default=str(ROOT / "data/pku_saferlhf_secure_v3/test_pref.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/e9_utility"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--n-prompts", type=int, default=300)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="Build the prompt set only. CPU.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = build_prompts(Path(args.test_pref), args.n_prompts)
    prompt_blob = canonical_json_hash([p["prompt"] for p in prompts])
    print(f"prompt set: n={len(prompts)} sha256={prompt_blob}")
    (out_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(p, sort_keys=True) + "\n" for p in prompts), encoding="utf-8")

    manifest: Dict[str, Any] = {
        "experiment": "E9_utility_axis",
        "base_model": args.model,
        "n_prompts": len(prompts),
        "prompt_set_sha256": prompt_blob,
        "prompt_source": str(Path(args.test_pref).relative_to(ROOT)),
        "prompt_split_role": "held-out (never trained on)",
        "prompt_sample": {"algorithm": "deterministic_sample", "seed": SEED,
                          "namespace": PROMPT_NAMESPACE},
        "decoding": {"strategy": "greedy", "do_sample": False,
                     "max_new_tokens": args.max_new_tokens},
        "precision": "float32",
        "models": {},
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2)[:900])
        print("\n--dry-run: prompt set built, no generation performed.")
        (out_dir / "MANIFEST.dryrun.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "Refusing to generate on CPU"
    device = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {torch.cuda.get_device_name(0)}  total {total_mem:.1f} GiB")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Left padding: generation must not have pad tokens between the prompt and the first
    # generated token, or every batched row after the longest would continue from padding.
    tokenizer.padding_side = "left"

    base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    base = base.to(device).eval()
    weights_gb = sum(p.numel() * p.element_size() for p in base.parameters()) / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"FP32 weights {weights_gb:.2f} GiB, reserved {reserved:.2f} GiB, "
          f"headroom {total_mem - reserved:.2f} GiB")
    if total_mem - reserved < 2.0:
        raise RuntimeError(
            f"FP32 does not fit with usable headroom ({total_mem - reserved:.2f} GiB free). "
            "STOPPING rather than falling back to bf16 -- that is a recorded decision, not an "
            "implementation detail.")
    manifest["gpu"] = {"name": torch.cuda.get_device_name(0), "total_gib": total_mem,
                       "fp32_weights_gib": weights_gb}

    for label, rel in MODELS.items():
        print(f"\n=== {label} ===", flush=True)
        if rel is None:
            model = base
            adapter_dir = None
        else:
            mp = Path(args.stage1_root) / rel
            adapter_dir = resolve_adapter(mp)
            print(f"  adapter: {adapter_dir}")
            model = PeftModel.from_pretrained(base, adapter_dir, adapter_name="stage2")
            model = model.to(device).eval()

        rows: List[Dict[str, Any]] = []
        started = time.time()
        with torch.no_grad():
            for start in range(0, len(prompts), args.batch_size):
                chunk = prompts[start : start + args.batch_size]
                enc = tokenizer([c["prompt"] for c in chunk], return_tensors="pt",
                                padding=True, add_special_tokens=False).to(device)
                out = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                for c, seq in zip(chunk, gen):
                    ids = [int(t) for t in seq.tolist() if t != tokenizer.pad_token_id]
                    text = tokenizer.decode(ids, skip_special_tokens=True)
                    rows.append({
                        "id": c["id"], "prompt": c["prompt"], "response": text,
                        "response_tokens": len(ids),
                        "hit_max_new_tokens": len(ids) >= args.max_new_tokens,
                        "response_sha256": sha256_text(text),
                    })
                if (start // args.batch_size) % 5 == 0:
                    print(f"  [{min(start + args.batch_size, len(prompts))}/{len(prompts)}]",
                          flush=True)
        elapsed = time.time() - started

        path = out_dir / f"generations_{label}.jsonl"
        path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows),
                        encoding="utf-8")
        lengths = [r["response_tokens"] for r in rows]
        capped = sum(r["hit_max_new_tokens"] for r in rows)
        stats = length_stats(lengths)
        manifest["models"][label] = {
            "adapter_dir": adapter_dir,
            "generations": str(path.relative_to(ROOT)),
            "n": len(rows),
            "seconds": elapsed,
            "tokens_per_second": sum(lengths) / elapsed if elapsed else None,
            "response_length_tokens": stats,
            "n_hit_max_new_tokens": capped,
            "frac_hit_max_new_tokens": capped / len(rows) if rows else 0.0,
        }
        print(f"  {len(rows)} responses in {elapsed:.0f}s | "
              f"mean {stats['mean']:.1f} median {stats['median']} tok | "
              f"capped {capped}/{len(rows)} ({100*capped/len(rows):.1f}%)")

        if rel is not None:
            # Unload so the next model starts from the clean base rather than stacking adapters.
            model = model.unload()
            del model
            torch.cuda.empty_cache()

    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'MANIFEST.json'}")

    print(f"\n{'model':14s} {'mean':>7} {'median':>7} {'p95':>6} {'max':>6} {'capped%':>8}")
    for label, m in manifest["models"].items():
        s = m["response_length_tokens"]
        print(f"{label:14s} {s['mean']:>7.1f} {s['median']:>7.1f} {s['p95']:>6} {s['max']:>6} "
              f"{100 * m['frac_hit_max_new_tokens']:>7.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
