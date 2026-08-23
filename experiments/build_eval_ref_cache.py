#!/usr/bin/env python3
"""Build the ONE base-model reference cache that serves eval for the entire Stage 2 sweep.

WHY THIS EXISTS
  Both evaluators recompute base-model logprobs on every checkpoint via `disable_adapter()`:
  18,000 of the 36,000 forward passes per cell. The base model is identical across all 60
  checkpoints, all epochs, and all arms, and eval always scores the TRUE unflipped orientation,
  so those 18,000 passes are the same ~14,000 values recomputed 60 times over -- 1,080,000
  forwards where 14,000 suffice.

WHY ONE CACHE SERVES EVERY ARM
  The RR swap is applied to the TRAINER's input only. Both evaluators read the true unflipped
  files, so the (prompt, chosen, rejected) orientation they present is arm-independent. The cache
  is keyed on that orientation via `pair_key`, so it is valid for every arm -- unlike the
  per-arm TRAINING cache, which the swap invalidates by construction.
  This cache is tagged `cache_role="eval_reference_true_orientation"` and carries `rr_eps=None`
  so it can never be confused with a per-arm training cache in either direction.

THE UNION
  The two evaluators do NOT use the same rows, and in particular draw DIFFERENT 2,000-row train
  subsamples (namespace `stage2_train_acc_true` vs `members`). The union is therefore:
    - all 3,000 held-out pairs                      (accuracy eval, and a superset of the
                                                      privacy audit's 2,000 non-member sample)
    - the accuracy eval's 2,000-row train subsample
    - the privacy audit's 2,000-row member sample
  deduplicated on `pair_key`. Up to 7,000 distinct pairs, i.e. up to 14,000 base forwards.

BIT-IDENTITY
  Scoring goes through `torch_batched_response_token_logprobs`, the exact function both
  evaluators call, and stores the sum the way the evaluator computes it (a torch float32
  reduction) alongside the per-token list. Default batch size is 1, matching the evaluator's
  single-sequence call precisely, because the equivalence gate demands per-example agreement
  below 1e-4 and a one-time 48-minute build is not worth risking that for.
  PARITY_FIX.md records that in FP32 the batched right-padded path equals the unpadded oracle
  exactly (the historical blowup was BF16-specific), so --batch-size >1 is available, but it is
  not the default and the gate is what licenses it.

FP32 only, matching the locked project precision.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_debugging"))

from matched_core import pair_key, text_hash  # noqa: E402
from metric_primitives import deterministic_sample  # noqa: E402
from ref_logprob_core import (  # noqa: E402
    REFERENCE_DTYPE,
    TRUNCATION_MODE,
    encode_prompt_response,
)

CACHE_ROLE = "eval_reference_true_orientation"
SUBSAMPLE = 2000


def read_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in path.open(encoding="utf-8") if x.strip()]


def build_union(train_pref: Path, test_pref: Path, acc_subsample: Path) -> List[Dict[str, Any]]:
    """Exactly the rows the two evaluators will present, deduplicated on pair_key."""
    test_rows = read_rows(test_pref)
    train_rows = read_rows(train_pref)
    if not acc_subsample.is_file():
        # run_stage2_rr.evaluate() normally writes this on its first call. The cache is built
        # BEFORE any evaluation, so generate it here with the identical deterministic call
        # (same size, seed and namespace) rather than depending on call order.
        print(f"accuracy-eval train subsample absent; generating {acc_subsample}")
        sample = deterministic_sample(train_rows, SUBSAMPLE, 42, "stage2_train_acc_true")
        acc_subsample.parent.mkdir(parents=True, exist_ok=True)
        with acc_subsample.open("w", encoding="utf-8") as h:
            for r in sample:
                h.write(json.dumps(r, sort_keys=True) + "\n")
    acc_train = read_rows(acc_subsample)

    # Mirrors eval_privacy_audit: sample_size = min(subset, len(members), len(nonmembers)).
    sample_size = min(SUBSAMPLE, len(train_rows), len(test_rows))
    members = deterministic_sample(train_rows, sample_size, 42, "members")
    nonmembers = deterministic_sample(test_rows, sample_size, 42, "nonmembers")

    sources = [
        ("accuracy_heldout(all test)", test_rows),
        ("accuracy_train(subsample file)", acc_train),
        ("privacy_members(sample)", members),
        ("privacy_nonmembers(sample)", nonmembers),
    ]
    union: Dict[str, Dict[str, Any]] = {}
    print("union sources:")
    for label, rows in sources:
        before = len(union)
        for row in rows:
            union.setdefault(pair_key(row), row)
        print(f"  {label:34s} n={len(rows):>5}  new={len(union) - before:>5}  union={len(union):>5}")
    print(f"\ndistinct pairs in union = {len(union)}  ->  {2 * len(union)} base forward passes")
    return list(union.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-pref", default=str(ROOT / "data/pku_saferlhf_secure_v3/train_pref.jsonl"))
    ap.add_argument("--test-pref", default=str(ROOT / "data/pku_saferlhf_secure_v3/test_pref.jsonl"))
    ap.add_argument("--acc-subsample", default=str(ROOT / "experiments/stage2_rr/train_acc_subsample_true.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "experiments/ref_logps_stage2/eval_ref_true_fp32.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="1 matches the evaluator's call exactly. >1 requires the gate to pass.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Build and report the union only. CPU only, no model, no GPU.")
    args = ap.parse_args()

    out = Path(args.out)
    rows = build_union(Path(args.train_pref), Path(args.test_pref), Path(args.acc_subsample))

    if args.dry_run:
        print("\n--dry-run: union verified, no scoring performed.")
        return 0

    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing eval reference cache: {out}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ref_logprob_core import torch_batched_response_token_logprobs

    assert torch.cuda.is_available(), "Refusing to build the reference cache on CPU"
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()

    def score(pairs: List[tuple[str, str]]) -> List[tuple[float, List[float]]]:
        """Sum and per-token logprobs, computed exactly as the evaluators compute them."""
        sequences, prompt_lens = [], []
        for prompt, response in pairs:
            ids, prompt_len = encode_prompt_response(tokenizer, prompt, response, args.max_len)
            sequences.append(ids)
            prompt_lens.append(prompt_len)
        with torch.no_grad():
            values, mask = torch_batched_response_token_logprobs(
                model, sequences, prompt_lens, tokenizer.pad_token_id, device
            )
        totals = (values * mask).sum(dim=1)
        results = []
        for i in range(len(sequences)):
            per_token = [float(v) for v in values[i][mask[i].bool()].tolist()]
            results.append((float(totals[i].item()), per_token))
        return results

    records: List[Dict[str, Any]] = []
    total_forwards = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        chosen = score([(r["prompt"], r["chosen"]) for r in batch])
        rejected = score([(r["prompt"], r["rejected"]) for r in batch])
        total_forwards += 2 * len(batch)
        for row, (c_sum, c_pt), (r_sum, r_pt) in zip(batch, chosen, rejected):
            records.append({
                "pair_key": pair_key(row),
                "cache_role": CACHE_ROLE,
                # Explicitly arm-independent: eval always scores the true unflipped orientation.
                "rr_eps": None,
                "base_model": args.model,
                "scoring_dtype": REFERENCE_DTYPE,
                "truncation": TRUNCATION_MODE,
                "max_len": args.max_len,
                "build_batch_size": args.batch_size,
                "prompt_hash": text_hash(str(row["prompt"])),
                "chosen_hash": text_hash(str(row["chosen"])),
                "rejected_hash": text_hash(str(row["rejected"])),
                "chosen_ref_logp": c_sum,
                "rejected_ref_logp": r_sum,
                "chosen_ref_per_token": c_pt,
                "rejected_ref_per_token": r_pt,
            })
        if (start // max(1, args.batch_size)) % 200 == 0:
            print(f"  [{min(start + args.batch_size, len(rows))}/{len(rows)}] "
                  f"forwards={total_forwards}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(out)
    print(f"\nwrote {len(records)} pairs to {out}")
    print(f"base forward passes spent: {total_forwards} "
          f"(vs {18000 * 60} on the unoptimized path across 60 cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
