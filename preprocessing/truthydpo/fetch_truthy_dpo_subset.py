#!/usr/bin/env python3
"""
Fetch a subset of jondurbin/truthy-dpo-v0.1 for quick analysis.

Outputs a JSONL with columns: prompt, chosen, rejected (and id).

Usage:
  python fetch_truthy_dpo_subset.py \
    --output preprocessing/truthydpo/truthy_dpo_subset.jsonl \
    --n 5000 \
    --seed 0
"""

import argparse
import json
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="preprocessing/truthydpo/truthy_dpo_subset.jsonl")
    ap.add_argument("--n", type=int, default=5000, help="Number of rows to keep")
    ap.add_argument("--seed", type=int, default=0, help="Shuffle seed; set -1 to keep original order")
    args = ap.parse_args()

    print("Loading dataset jondurbin/truthy-dpo-v0.1 ...")
    ds = load_dataset("jondurbin/truthy-dpo-v0.1", split="train")
    print(f"Total rows: {len(ds)}")

    if args.seed >= 0:
        ds = ds.shuffle(seed=args.seed)

    ds = ds.select(range(min(args.n, len(ds))))
    print(f"Keeping {len(ds)} rows")

    keep_cols = ["prompt", "chosen", "rejected"]
    with open(args.output, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            out = {"id": i}
            for k in keep_cols:
                if k not in ex:
                    raise ValueError(f"Missing column {k} in dataset")
                out[k] = ex[k]
            f.write(json.dumps(out) + "\n")

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

