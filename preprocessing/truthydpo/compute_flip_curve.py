#!/usr/bin/env python3
"""
Compute flip/keep curve by margin bin (Conditional MI proxy) for the privatized Truthy-DPO data.

We use:
  - margin (default: margin_normalized)
  - flipped (bool), where flipped=True means the label was swapped (y_tilde != y_original)

Per bin on |margin|, we report flip_rate = Pr(y_tilde != y | |m| in bin) and keep_rate = 1 - flip_rate.

Outputs a JSON file under eval_stage1 (by default).
"""

import argparse
import json
from pathlib import Path
import numpy as np


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="preprocessing/truthydpo/dpo_privatized_dataset.jsonl", help="Privatized JSONL with margin + flipped")
    ap.add_argument("--margin_field", default="margin_normalized", help="Field to use as margin")
    ap.add_argument("--bins", type=int, default=10, help="Number of bins on |margin|")
    ap.add_argument("--output_json", default="eval_stage1/truthydpo/flip_curve.json", help="Where to write metrics JSON")
    args = ap.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise ValueError("No rows loaded.")

    margins = []
    flips = []
    for r in rows:
        if args.margin_field not in r:
            raise ValueError(f"Missing {args.margin_field} in a row.")
        margins.append(float(r[args.margin_field]))
        flips.append(bool(r.get("flipped", False)))

    margins = np.array(margins, dtype=float)
    abs_m = np.abs(margins)
    flips = np.array(flips, dtype=bool)

    # Bin on |margin|
    lo, hi = abs_m.min(), abs_m.max()
    edges = np.linspace(lo, hi, args.bins + 1)
    bin_idx = np.digitize(abs_m, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, args.bins - 1)

    per_bin = []
    for b in range(args.bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            flip_rate = None
            keep_rate = None
            flips_sum = 0
        else:
            flips_sum = int(flips[mask].sum())
            flip_rate = flips_sum / count
            keep_rate = 1.0 - flip_rate
        per_bin.append(
            {
                "bin": b,
                "bin_start": float(edges[b]),
                "bin_end": float(edges[b + 1]),
                "count": count,
                "flips": flips_sum,
                "flip_rate": flip_rate,
                "keep_rate": keep_rate,
            }
        )

    overall_flip = float(flips.mean())

    out = {
        "input": args.input,
        "margin_field": args.margin_field,
        "bins": args.bins,
        "abs_margin_min": float(lo),
        "abs_margin_max": float(hi),
        "overall_flip_rate": overall_flip,
        "overall_keep_rate": 1.0 - overall_flip,
        "edges": edges.tolist(),
        "per_bin": per_bin,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved flip/keep curve to {out_path}")


if __name__ == "__main__":
    main()

