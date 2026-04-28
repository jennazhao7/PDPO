#!/usr/bin/env python3
"""
Simulate PROPs Stage-1 RR on the 1016 Truthy-DPO samples, using existing margins.

Inputs (must contain prompt, chosen, rejected, and a margin field):
  - Default tries to load HuggingFace dataset at ./props_with_processed_margins
    (the disk dataset produced earlier in this workflow).
  - Or pass a JSONL/Arrow/Parquet via --input_jsonl with margin_field present.

RR parameters:
  q0 = 1 / (exp(eps) + 1)
  keep_prob = exp(eps) / (exp(eps) + 1)
  flipped ~ Bernoulli(q0)
  y_true = +1 (chosen preferred), y_tilde = +1 if not flipped else -1

Outputs (JSONL):
  eval_stage1/PROPs_stage1/truthydpo/truthy_1016_props_rr_stage1.jsonl
Fields: prompt, chosen, rejected, margin_field value, y_true, flip_prob, flipped, y_tilde, id

Also writes metrics:
  - MWSR/SNR to eval_stage1/PROPs_stage1/truthydpo/mwsr_snr_props.json
  - Flip curve to eval_stage1/PROPs_stage1/truthydpo/flip_curve_props.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset, load_from_disk


def load_source(args):
    if args.input_jsonl:
        ds = load_dataset("json", data_files=args.input_jsonl, split="train")
    else:
        ds = load_from_disk(args.input_hf)
        if "train" in ds:
            ds = ds["train"]
    needed = ["prompt", "chosen", "rejected", args.margin_field]
    missing = [c for c in needed if c not in ds.column_names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return ds


def save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def compute_mwsr_snr(margins, y_tilde):
    z = y_tilde * margins
    mwsr = float(np.mean(z))
    var_z = float(np.var(z))
    snr = float((np.mean(z) ** 2) / (var_z + 1e-12))
    return {
        "mwsr": mwsr,
        "snr": snr,
        "mean_z": float(np.mean(z)),
        "var_z": var_z,
    }


def compute_flip_curve(abs_m, flips, bins):
    lo, hi = abs_m.min(), abs_m.max()
    edges = np.linspace(lo, hi, bins + 1)
    bin_idx = np.digitize(abs_m, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, bins - 1)
    per_bin = []
    for b in range(bins):
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
    return {
        "abs_margin_min": float(lo),
        "abs_margin_max": float(hi),
        "overall_flip_rate": overall_flip,
        "overall_keep_rate": 1.0 - overall_flip,
        "edges": edges.tolist(),
        "per_bin": per_bin,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=1.0, help="RR epsilon")
    ap.add_argument("--margin_field", default="margin_normalized", help="Margin field to use")
    ap.add_argument("--input_hf", default="/users/jzhao7/PDPO/props_with_processed_margins", help="Path to HF disk dataset (train split)")
    ap.add_argument("--input_jsonl", default=None, help="Optional JSONL input with prompt/chosen/rejected/margin")
    ap.add_argument("--bins", type=int, default=10, help="Bins for flip curve on |margin|")
    ap.add_argument("--out_dir", default="/users/jzhao7/PDPO/eval_stage1/PROPs_stage1/truthydpo", help="Output directory")
    args = ap.parse_args()

    ds = load_source(args)
    keep_prob = math.exp(args.eps) / (math.exp(args.eps) + 1.0)
    flip_prob = 1.0 - keep_prob

    rows_out = []
    margins = []
    flips = []

    for i, ex in enumerate(ds):
        m = float(ex[args.margin_field])
        flipped = bool(np.random.rand() < flip_prob)
        y_true = 1
        y_tilde = 1 if not flipped else -1
        rows_out.append(
            {
                "id": int(i),
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
                args.margin_field: m,
                "y_true": y_true,
                "flip_prob": flip_prob,
                "flipped": flipped,
                "y_tilde": y_tilde,
            }
        )
        margins.append(m)
        flips.append(flipped)

    margins = np.array(margins, dtype=float)
    flips = np.array(flips, dtype=bool)
    y_tilde = np.where(flips, -1.0, 1.0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "truthy_1016_props_rr_stage1.jsonl"
    save_jsonl(rows_out, out_file)
    print(f"Saved RR-simulated file: {out_file} (n={len(rows_out)})")

    # Metrics
    mwsr_snr = compute_mwsr_snr(margins, y_tilde)
    flip_curve = compute_flip_curve(np.abs(margins), flips, args.bins)

    mwsr_snr.update(
        {
            "input": args.input_jsonl or args.input_hf,
            "margin_field": args.margin_field,
            "eps": args.eps,
            "flip_prob": flip_prob,
            "keep_prob": keep_prob,
        }
    )
    flip_curve.update(
        {
            "input": args.input_jsonl or args.input_hf,
            "margin_field": args.margin_field,
            "eps": args.eps,
            "flip_prob": flip_prob,
            "keep_prob": keep_prob,
            "bins": args.bins,
        }
    )

    mwsr_path = out_dir / "mwsr_snr_props.json"
    flip_path = out_dir / "flip_curve_props.json"
    with mwsr_path.open("w", encoding="utf-8") as f:
        json.dump(mwsr_snr, f, indent=2)
    with flip_path.open("w", encoding="utf-8") as f:
        json.dump(flip_curve, f, indent=2)
    print(f"Saved metrics: {mwsr_path} and {flip_path}")


if __name__ == "__main__":
    main()

