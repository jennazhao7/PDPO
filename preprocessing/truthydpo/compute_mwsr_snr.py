#!/usr/bin/env python3
"""
Compute MWSR and SNR from a privatized Truthy-DPO JSONL with:
  - margin_normalized (or margin_raw)
  - flipped (bool) where flipped=True means label was swapped

We set y_tilde = +1 if not flipped else -1.
Metrics:
  MWSR = mean(y_tilde * margin)
  SNR  = (mean(z)^2) / (var(z) + 1e-12) where z = y_tilde * margin

Outputs a JSON file with metrics.
"""

import argparse
import json
import numpy as np
from pathlib import Path


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
    ap.add_argument("--input", default="preprocessing/truthydpo/dpo_privatized_dataset.jsonl", help="Privatized JSONL with margin_normalized and flipped")
    ap.add_argument("--margin_field", default="margin_normalized", help="Which margin field to use")
    ap.add_argument("--output_json", default="eval_stage1/truthydpo/mwsr_snr.json", help="Where to write metrics JSON")
    args = ap.parse_args()

    rows = load_rows(args.input)
    if not rows:
        raise ValueError("No rows loaded.")

    margins = []
    flips = []
    for r in rows:
        if args.margin_field not in r:
            raise ValueError(f"Missing {args.margin_field} in row.")
        margins.append(float(r[args.margin_field]))
        flips.append(bool(r.get("flipped", False)))

    margins = np.array(margins, dtype=float)
    y_tilde = np.where(np.array(flips, dtype=bool), -1.0, 1.0)
    z = y_tilde * margins

    mwsr = float(np.mean(z))
    var_z = float(np.var(z))
    snr = float((np.mean(z) ** 2) / (var_z + 1e-12))

    out = {
        "input": args.input,
        "margin_field": args.margin_field,
        "num_rows": len(rows),
        "mwsr": mwsr,
        "snr": snr,
        "mean_z": float(np.mean(z)),
        "var_z": var_z,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Saved metrics to {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

