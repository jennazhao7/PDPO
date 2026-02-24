#!/usr/bin/env python3
"""
Check RR flip rate from audit file (d2_rr_audit_eps1.0_seed42.jsonl).
The main d2_rr_flipped JSONL does not contain was_flipped; the audit has observed_flip_rate.
"""
import argparse
import json
import math

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit_jsonl", required=True, help="Path to d2_rr_audit_eps*.jsonl")
    ap.add_argument("--tol", type=float, default=0.03, help="Tolerance for flip rate")
    args = ap.parse_args()

    with open(args.audit_jsonl, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        raise RuntimeError(f"No records in {args.audit_jsonl}")

    record = json.loads(lines[0])
    observed = record.get("observed_flip_rate")
    eps = record.get("epsilon", 1.0)
    n = record.get("N", 0)

    if observed is None:
        raise RuntimeError(f"audit record missing observed_flip_rate: {list(record.keys())}")

    expected = 1.0 / (math.exp(eps) + 1.0)
    lo = max(0.0, expected - args.tol)
    hi = min(1.0, expected + args.tol)
    ok = lo <= observed <= hi

    print(f"rows={n}")
    print(f"observed_flip_rate={observed:.6f}")
    print(f"expected_flip_rate={expected:.6f} (epsilon={eps})")
    print(f"tolerance=[{lo:.6f},{hi:.6f}]")
    print(f"PASS={ok}")

if __name__ == "__main__":
    main()
