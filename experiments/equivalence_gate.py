#!/usr/bin/env python3
"""EQUIVALENCE GATE -- does the optimized eval path reproduce the committed cell exactly?

Compares a freshly recomputed checkpoint evaluation against the already-committed artifacts for
the same checkpoint. The optimization is only admissible if it changes nothing measurable.

PASS CRITERIA (both required)
  - per-example score max |diff| < 1e-4, over EVERY statistic and BOTH splits
  - AUC agreement to 4 decimal places

WHY PER-EXAMPLE AND NOT JUST AUC
  AUC is a rank statistic: it can agree to four decimals while individual scores drift, because
  ranks survive small perturbations. Per-example agreement is the property that actually licenses
  reusing the Stage 1 calibration curve, so it is checked first and independently.

Reference values for rr_eps0.0|e1|r16|p025: signed 0.5149, magnitude 0.6322.

CPU only. Compares persisted artifacts; runs no model.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stage2_debugging"))

from metric_primitives import MIA_STATISTICS  # noqa: E402

SCORE_TOL = 1e-4
AUC_PLACES = 4


def load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compare_scores(baseline: List[Dict], candidate: List[Dict], label: str) -> Dict[str, Any]:
    """Per-example agreement, joined on pair_key so a reordering cannot hide a mismatch."""
    if len(baseline) != len(candidate):
        return {"split": label, "ok": False,
                "error": f"length {len(baseline)} != {len(candidate)}"}
    b_by = {r["pair_key"]: r for r in baseline}
    c_by = {r["pair_key"]: r for r in candidate}
    missing = set(b_by) ^ set(c_by)
    if missing:
        return {"split": label, "ok": False,
                "error": f"{len(missing)} pair_keys differ between runs"}
    worst: Dict[str, Any] = {}
    for stat in MIA_STATISTICS:
        if stat not in baseline[0]:
            continue
        m, arg = 0.0, None
        for key, b in b_by.items():
            d = abs(float(b[stat]) - float(c_by[key][stat]))
            if d > m:
                m, arg = d, key
        worst[stat] = {"max_abs_diff": m, "at_pair_key": arg, "ok": m < SCORE_TOL}
    return {"split": label, "n": len(baseline), "per_statistic": worst,
            "max_abs_diff": max((v["max_abs_diff"] for v in worst.values()), default=0.0),
            "ok": all(v["ok"] for v in worst.values())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-mia", required=True, help="committed mia.json")
    ap.add_argument("--candidate-mia", required=True, help="mia.json from the optimized path")
    ap.add_argument("--baseline-eval", default=None, help="committed eval.json")
    ap.add_argument("--candidate-eval", default=None, help="eval.json from the optimized path")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    b = load(Path(args.baseline_mia))
    c = load(Path(args.candidate_mia))
    report: Dict[str, Any] = {
        "score_tolerance": SCORE_TOL, "auc_decimal_places": AUC_PLACES,
        "baseline_mia": args.baseline_mia, "candidate_mia": args.candidate_mia,
        "candidate_reference_cache": c.get("reference_cache"),
    }

    print("=" * 96)
    print("GATE 1/3  PER-EXAMPLE SCORES  (tolerance |diff| < 1e-4, joined on pair_key)")
    print("=" * 96)
    splits = [
        compare_scores(b["member_scores"], c["member_scores"], "members"),
        compare_scores(b["nonmember_scores"], c["nonmember_scores"], "nonmembers"),
    ]
    report["score_comparison"] = splits
    for s in splits:
        if "error" in s:
            print(f"  {s['split']:12s} ERROR: {s['error']}")
            continue
        print(f"  {s['split']:12s} n={s['n']}  max|diff| over all statistics = "
              f"{s['max_abs_diff']:.3e}  -> {'PASS' if s['ok'] else 'FAIL'}")
        for stat, v in s["per_statistic"].items():
            print(f"      {stat:38s} {v['max_abs_diff']:.3e}  {'ok' if v['ok'] else 'FAIL'}")
    scores_ok = all(s.get("ok") for s in splits)

    print()
    print("=" * 96)
    print(f"GATE 2/3  AUC  (agreement to {AUC_PLACES} decimal places)")
    print("=" * 96)
    ba, ca = b.get("attacks", {}), c.get("attacks", {})
    auc_rows = []
    print(f"  {'statistic':38s} {'baseline':>10} {'candidate':>10} {'|diff|':>10}  verdict")
    for stat in sorted(set(ba) | set(ca)):
        x, y = ba.get(stat, {}).get("auc"), ca.get(stat, {}).get("auc")
        if x is None or y is None:
            auc_rows.append({"statistic": stat, "ok": False, "error": "missing in one run"})
            print(f"  {stat:38s} {'--' if x is None else f'{x:.6f}':>10} "
                  f"{'--' if y is None else f'{y:.6f}':>10} {'--':>10}  FAIL (absent)")
            continue
        ok = round(float(x), AUC_PLACES) == round(float(y), AUC_PLACES)
        auc_rows.append({"statistic": stat, "baseline": x, "candidate": y,
                         "abs_diff": abs(x - y), "ok": ok})
        print(f"  {stat:38s} {x:>10.6f} {y:>10.6f} {abs(x - y):>10.3e}  "
              f"{'PASS' if ok else 'FAIL'}")
    report["auc_comparison"] = auc_rows
    auc_ok = all(r["ok"] for r in auc_rows)

    print()
    print("=" * 96)
    print("GATE 3/3  ARTIFACT SHAPE  (the per-token artifact must not change shape)")
    print("=" * 96)
    bp, cp = b.get("per_token_artifact", {}), c.get("per_token_artifact", {})
    shape_ok = True
    for field in ("n_rows", "n_token_logprobs"):
        x, y = bp.get(field), cp.get(field)
        ok = (x == y)
        shape_ok &= ok
        print(f"  {field:24s} baseline={x}  candidate={y}  -> {'PASS' if ok else 'FAIL'}")
    report["per_token_shape"] = {"baseline": bp, "candidate": cp, "ok": shape_ok}
    if not shape_ok:
        print("  Reference per-token arrays were NOT served from the cache. n_token_logprobs")
        print("  halving means the artifact lost the ref_chosen/ref_rejected series, which the")
        print("  four already-committed cells contain. Not admissible.")

    if args.baseline_eval and args.candidate_eval:
        print()
        print("=" * 96)
        print("SUPPLEMENTARY  accuracy eval agreement")
        print("=" * 96)
        be, ce = load(Path(args.baseline_eval)), load(Path(args.candidate_eval))
        acc_rows = []
        for split in ("train", "heldout"):
            for stat in ("reference_calibrated_dpo_margin", "raw_policy_preference_gap"):
                x = be.get("splits", {}).get(split, {}).get(stat, {}).get("accuracy")
                y = ce.get("splits", {}).get(split, {}).get(stat, {}).get("accuracy")
                if x is None or y is None:
                    continue
                ok = abs(x - y) < SCORE_TOL
                acc_rows.append({"split": split, "statistic": stat, "baseline": x,
                                 "candidate": y, "ok": ok})
                print(f"  {split:9s} {stat:38s} {x:.6f} vs {y:.6f}  "
                      f"{'PASS' if ok else 'FAIL'}")
        report["accuracy_comparison"] = acc_rows

    verdict = "PASS" if (scores_ok and auc_ok and shape_ok) else "FAIL"
    report["verdict"] = verdict
    report["gates"] = {"per_example_scores": scores_ok, "auc": auc_ok,
                       "per_token_shape": shape_ok}
    print()
    print("=" * 96)
    print(f"EQUIVALENCE GATE: {verdict}   "
          f"(scores={'PASS' if scores_ok else 'FAIL'}, "
          f"auc={'PASS' if auc_ok else 'FAIL'}, "
          f"shape={'PASS' if shape_ok else 'FAIL'})")
    print("=" * 96)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"wrote {args.out_json}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
