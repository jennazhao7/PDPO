#!/usr/bin/env python3
"""Per-arm read-out for Stage 2 RR. Emits the report required at the completion of EACH arm:
a metrics table, signed versus magnitude side by side, and observed signed attenuation against
the tanh(eps/2) prediction.

FIT-MATCHING IS ON MEASURED GAP, NEVER ON EPOCH FRACTION.
  HH-RLHF established that epoch fraction does not transfer across conditions. So each RR
  checkpoint is matched to the Stage 1 clean curve by linear interpolation on measured
  train/held-out gap, and attenuation is

      observed = (AUC_rr - 0.5) / (AUC_clean_at_the_same_gap - 0.5)

  If an RR checkpoint's gap falls outside the Stage 1 curve's measured gap range, this script
  REFUSES to interpolate and reports the cell as out-of-support. Extrapolating the clean curve
  past its data would manufacture an attenuation number, and at low eps that is the common case:
  coin-flip labels leave little to fit, so the gap lands below anything Stage 1 measured.

PREDICTION AXIS -- TWO NUMBERS, NOT ONE.
  Nominal tanh(eps/2) uses the nominal gamma. But the flip draw is finite-sample: the realized
  flip rate in data/stage2_rr/rr_audit.json differs from gamma, and 1 - 2*realized_rate is what
  the data can actually show. Both are printed. Neither is fitted to anything.

Only AUC and TPR@1%FPR are reported, each with the count of non-members defining the threshold.
TPR@0.1%FPR is retired project-wide and is deliberately absent.

CPU only. Reads persisted artifacts. Reports what was measured, not what was predicted.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]

SIGNED = "reference_calibrated_dpo_margin"
MAGNITUDE = "abs_reference_calibrated_margin"
ARM_EPS = (0.0, 0.1, 0.5, 1.0, 2.0)
CHECKPOINTS_PER_ARM = 12  # 3 epochs x 4 checkpoint percents


def tanh_half(eps: float) -> float:
    return math.tanh(eps / 2.0)


def attacks_of(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return metrics.get("mia_attacks") or metrics.get("attacks") or {}


def gap_of(metrics: Dict[str, Any]) -> Optional[float]:
    for key in ("train_heldout_gap_true", "train_heldout_gap"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def read_ledger(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"ledger not found: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_curve(rows: List[Dict[str, Any]]) -> List[Tuple[float, float, str]]:
    """(gap, signed AUC, cell key) for the Stage 1 clean PKU checkpoints, sorted by gap."""
    points = []
    for row in rows:
        if row.get("dataset") not in (None, "pku"):
            continue
        metrics = row.get("metrics", {})
        gap = gap_of(metrics)
        auc = attacks_of(metrics).get(SIGNED, {}).get("auc")
        if gap is None or not isinstance(auc, (int, float)):
            continue
        points.append((gap, float(auc), row.get("key", "?")))
    points.sort()
    if len(points) < 2:
        raise ValueError("clean curve needs at least two usable checkpoints")
    return points


def interpolate_clean(curve: List[Tuple[float, float, str]], gap: float) -> Tuple[Optional[float], str]:
    """Linear interpolation of clean signed AUC at a measured gap. Refuses to extrapolate."""
    low, high = curve[0][0], curve[-1][0]
    if gap < low:
        return None, f"gap {gap:.4f} below clean support [{low:.4f}, {high:.4f}]"
    if gap > high:
        return None, f"gap {gap:.4f} above clean support [{low:.4f}, {high:.4f}]"
    for (g0, a0, _), (g1, a1, _) in zip(curve, curve[1:]):
        if g0 <= gap <= g1:
            if g1 == g0:
                return a0, "exact"
            frac = (gap - g0) / (g1 - g0)
            return a0 + frac * (a1 - a0), f"interpolated between gap {g0:.4f} and {g1:.4f}"
    return None, "no bracketing interval"


def realized_rates(audit_path: Path) -> Dict[float, float]:
    if not audit_path.is_file():
        return {}
    return {float(a["eps"]): float(a["observed_flip_rate"]) for a in json.loads(audit_path.read_text())}


def fmt(value: Any, places: int = 4) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if value != 0 and abs(value) < 1e-3:
        return f"{value:.3e}"
    return f"{value:.{places}f}"


def ci(block: Dict[str, Any], key: str) -> str:
    pair = block.get(key)
    if not isinstance(pair, (list, tuple)) or len(pair) != 2:
        return "n/a"
    return f"[{fmt(pair[0])}, {fmt(pair[1])}]"


def render_arm(
    eps: float,
    cells: List[Dict[str, Any]],
    curve: List[Tuple[float, float, str]],
    realized: Dict[float, float],
) -> str:
    cells = sorted(cells, key=lambda r: (r.get("epochs", 0), r.get("percent", 0)))
    nominal = tanh_half(eps)
    rate = realized.get(eps)
    lines = [
        f"# Stage 2 RR — arm eps = {eps}",
        "",
        f"Cells committed: **{len(cells)} / {CHECKPOINTS_PER_ARM}**"
        + ("" if len(cells) == CHECKPOINTS_PER_ARM else "  — **ARM INCOMPLETE**"),
        "",
        f"Prediction axis: nominal tanh({eps}/2) = **{nominal:.6f}**"
        + (f"; realized flip rate {rate:.4f} gives 1 − 2·rate = **{1 - 2 * rate:.6f}**"
           if rate is not None else "; realized flip rate unavailable"),
        "",
        "## Metrics — signed and magnitude side by side",
        "",
        "| cell | gap | loss | signed AUC | 95% CI | signed TPR@1% | 95% CI "
        "| mag AUC | 95% CI | mag TPR@1% | 95% CI | n_non@1% |",
        "|---|---:|---:|---:|---|---:|---|---:|---|---:|---|---:|",
    ]
    for row in cells:
        metrics = row.get("metrics", {})
        atk = attacks_of(metrics)
        s = atk.get(SIGNED, {})
        m = atk.get(MAGNITUDE, {})
        lines.append(
            f"| `e{row.get('epochs')}·p{row.get('percent'):03d}` "
            f"| {fmt(gap_of(metrics))} | {fmt(metrics.get('loss'))} "
            f"| {fmt(s.get('auc'))} | {ci(s, 'auc_ci')} "
            f"| {fmt(s.get('tpr_at_1pct_fpr'))} | {ci(s, 'tpr_at_1pct_fpr_ci')} "
            f"| {fmt(m.get('auc'))} | {ci(m, 'auc_ci')} "
            f"| {fmt(m.get('tpr_at_1pct_fpr'))} | {ci(m, 'tpr_at_1pct_fpr_ci')} "
            f"| {metrics.get('n_nonmembers_defining_fpr_1pct', 'n/a')} |"
        )

    lines += [
        "",
        "Every TPR above is at 1% FPR, with the `n_non@1%` column giving the number of "
        "non-members that defines that threshold. TPR@0.1%FPR is retired project-wide.",
        "",
        "## Observed signed attenuation vs prediction",
        "",
        "Each RR checkpoint is matched to the Stage 1 clean curve on **measured gap**, never on "
        "epoch fraction. Out-of-support cells are refused, not extrapolated.",
        "",
        "| cell | gap | clean signed AUC @ same gap | RR signed AUC | observed attenuation "
        "| nominal tanh(eps/2) | realized 1−2·rate | basis |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    observed_values = []
    for row in cells:
        metrics = row.get("metrics", {})
        gap = gap_of(metrics)
        rr_auc = attacks_of(metrics).get(SIGNED, {}).get("auc")
        cell = f"`e{row.get('epochs')}·p{row.get('percent'):03d}`"
        if gap is None or not isinstance(rr_auc, (int, float)):
            lines.append(f"| {cell} | {fmt(gap)} | — | {fmt(rr_auc)} | — | {nominal:.6f} | — | missing inputs |")
            continue
        clean_auc, basis = interpolate_clean(curve, gap)
        if clean_auc is None:
            lines.append(
                f"| {cell} | {fmt(gap)} | — | {fmt(rr_auc)} | **out of support** "
                f"| {nominal:.6f} | {fmt(1 - 2 * rate, 6) if rate is not None else '—'} | {basis} |"
            )
            continue
        denominator = clean_auc - 0.5
        if abs(denominator) < 1e-9:
            lines.append(
                f"| {cell} | {fmt(gap)} | {fmt(clean_auc)} | {fmt(rr_auc)} "
                f"| undefined (clean at chance) | {nominal:.6f} | — | {basis} |"
            )
            continue
        observed = (rr_auc - 0.5) / denominator
        observed_values.append(observed)
        lines.append(
            f"| {cell} | {fmt(gap)} | {fmt(clean_auc)} | {fmt(rr_auc)} | **{observed:+.4f}** "
            f"| {nominal:.6f} | {fmt(1 - 2 * rate, 6) if rate is not None else '—'} | {basis} |"
        )

    lines.append("")
    if observed_values:
        mean_obs = sum(observed_values) / len(observed_values)
        lines += [
            f"Mean observed attenuation over the {len(observed_values)} in-support cell(s): "
            f"**{mean_obs:+.4f}**, against nominal **{nominal:.6f}**"
            + (f" and realized **{1 - 2 * rate:.6f}**." if rate is not None else "."),
            "",
            f"Difference from nominal: **{mean_obs - nominal:+.4f}**."
            + (f" From realized: **{mean_obs - (1 - 2 * rate):+.4f}**." if rate is not None else ""),
        ]
    else:
        lines += [
            "**No cell in this arm is in-support on the measured-gap axis**, so no attenuation "
            "figure is computable against the Stage 1 clean curve. This is a reportable fact "
            "about where the arm lands in fit, not a missing measurement: the clean curve's "
            f"measured gap range is [{curve[0][0]:.4f}, {curve[-1][0]:.4f}] and every cell here "
            "falls outside it.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage2-ledger", default=str(ROOT / "experiments/stage2_rr/ledger.jsonl"))
    ap.add_argument("--stage1-ledger", default=str(ROOT / "analysis_cpu/pulled/ledger_full.jsonl"))
    ap.add_argument("--rr-audit", default=str(ROOT / "data/stage2_rr/rr_audit.json"))
    ap.add_argument("--eps", type=float, action="append", default=[],
                    help="Report only these arms. Default: every arm with at least one cell.")
    ap.add_argument("--only-complete", action="store_true",
                    help="Skip arms with fewer than 12 committed cells.")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    stage2 = read_ledger(Path(args.stage2_ledger))
    curve = clean_curve(read_ledger(Path(args.stage1_ledger)))
    realized = realized_rates(Path(args.rr_audit))

    by_eps: Dict[float, List[Dict[str, Any]]] = {}
    for row in stage2:
        by_eps.setdefault(float(row["eps"]), []).append(row)

    wanted = args.eps or sorted(by_eps)
    sections = [
        "Stage 1 clean curve support on the measured-gap axis: "
        f"[{curve[0][0]:.4f}, {curve[-1][0]:.4f}] over {len(curve)} checkpoints.",
        "",
    ]
    for eps in wanted:
        cells = by_eps.get(eps, [])
        if not cells:
            sections.append(f"# Stage 2 RR — arm eps = {eps}\n\nNo committed cells.\n")
            continue
        if args.only_complete and len(cells) < CHECKPOINTS_PER_ARM:
            continue
        sections.append(render_arm(eps, cells, curve, realized))

    report = "\n".join(sections)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(report, encoding="utf-8")
        print(f"[arm-report] wrote {args.out_md}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
