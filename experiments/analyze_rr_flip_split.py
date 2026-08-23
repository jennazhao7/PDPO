#!/usr/bin/env python3
"""Split Stage 2 RR members by the persisted `was_flipped` flag and report the magnitude
statistic separately for flipped and unflipped members.

WHY THIS EXISTS
  At eps=0 the magnitude statistic scored AUC 0.6322, ABOVE the clean control's 0.5900. That
  cross-model comparison is CONFOUNDED: the two models differ in fit (gap 0.027 vs 0.111), so it
  cannot speak to whether RR changed sign-invariant leakage. The within-model comparison below
  holds fit exactly constant -- same checkpoint, same weights, same non-member pool -- and is the
  correct test.

  This script REPORTS. It does not interpret.

HOW THE JOIN IS SOUND
  `preference_pair_key` hashes the prompt plus the SORTED pair of response hashes, so it is
  orientation-independent: the same example yields the same key in the TRUE unflipped train split
  and in the RR-flipped trainer file. Joining member scores to `rr_flipped` on that key is exact.
  Join coverage is asserted at 100%; anything less is a hard failure, not a warning.

  The member scores themselves come from the MIA artifact, which scored the TRUE unflipped label.
  Nothing here re-scores anything, so the invariant that the attack reads true labels is untouched.

CPU only. No GPU, no forward passes, no model loading. Reads persisted artifacts only.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "stage2_debugging"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from metric_primitives import (  # noqa: E402
    percentile,
    preference_pair_key,
    roc_auc_score,
    stratified_bootstrap_ci,
    tpr_at_fpr,
)

# Primary is the magnitude statistic. The signed one is carried for context only: at eps=0 it is
# at chance by construction, so it is not the quantity this deliverable is about.
MAGNITUDE_STATISTIC = "abs_reference_calibrated_margin"
SIGNED_STATISTIC = "reference_calibrated_dpo_margin"
STATISTICS = (MAGNITUDE_STATISTIC, SIGNED_STATISTIC)

CELL_RE = re.compile(r"rr_eps(?P<eps>[0-9.]+)_e(?P<epochs>\d+)_r(?P<lora>\d+)_seed(?P<seed>\d+)")
PCT_RE = re.compile(r"checkpoint_p(?P<pct>\d+)")


def bootstrap_mean_ci(values: Sequence[float], seed: int, samples: int) -> List[float]:
    """Percentile bootstrap on the mean of one stratum. Same RNG discipline as
    stratified_bootstrap_ci so results are reproducible from (seed, samples) alone."""
    if not values:
        raise ValueError("bootstrap requires a non-empty sample")
    data = [float(v) for v in values]
    rng = random.Random(seed)
    n = len(data)
    means = []
    for _ in range(samples):
        total = 0.0
        for _ in range(n):
            total += data[rng.randrange(n)]
        means.append(total / n)
    return [percentile(means, 0.025), percentile(means, 0.975)]


def bootstrap_diff_ci(
    a: Sequence[float], b: Sequence[float], seed: int, samples: int
) -> List[float]:
    """CI on mean(a) - mean(b), resampling the two strata independently."""
    if not a or not b:
        raise ValueError("bootstrap requires two non-empty samples")
    xa = [float(v) for v in a]
    xb = [float(v) for v in b]
    rng = random.Random(seed)
    diffs = []
    for _ in range(samples):
        sa = sum(xa[rng.randrange(len(xa))] for _ in xa) / len(xa)
        sb = sum(xb[rng.randrange(len(xb))] for _ in xb) / len(xb)
        diffs.append(sa - sb)
    return [percentile(diffs, 0.025), percentile(diffs, 0.975)]


def load_flip_map(rr_path: Path) -> Dict[str, bool]:
    flips: Dict[str, bool] = {}
    with rr_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "rr_flipped" not in row:
                raise KeyError(f"{rr_path} row lacks rr_flipped; cannot form the split")
            key = preference_pair_key(row)
            if key in flips and flips[key] != bool(row["rr_flipped"]):
                raise ValueError(f"conflicting rr_flipped for pair_key {key} in {rr_path}")
            flips[key] = bool(row["rr_flipped"])
    if not flips:
        raise ValueError(f"no rows read from {rr_path}")
    return flips


def group_block(
    member_values: Sequence[float],
    nonmember_values: Sequence[float],
    seed: int,
    samples: int,
) -> Dict[str, Any]:
    """Metrics for one member group against the SHARED non-member pool.

    Only AUC and TPR@1%FPR are reported, each with the count of non-members that defines the
    threshold. TPR@0.1%FPR is retired project-wide and is deliberately absent."""
    n_non = len(nonmember_values)
    labels = [1] * len(member_values) + [0] * n_non
    scores = list(member_values) + list(nonmember_values)

    def auc_metric(pos: Sequence[float], neg: Sequence[float]) -> float:
        return roc_auc_score([1] * len(pos) + [0] * len(neg), list(pos) + list(neg))

    def tpr_metric(pos: Sequence[float], neg: Sequence[float]) -> float:
        return tpr_at_fpr(pos, neg, 0.01)

    return {
        "n_members": len(member_values),
        "n_nonmembers": n_non,
        # The threshold at 1% FPR is set by this many non-members. Required beside every TPR.
        "n_nonmembers_defining_fpr_1pct": int(round(0.01 * n_non)),
        "mean": sum(member_values) / len(member_values),
        "mean_ci": bootstrap_mean_ci(member_values, seed, samples),
        "auc": roc_auc_score(labels, scores),
        "auc_ci": stratified_bootstrap_ci(
            member_values, nonmember_values, auc_metric, seed, samples
        ),
        "tpr_at_1pct_fpr": tpr_at_fpr(member_values, nonmember_values, 0.01),
        "tpr_at_1pct_fpr_ci": stratified_bootstrap_ci(
            member_values, nonmember_values, tpr_metric, seed, samples
        ),
    }


def analyze_checkpoint(
    mia_path: Path, rr_dir: Path, seed: int, samples: int
) -> Dict[str, Any]:
    mia = json.loads(mia_path.read_text(encoding="utf-8"))
    cell = CELL_RE.search(str(mia_path))
    pct = PCT_RE.search(str(mia_path))
    if not cell or not pct:
        raise ValueError(f"cannot parse arm/checkpoint from path: {mia_path}")
    eps = float(cell.group("eps"))
    rr_path = rr_dir / f"train_rr_eps{eps}_seed{cell.group('seed')}.jsonl"
    if not rr_path.is_file():
        raise FileNotFoundError(f"RR trainer file missing, cannot form the split: {rr_path}")
    flips = load_flip_map(rr_path)

    members = mia["member_scores"]
    nonmembers = mia["nonmember_scores"]
    missing = [r["pair_key"] for r in members if r["pair_key"] not in flips]
    if missing:
        raise KeyError(
            f"{len(missing)}/{len(members)} member pair_keys absent from {rr_path.name}. "
            "The split would be computed on a subset, which is not reportable. "
            f"First missing: {missing[0]}"
        )

    flagged = [r for r in members if flips[r["pair_key"]]]
    unflagged = [r for r in members if not flips[r["pair_key"]]]
    if not flagged or not unflagged:
        raise ValueError(
            f"eps={eps}: one group is empty (flipped={len(flagged)}, unflipped={len(unflagged)}); "
            "no within-model comparison is possible"
        )

    out: Dict[str, Any] = {
        "mia_json": str(mia_path),
        "eps": eps,
        "epochs": int(cell.group("epochs")),
        "lora_r": int(cell.group("lora")),
        "seed": int(cell.group("seed")),
        "percent": int(pct.group("pct")),
        "key": f"rr_eps{eps}|e{cell.group('epochs')}|r{cell.group('lora')}|p{pct.group('pct')}",
        "rr_trainer_file": str(rr_path.relative_to(ROOT)),
        "join": {
            "pair_key": "preference_pair_key (orientation-independent)",
            "n_members": len(members),
            "coverage": 1.0,
            "n_flipped": len(flagged),
            "n_unflipped": len(unflagged),
            "observed_flip_rate_in_member_sample": len(flagged) / len(members),
        },
        "bootstrap": {"samples": samples, "seed": seed},
        "statistics": {},
    }

    for stat in STATISTICS:
        non_values = [r[stat] for r in nonmembers]
        flip_values = [r[stat] for r in flagged]
        unflip_values = [r[stat] for r in unflagged]
        all_values = [r[stat] for r in members]
        block = {
            "family": "magnitude" if stat == MAGNITUDE_STATISTIC else "signed",
            "flipped": group_block(flip_values, non_values, seed, samples),
            "unflipped": group_block(unflip_values, non_values, seed, samples),
            "all_members": group_block(all_values, non_values, seed, samples),
            "mean_nonmember": sum(non_values) / len(non_values),
            "flipped_minus_unflipped_mean": (
                sum(flip_values) / len(flip_values) - sum(unflip_values) / len(unflip_values)
            ),
            "flipped_minus_unflipped_mean_ci": bootstrap_diff_ci(
                flip_values, unflip_values, seed, samples
            ),
        }
        out["statistics"][stat] = block
    return out


def render_markdown(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# Stage 2 RR — magnitude statistic by `was_flipped`, within model",
        "",
        "Members split by the persisted `rr_flipped` flag and scored against the SAME non-member",
        "pool at the SAME checkpoint, so fit is held exactly constant. `|s|` is the magnitude",
        f"statistic `{MAGNITUDE_STATISTIC}`. AUC and TPR@1%FPR only; the count of non-members",
        "defining the 1% threshold is printed beside every TPR. Reported, not interpreted.",
        "",
    ]
    for res in results:
        st = res["statistics"][MAGNITUDE_STATISTIC]
        j = res["join"]
        lines += [
            f"## `{res['key']}`  (eps={res['eps']}, epochs={res['epochs']}, {res['percent']}%)",
            "",
            f"Members {j['n_members']} = {j['n_flipped']} flipped + {j['n_unflipped']} unflipped "
            f"(observed flip rate {j['observed_flip_rate_in_member_sample']:.4f}). "
            f"Non-members {st['flipped']['n_nonmembers']}; "
            f"{st['flipped']['n_nonmembers_defining_fpr_1pct']} define FPR=1%. "
            f"Bootstrap {res['bootstrap']['samples']}, seed {res['bootstrap']['seed']}.",
            "",
            "| member group | n | mean \\|s\\| | 95% CI | AUC | 95% CI | TPR@1%FPR | 95% CI |",
            "|---|---:|---:|---|---:|---|---:|---|",
        ]
        for label in ("flipped", "unflipped", "all_members"):
            b = st[label]
            lines.append(
                f"| {label} | {b['n_members']} | {b['mean']:.4f} | "
                f"[{b['mean_ci'][0]:.4f}, {b['mean_ci'][1]:.4f}] | {b['auc']:.4f} | "
                f"[{b['auc_ci'][0]:.4f}, {b['auc_ci'][1]:.4f}] | {b['tpr_at_1pct_fpr']:.4f} | "
                f"[{b['tpr_at_1pct_fpr_ci'][0]:.4f}, {b['tpr_at_1pct_fpr_ci'][1]:.4f}] |"
            )
        d, dci = st["flipped_minus_unflipped_mean"], st["flipped_minus_unflipped_mean_ci"]
        crosses = dci[0] <= 0.0 <= dci[1]
        lines += [
            f"| non-members | {st['flipped']['n_nonmembers']} | {st['mean_nonmember']:.4f} "
            "| — | — | — | — | — |",
            "",
            f"mean \\|s\\| flipped − unflipped = **{d:+.4f}** "
            f"[{dci[0]:+.4f}, {dci[1]:+.4f}] — CI "
            f"{'includes' if crosses else 'excludes'} zero.",
            "",
        ]
        sg = res["statistics"][SIGNED_STATISTIC]
        lines += [
            "Signed statistic at the same checkpoint, for context: "
            f"flipped AUC {sg['flipped']['auc']:.4f} "
            f"[{sg['flipped']['auc_ci'][0]:.4f}, {sg['flipped']['auc_ci'][1]:.4f}], "
            f"unflipped AUC {sg['unflipped']['auc']:.4f} "
            f"[{sg['unflipped']['auc_ci'][0]:.4f}, {sg['unflipped']['auc_ci'][1]:.4f}].",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mia-json", action="append", default=[],
                    help="Path to a checkpoint's mia.json. Repeatable.")
    ap.add_argument("--scan-root", default=None,
                    help="Directory to scan recursively for */eval/mia.json.")
    ap.add_argument("--rr-dir", default=str(ROOT / "data/stage2_rr"))
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--bootstrap-samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = [Path(p) for p in args.mia_json]
    if args.scan_root:
        paths += sorted(Path(args.scan_root).rglob("eval/mia.json"))
    seen, ordered = set(), []
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            ordered.append(p)
    if not ordered:
        raise SystemExit("no mia.json given; pass --mia-json and/or --scan-root")

    results = []
    for p in ordered:
        print(f"[flip-split] {p}", flush=True)
        results.append(
            analyze_checkpoint(p, Path(args.rr_dir), args.seed, args.bootstrap_samples)
        )
    results.sort(key=lambda r: (r["eps"], r["epochs"], r["percent"]))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[flip-split] wrote {args.out_json}")
    md = render_markdown(results)
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")
        print(f"[flip-split] wrote {args.out_md}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
