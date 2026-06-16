#!/usr/bin/env python3
import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aggregate
import config


BASELINE_ALIASES = {
    "rdpo": {"rdpo"},
    "redpo": {"redpo", "re-dpo", "re_dpo"},
    "mle": {"mle", "mle_dpo", "mle-dpo"},
    "map": {"map", "props_map", "map_retrain", "map-retrain"},
}


def norm_method(method: str) -> str:
    return method.strip().lower().replace("-", "_")


def method_matches(method: str, aliases: Iterable[str]) -> bool:
    m = norm_method(method)
    normalized = {norm_method(alias) for alias in aliases}
    return m in normalized


def find_summary(
    summaries: Sequence[Dict[str, Any]],
    dataset: str,
    eps: float,
    aliases: Iterable[str],
    require_results: bool = True,
) -> Optional[Dict[str, Any]]:
    matches = [
        row
        for row in summaries
        if row["dataset"] == dataset
        and float(row["eps"]) == float(eps)
        and method_matches(row["method"], aliases)
        and (row["n_results"] > 0 or not require_results)
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: row["n_results"])


def find_manifest_summary(
    summaries: Sequence[Dict[str, Any]],
    manifest_id: str,
    dataset: str,
    eps: float,
    require_results: bool = True,
) -> Optional[Dict[str, Any]]:
    matches = [
        row
        for row in summaries
        if row["manifest_id"] == manifest_id
        and row["dataset"] == dataset
        and float(row["eps"]) == float(eps)
        and (row["n_results"] > 0 or not require_results)
    ]
    if not matches:
        return None
    return max(matches, key=lambda row: row["n_results"])


def utility_within_ci(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    # Candidate is not clearly below baseline if its upper CI reaches the
    # baseline lower CI. This is intentionally conservative with few seeds.
    return float(candidate["acc_ci"][1]) >= float(baseline["acc_ci"][0])


def mia_strictly_lower(candidate: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    return float(candidate["mia_auc_mean"]) < float(baseline["mia_auc_mean"])


def yes_no(value: Optional[bool]) -> str:
    if value is None:
        return "Insufficient data"
    return "Yes" if value else "No"


def fmt(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def metric_cell(row: Optional[Dict[str, Any]]) -> str:
    if row is None or row.get("n_results", 0) == 0:
        return "NA"
    return f"acc {fmt(row['acc_mean'])} [{fmt(row['acc_ci'][0])}, {fmt(row['acc_ci'][1])}], MIA {fmt(row['mia_auc_mean'])}"


def build_limo_gate_rows(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    limo_expected = [
        row
        for row in summaries
        if method_matches(row["method"], {"limo"}) or row["manifest_id"] == "e2_limo_matched"
    ]
    rows = []
    for expected in sorted(limo_expected, key=lambda row: (row["dataset"], float(row["eps"]), row["manifest_id"])):
            dataset = expected["dataset"]
            eps = float(expected["eps"])
            limo = expected if expected["n_results"] > 0 else None
            baseline_results = {}
            baseline_decisions = {}
            for label, aliases in BASELINE_ALIASES.items():
                baseline = find_summary(summaries, dataset, eps, aliases)
                baseline_results[label] = baseline
                if baseline is None or limo is None:
                    baseline_decisions[label] = None
                else:
                    baseline_decisions[label] = utility_within_ci(limo, baseline) and mia_strictly_lower(limo, baseline)
            available = [v for v in baseline_decisions.values() if v is not None]
            overall = None if not available else all(available)
            rows.append(
                {
                    "dataset": dataset,
                    "eps": eps,
                    "limo": limo or expected,
                    "baselines": baseline_results,
                    "decisions": baseline_decisions,
                    "overall": overall,
                }
            )
    return rows


def build_e2_rows(summaries: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    e2_expected = [
        row
        for row in summaries
        if row["manifest_id"] in {"e2_limo_matched", "e2_random"}
    ]
    pairs = sorted({(row["dataset"], float(row["eps"])) for row in e2_expected})
    detail = []
    by_dataset_raw: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: {"utility": [], "mia": []})
    for dataset, eps in pairs:
        limo_any = find_manifest_summary(summaries, "e2_limo_matched", dataset, eps, require_results=False)
        random_any = find_manifest_summary(summaries, "e2_random", dataset, eps, require_results=False)
        limo = limo_any if limo_any and limo_any["n_results"] > 0 else None
        random_subset = random_any if random_any and random_any["n_results"] > 0 else None
        utility_win = None
        mia_win = None
        if limo is not None and random_subset is not None:
            utility_win = float(limo["acc_mean"]) > float(random_subset["acc_mean"])
            mia_win = float(limo["mia_auc_mean"]) < float(random_subset["mia_auc_mean"])
            by_dataset_raw[dataset]["utility"].append(utility_win)
            by_dataset_raw[dataset]["mia"].append(mia_win)
        detail.append(
            {
                "dataset": dataset,
                "eps": eps,
                "limo": limo or limo_any,
                "random": random_subset or random_any,
                "utility_win": utility_win,
                "mia_win": mia_win,
            }
        )
    by_dataset = []
    for dataset in sorted({dataset for dataset, _eps in pairs}):
        dataset_utility = by_dataset_raw[dataset]["utility"]
        dataset_mia = by_dataset_raw[dataset]["mia"]
        by_dataset.append(
            {
                "dataset": dataset,
                "utility_win_all_eps": all(dataset_utility) if dataset_utility else None,
                "mia_win_all_eps": all(dataset_mia) if dataset_mia else None,
                "n_eps": len(dataset_utility),
            }
        )
    return detail, by_dataset


def detect_anomalies(
    summaries: Sequence[Dict[str, Any]],
    missing: Sequence[Tuple[str, str, str, float, int]],
    stale: Sequence[Dict[str, Any]],
) -> List[str]:
    notes = []
    for row in summaries:
        cell = aggregate.display_group(row["key"])
        if row["n_results"] == 0:
            continue
        if float(row["mia_auc_mean"]) < 0.5:
            notes.append(f"`{cell}` has inverted-MIA signal: mean MIA AUC {fmt(row['mia_auc_mean'])} < 0.5.")
        if float(row["mia_auc_mean"]) > 0.9:
            notes.append(f"`{cell}` has very high MIA AUC: {fmt(row['mia_auc_mean'])}.")
        if row["acc_ci"][1] - row["acc_ci"][0] > 0.20:
            notes.append(f"`{cell}` has wide utility CI: [{fmt(row['acc_ci'][0])}, {fmt(row['acc_ci'][1])}].")
        if row["mia_auc_ci"][1] - row["mia_auc_ci"][0] > 0.20:
            notes.append(f"`{cell}` has wide MIA CI: [{fmt(row['mia_auc_ci'][0])}, {fmt(row['mia_auc_ci'][1])}].")
    if missing:
        notes.append(f"`{len(missing)}` manifest seed cells are missing.")
    for item in stale:
        notes.append(f"`{item['cell']}` is stale: {', '.join(item['reasons'])}.")
    return notes


def write_report(
    path: Path,
    manifest_path: Path,
    result_roots: Sequence[Path],
    limo_rows: Sequence[Dict[str, Any]],
    e2_detail: Sequence[Dict[str, Any]],
    e2_by_dataset: Sequence[Dict[str, Any]],
    anomalies: Sequence[str],
    counts: Dict[str, int],
    ledger_path: Path,
    spend_summary: Tuple[float, float, int],
) -> None:
    lines = [
        "# Decision Report",
        "",
        f"Generated: `{aggregate.utc_now()}`",
        f"Manifest: `{manifest_path}`",
        f"Result roots: `{', '.join(str(p) for p in result_roots)}`",
        "",
        "## Gate 1: LIMO Pareto-Competitive",
        "",
        "Question: per `(dataset, eps)`, is LIMO Pareto-competitive vs rDPO / RE-DPO / MLE / MAP?",
        "",
        "Rule: `Yes` only when LIMO utility is within the baseline bootstrap CI and LIMO mean MIA AUC is strictly lower.",
        "",
        "| Dataset | Eps | Overall | vs rDPO | vs RE-DPO | vs MLE | vs MAP | LIMO metrics |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    if limo_rows:
        for row in limo_rows:
            decisions = row["decisions"]
            lines.append(
                "| {dataset} | {eps} | {overall} | {rdpo} | {redpo} | {mle} | {map_} | {metrics} |".format(
                    dataset=row["dataset"],
                    eps=aggregate.slug_eps(row["eps"]),
                    overall=yes_no(row["overall"]),
                    rdpo=yes_no(decisions["rdpo"]),
                    redpo=yes_no(decisions["redpo"]),
                    mle=yes_no(decisions["mle"]),
                    map_=yes_no(decisions["map"]),
                    metrics=metric_cell(row["limo"]),
                )
            )
    else:
        lines.append("| NA | NA | Insufficient data | Insufficient data | Insufficient data | Insufficient data | Insufficient data | No LIMO results found |")

    lines.extend(
        [
            "",
            "## Gate 2: E2 LIMO vs Random Subset",
            "",
            "Question: does private-scorer LIMO beat random subset at matched keep on utility and/or MIA?",
            "",
            "### Per Dataset",
            "",
            "| Dataset | Eps covered | Utility win all eps | MIA win all eps |",
            "| --- | ---: | --- | --- |",
        ]
    )
    if e2_by_dataset:
        for row in e2_by_dataset:
            lines.append(
                f"| {row['dataset']} | {row['n_eps']} | {yes_no(row['utility_win_all_eps'])} | {yes_no(row['mia_win_all_eps'])} |"
            )
    else:
        lines.append("| NA | 0 | Insufficient data | Insufficient data |")

    lines.extend(
        [
            "",
            "### Per Dataset/Eps Detail",
            "",
            "| Dataset | Eps | Utility win | MIA win | LIMO | Random subset |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    if e2_detail:
        for row in e2_detail:
            lines.append(
                "| {dataset} | {eps} | {utility} | {mia} | {limo} | {random} |".format(
                    dataset=row["dataset"],
                    eps=aggregate.slug_eps(row["eps"]),
                    utility=yes_no(row["utility_win"]),
                    mia=yes_no(row["mia_win"]),
                    limo=metric_cell(row["limo"]),
                    random=metric_cell(row["random"]),
                )
            )
    else:
        lines.append("| NA | NA | Insufficient data | Insufficient data | NA | NA |")

    lines.extend(
        [
            "",
            "## Anomalies",
            "",
        ]
    )
    if anomalies:
        for note in anomalies:
            lines.append(f"- {note}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Cost",
            "",
        ]
    )
    total_cost, total_hours, ledger_rows = spend_summary
    lines.append(f"- Ledger: `{ledger_path}`")
    lines.append(f"- Ledger rows: `{ledger_rows}`")
    lines.append(f"- Estimated GPU-hours: `{total_hours:.4f}`")
    lines.append(f"- Estimated spend: `${total_cost:.2f}`")
    lines.append(f"- Credit ceiling: `${config.CREDIT_CEILING_USD:.2f}`")
    lines.append(f"- Estimated remaining credit: `${max(0.0, config.CREDIT_CEILING_USD - total_cost):.2f}`")

    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Valid result JSONs: `{counts['valid_results']}`",
            f"- Missing manifest seed cells: `{counts['missing']}`",
            f"- Stale cells: `{counts['stale']}`",
            f"- Extra valid result JSONs outside manifest: `{counts['extras']}`",
            f"- Invalid JSONs skipped: `{counts['invalid']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = aggregate.repo_root()
    ap = argparse.ArgumentParser(description="Emit DECISION_REPORT.md from PDPO result JSONs.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--result-root", action="append", default=[str(root / "experiments/run_queue")])
    ap.add_argument("--out-md", default=str(root / "DECISION_REPORT.md"))
    ap.add_argument("--ledger", default=str(root / "cost_ledger.csv"))
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = aggregate.repo_root()
    manifest_path = Path(args.manifest)
    result_roots = [Path(p) for p in args.result_root]
    manifest = aggregate.load_manifest(manifest_path)
    expected_cells = aggregate.expand_manifest(manifest)
    expected_seed_keys = {aggregate.expected_seed_key(cell) for cell in expected_cells}
    expected_groups = {(mid, method, dataset, eps) for (mid, method, dataset, eps, _seed) in expected_seed_keys}
    expected_by_seed = {aggregate.expected_seed_key(cell): cell for cell in expected_cells}

    results, invalid = aggregate.scan_results(result_roots)
    current_sha = aggregate.git_sha(root)
    manifest_mtime = aggregate.datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=aggregate.timezone.utc)

    valid_by_seed: Dict[Tuple[str, str, str, float, int], Dict[str, Any]] = {}
    extras = []
    stale = []
    grouped: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]] = defaultdict(list)

    for result in results:
        skey = aggregate.seed_key_from_result(result)
        if skey in expected_seed_keys:
            previous = valid_by_seed.get(skey)
            if previous is None or str(result.get("timestamp", "")) > str(previous.get("timestamp", "")):
                valid_by_seed[skey] = result
        else:
            extras.append(result)

    for skey, result in sorted(valid_by_seed.items()):
        grouped[aggregate.group_key_from_result(result)].append(result)
        reasons = []
        if current_sha != "unknown" and result.get("git_sha") != current_sha:
            reasons.append(f"git_sha {result.get('git_sha')} != current {current_sha}")
        result_time = aggregate.parse_timestamp(str(result.get("timestamp", "")))
        if result_time is None:
            reasons.append("unparseable timestamp")
        elif result_time < manifest_mtime:
            reasons.append("older than manifest")
        if reasons:
            stale.append({"cell": aggregate.cell_name_from_seed_key(skey), "path": result.get("_path"), "reasons": reasons})

    missing = sorted(key for key in expected_by_seed if key not in valid_by_seed)
    summaries = aggregate.summarize_groups(expected_groups, grouped, args.bootstrap_reps, args.bootstrap_seed)
    limo_rows = build_limo_gate_rows(summaries)
    e2_detail, e2_by_dataset = build_e2_rows(summaries)
    anomalies = detect_anomalies(summaries, missing, stale)
    write_report(
        Path(args.out_md),
        manifest_path,
        result_roots,
        limo_rows,
        e2_detail,
        e2_by_dataset,
        anomalies,
        {
            "valid_results": len(results),
            "missing": len(missing),
            "stale": len(stale),
            "extras": len(extras),
            "invalid": len(invalid),
        },
        Path(args.ledger),
        aggregate.ledger_spend(Path(args.ledger)),
    )
    print(args.out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
