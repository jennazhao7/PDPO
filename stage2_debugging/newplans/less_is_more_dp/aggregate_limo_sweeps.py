#!/usr/bin/env python3
"""Aggregate LIMO-DP sweep outputs into CSV and Markdown tables."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_eval(root: Path, model_dir: Path, stats: Dict[str, Any]) -> Optional[Path]:
    eval_dir = root / "eval"
    if not eval_dir.exists():
        return None
    dataset = infer_dataset(model_dir)
    mode = stats.get("mode")
    keep_fraction = stats.get("requested_keep_fraction")
    candidates = list(eval_dir.glob("*.json"))
    preferred: List[Path] = []
    for path in candidates:
        name = path.name
        if dataset and dataset not in name:
            continue
        if mode == "selection_plus_weighting" and "both" in model_dir.parts and "both" not in name:
            continue
        if isinstance(keep_fraction, float):
            tag = f"keep{str(keep_fraction).replace('.', 'p')}"
            if tag in name:
                preferred.append(path)
        if model_dir.name in name or (mode and str(mode).replace("selection_plus_weighting", "both") in name):
            preferred.append(path)
    if preferred:
        return sorted(set(preferred), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    if dataset:
        dataset_matches = [p for p in candidates if dataset in p.name]
        if dataset_matches:
            return sorted(dataset_matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    return None


def infer_dataset(model_dir: Path) -> str:
    for part in model_dir.parts:
        if part in {"truthy", "hhrlhf", "pku"}:
            return part
    name = model_dir.as_posix()
    for ds in ("truthy", "hhrlhf", "pku"):
        if ds in name:
            return ds
    return ""


def extract_accuracy(eval_json: Optional[Path]) -> Dict[str, Any]:
    if eval_json is None or not eval_json.exists():
        return {"eval_json": "", "accuracy": "", "n_eval": ""}
    data = load_json(eval_json)
    accuracy = data.get("accuracy", data.get("acc", data.get("preference_accuracy", "")))
    n_eval = data.get("n", data.get("n_eval", data.get("num_examples", "")))
    return {"eval_json": str(eval_json), "accuracy": accuracy, "n_eval": n_eval}


def collect_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for stats_path in sorted(root.rglob("selection_stats.json")):
        model_dir = stats_path.parent
        stats = load_json(stats_path)
        manifest_path = model_dir / "M2_manifest.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        params = manifest.get("params", {})
        eval_path = find_eval(root, model_dir, stats)
        eval_fields = extract_accuracy(eval_path)
        rows.append(
            {
                "dataset": infer_dataset(model_dir),
                "sweep_root": str(root),
                "model_dir": str(model_dir),
                "mode": stats.get("mode", params.get("selection_mode", "")),
                "selection_rule": stats.get("selection_rule", params.get("selection_rule", "")),
                "epsilon": stats.get("epsilon", params.get("epsilon", "")),
                "tau": stats.get("tau", params.get("tau", "")),
                "tau_drop": stats.get("tau_drop", params.get("tau_drop", "")),
                "requested_keep_fraction": stats.get("requested_keep_fraction", params.get("keep_fraction", "")),
                "actual_keep_fraction": stats.get("keep_fraction", ""),
                "N_total": stats.get("N_total", ""),
                "N_kept": stats.get("N_kept", ""),
                "q_mean_all": stats.get("q_mean_all", ""),
                "q_mean_kept": stats.get("q_mean_kept", ""),
                "w_mean_all": stats.get("w_mean_all", ""),
                "w_mean_kept": stats.get("w_mean_kept", ""),
                "accuracy": eval_fields["accuracy"],
                "n_eval": eval_fields["n_eval"],
                "eval_json": eval_fields["eval_json"],
                "selection_jsonl": stats.get("selection_jsonl", ""),
                "score_cache_in": stats.get("score_cache_in", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return "" if value is None else str(value)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset",
        "mode",
        "epsilon",
        "requested_keep_fraction",
        "actual_keep_fraction",
        "N_kept",
        "accuracy",
        "tau",
        "selection_rule",
    ]
    lines = ["# LIMO-DP Sweep Summary", ""]
    if not rows:
        lines.append("No `selection_stats.json` files found.")
    else:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in cols) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def unique_roots(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    roots: List[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            roots.append(path)
            seen.add(resolved)
    return roots


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Aggregate LIMO-DP sweep outputs")
    ap.add_argument("--root", action="append", default=[], help="Sweep root to scan; can be repeated.")
    ap.add_argument(
        "--default_root",
        default="stage2_debugging/newplans/less_is_more_dp/sweeps",
        help="Root used when --root is omitted.",
    )
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_md", default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    roots = unique_roots(Path(p) for p in (args.root or [args.default_root]))
    all_rows: List[Dict[str, Any]] = []
    for root in roots:
        all_rows.extend(collect_rows(root))
    all_rows.sort(key=lambda r: (str(r["dataset"]), float(r["epsilon"] or 0), str(r["mode"]), str(r["model_dir"])))

    out_base = roots[0] if len(roots) == 1 else Path(args.default_root)
    out_csv = Path(args.out_csv) if args.out_csv else out_base / "limo_sweep_summary.csv"
    out_md = Path(args.out_md) if args.out_md else out_base / "limo_sweep_summary.md"
    write_csv(out_csv, all_rows)
    write_markdown(out_md, all_rows)
    print(f"[Aggregate] rows={len(all_rows)}")
    print(f"[Aggregate] wrote {out_csv}")
    print(f"[Aggregate] wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
