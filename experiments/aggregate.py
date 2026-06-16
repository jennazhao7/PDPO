#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORTS))

import config


REQUIRED_RESULT_FIELDS = {
    "method": str,
    "dataset": str,
    "eps": (int, float),
    "seed": int,
    "n": int,
    "acc": (int, float),
    "mia_auc": (int, float),
    "git_sha": str,
    "timestamp": str,
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def ledger_spend(path: Path) -> Tuple[float, float, int]:
    if not path.exists():
        return 0.0, 0.0, 0
    total_cost = 0.0
    total_hours = 0.0
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows += 1
            try:
                total_cost += float(row.get("est_cost", 0.0) or 0.0)
                total_hours += float(row.get("gpu_hours", 0.0) or 0.0)
            except ValueError:
                continue
    return total_cost, total_hours, rows


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        out: Dict[str, Any] = {}
        if not inner:
            return out
        for part in inner.split(","):
            key, raw = part.split(":", 1)
            out[key.strip()] = parse_scalar(raw.strip())
        return out
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Manifest must be a mapping: {path}")
        return data
    except ImportError:
        return load_manifest_minimal_yaml(path)


def load_manifest_minimal_yaml(path: Path) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    section: Optional[str] = None
    current_item: Optional[Dict[str, Any]] = None
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith(" "):
                key, value = stripped.split(":", 1)
                section = key.strip()
                data[section] = parse_scalar(value.strip()) if value.strip() else ([] if section == "queue" else {})
                current_item = None
                continue
            if section == "queue" and stripped.startswith("- "):
                current_item = {}
                data.setdefault("queue", []).append(current_item)
                rest = stripped[2:].strip()
                if rest:
                    key, value = rest.split(":", 1)
                    current_item[key.strip()] = parse_scalar(value.strip())
                continue
            key, value = stripped.split(":", 1)
            target = current_item if current_item is not None else data.setdefault(section or "", {})
            target[key.strip()] = parse_scalar(value.strip())
    return data


def expand_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    defaults = manifest.get("defaults", {}) or {}
    cells: List[Dict[str, Any]] = []
    for item in manifest.get("queue", []) or []:
        seeds = item.get("seeds", defaults.get("seeds", [42]))
        for dataset, eps, seed in itertools.product(item["datasets"], item["eps"], seeds):
            cell = dict(defaults)
            cell.update(item)
            cell.update({"dataset": dataset, "eps": float(eps), "seed": int(seed)})
            cells.append(cell)
    return cells


def slug_eps(eps: float) -> str:
    return str(eps).rstrip("0").rstrip(".") if "." in str(eps) else str(eps)


def expected_seed_key(cell: Dict[str, Any]) -> Tuple[str, str, str, float, int]:
    return (str(cell["id"]), str(cell["method"]), str(cell["dataset"]), float(cell["eps"]), int(cell["seed"]))


def group_key_from_result(result: Dict[str, Any]) -> Tuple[str, str, str, float]:
    manifest_id = str(result.get("manifest_id") or result.get("id") or result["method"])
    return (manifest_id, str(result["method"]), str(result["dataset"]), float(result["eps"]))


def seed_key_from_result(result: Dict[str, Any]) -> Tuple[str, str, str, float, int]:
    gid, method, dataset, eps = group_key_from_result(result)
    return (gid, method, dataset, eps, int(result["seed"]))


def display_group(key: Tuple[str, str, str, float]) -> str:
    manifest_id, method, dataset, eps = key
    return f"{manifest_id}/{method}/{dataset}/eps{slug_eps(eps)}"


def validate_result(obj: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RESULT_FIELDS if key not in obj]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for key, expected in REQUIRED_RESULT_FIELDS.items():
        if not isinstance(obj[key], expected):
            raise ValueError(f"{key} has invalid type {type(obj[key]).__name__}")
    if int(obj["n"]) < 0:
        raise ValueError("n must be nonnegative")
    for key in ("acc", "mia_auc"):
        value = float(obj[key])
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{key} must be in [0, 1]")


def load_result(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            return None, "not a JSON object"
        validate_result(obj)
        obj["_path"] = str(path)
        return obj, None
    except Exception as exc:
        return None, str(exc)


def scan_results(result_roots: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str]]]:
    results: List[Dict[str, Any]] = []
    invalid: List[Tuple[str, str]] = []
    seen_paths = set()
    for root in result_roots:
        if root.is_file():
            paths = [root]
        elif root.exists():
            paths = sorted(root.rglob("*.json"))
        else:
            continue
        for path in paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            result, error = load_result(path)
            if result is None:
                invalid.append((str(path), error or "invalid"))
            else:
                results.append(result)
    return results, invalid


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def bootstrap_ci(values: Sequence[float], reps: int, rng: random.Random) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    boot = []
    n = len(values)
    for _ in range(reps):
        boot.append(mean([values[rng.randrange(n)] for _ in range(n)]))
    return percentile(boot, 0.025), percentile(boot, 0.975)


def parse_timestamp(value: str) -> Optional[datetime]:
    try:
        fixed = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(fixed)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def summarize_groups(
    expected_groups: Iterable[Tuple[str, str, str, float]],
    grouped: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]],
    reps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    summaries = []
    for key in sorted(expected_groups):
        rows = sorted(grouped.get(key, []), key=lambda r: int(r["seed"]))
        accs = [float(r["acc"]) for r in rows]
        mias = [float(r["mia_auc"]) for r in rows]
        acc_lo, acc_hi = bootstrap_ci(accs, reps, rng)
        mia_lo, mia_hi = bootstrap_ci(mias, reps, rng)
        manifest_id, method, dataset, eps = key
        summaries.append(
            {
                "key": key,
                "manifest_id": manifest_id,
                "method": method,
                "dataset": dataset,
                "eps": eps,
                "seeds": [int(r["seed"]) for r in rows],
                "n_results": len(rows),
                "acc_mean": mean(accs),
                "acc_ci": (acc_lo, acc_hi),
                "mia_auc_mean": mean(mias),
                "mia_auc_ci": (mia_lo, mia_hi),
                "n_total": sum(int(r["n"]) for r in rows),
            }
        )
    return summaries


def fmt_float(value: float, digits: int = 4) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.{digits}f}"


def fmt_ci(mean_value: float, ci: Tuple[float, float]) -> str:
    return f"{fmt_float(mean_value)} [{fmt_float(ci[0])}, {fmt_float(ci[1])}]"


def cell_name_from_seed_key(key: Tuple[str, str, str, float, int]) -> str:
    manifest_id, method, dataset, eps, seed = key
    return f"{manifest_id}/{method}/{dataset}/eps{slug_eps(eps)}/seed{seed}"


def stable_result_id(result: Dict[str, Any]) -> str:
    text = json.dumps(
        {
            "method": result["method"],
            "dataset": result["dataset"],
            "eps": result["eps"],
            "seed": result["seed"],
            "path": result.get("_path", ""),
        },
        sort_keys=True,
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def write_markdown(
    path: Path,
    manifest_path: Path,
    result_roots: Sequence[Path],
    expected_count: int,
    ledger_path: Path,
    spend_summary: Tuple[float, float, int],
    summaries: List[Dict[str, Any]],
    missing: List[Tuple[str, str, str, float, int]],
    stale: List[Dict[str, Any]],
    extras: List[Dict[str, Any]],
    invalid: List[Tuple[str, str]],
    current_sha: str,
    plot_paths: Sequence[Path],
) -> None:
    lines = [
        "# Experiment Mega Summary",
        "",
        f"Generated: `{utc_now()}`",
        f"Manifest: `{manifest_path}`",
        f"Result roots: `{', '.join(str(p) for p in result_roots)}`",
        f"Current git SHA: `{current_sha}`",
        "",
        "## Seed-Aggregated Results",
        "",
        "| Cell | Seeds | N total | Accuracy 95% CI | MIA AUC 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {cell} | {seeds} | {n_total} | {acc} | {mia} |".format(
                cell=display_group(row["key"]),
                seeds=",".join(str(s) for s in row["seeds"]) if row["seeds"] else "missing",
                n_total=row["n_total"],
                acc=fmt_ci(row["acc_mean"], row["acc_ci"]),
                mia=fmt_ci(row["mia_auc_mean"], row["mia_auc_ci"]),
            )
        )

    lines.extend(["", "## Manifest Reconciliation", ""])
    lines.append(f"- Expected seed cells: `{expected_count}`")
    lines.append(f"- Valid result JSONs found: `{sum(row['n_results'] for row in summaries) + len(extras)}`")
    lines.append(f"- Missing manifest cells: `{len(missing)}`")
    lines.append(f"- Stale result JSONs: `{len(stale)}`")
    lines.append(f"- Extra valid result JSONs outside manifest: `{len(extras)}`")
    lines.append(f"- Invalid JSONs skipped: `{len(invalid)}`")

    total_cost, total_hours, ledger_rows = spend_summary
    lines.extend(["", "## Cost", ""])
    lines.append(f"- Ledger: `{ledger_path}`")
    lines.append(f"- Ledger rows: `{ledger_rows}`")
    lines.append(f"- Estimated GPU-hours: `{total_hours:.4f}`")
    lines.append(f"- Estimated spend: `${total_cost:.2f}`")
    lines.append(f"- Credit ceiling: `${config.CREDIT_CEILING_USD:.2f}`")
    lines.append(f"- Estimated remaining credit: `${max(0.0, config.CREDIT_CEILING_USD - total_cost):.2f}`")

    lines.extend(["", "### Missing Cells", ""])
    if missing:
        for key in missing:
            lines.append(f"- `{cell_name_from_seed_key(key)}`")
    else:
        lines.append("- None")

    lines.extend(["", "### Stale Cells", ""])
    if stale:
        for item in stale:
            reasons = ", ".join(item["reasons"])
            lines.append(f"- `{item['cell']}`: {reasons} at `{item['path']}`")
    else:
        lines.append("- None")

    lines.extend(["", "### Extra Valid Results", ""])
    if extras:
        for item in extras:
            lines.append(f"- `{cell_name_from_seed_key(seed_key_from_result(item))}` at `{item.get('_path')}`")
    else:
        lines.append("- None")

    lines.extend(["", "### Invalid JSONs", ""])
    if invalid:
        for invalid_path, error in invalid[:100]:
            lines.append(f"- `{invalid_path}`: {error}")
        if len(invalid) > 100:
            lines.append(f"- ... and {len(invalid) - 100} more")
    else:
        lines.append("- None")

    lines.extend(["", "## Privacy-Utility Frontiers", ""])
    if plot_paths:
        for plot_path in plot_paths:
            try:
                target = plot_path.relative_to(path.parent)
            except ValueError:
                target = plot_path
            lines.append(f"- [{plot_path.name}]({target.as_posix()})")
    else:
        lines.append("- No plots generated")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def color_for(label: str) -> str:
    palette = [
        "#1b9e77",
        "#d95f02",
        "#7570b3",
        "#e7298a",
        "#66a61e",
        "#e6ab02",
        "#a6761d",
        "#1f78b4",
    ]
    return palette[sum(ord(c) for c in label) % len(palette)]


def scale(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2.0
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def frontier(points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = sorted(points, key=lambda p: (p["mia_auc_mean"], -p["acc_mean"]))
    out = []
    best_acc = -1.0
    for point in ordered:
        if point["acc_mean"] >= best_acc:
            out.append(point)
            best_acc = point["acc_mean"]
    return out


def write_frontier_svg(dataset: str, points: List[Dict[str, Any]], out_path: Path) -> None:
    width, height = 900, 620
    left, right, top, bottom = 90, 40, 60, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    valid = [p for p in points if not math.isnan(p["acc_mean"]) and not math.isnan(p["mia_auc_mean"])]
    if valid:
        x_values = [p["mia_auc_mean"] for p in valid]
        y_values = [p["acc_mean"] for p in valid]
        x_lo = max(0.0, min(x_values) - 0.02)
        x_hi = min(1.0, max(x_values) + 0.02)
        y_lo = max(0.0, min(y_values) - 0.02)
        y_hi = min(1.0, max(y_values) + 0.02)
    else:
        x_lo, x_hi = 0.45, 0.55
        y_lo, y_hi = 0.0, 1.0

    def x(value: float) -> float:
        return scale(value, x_lo, x_hi, left, left + plot_w)

    def y(value: float) -> float:
        return scale(value, y_lo, y_hi, top + plot_h, top)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="32" font-family="Arial" font-size="22" font-weight="700">Privacy-utility frontier: {svg_escape(dataset)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="{left + plot_w / 2}" y="{height - 35}" text-anchor="middle" font-family="Arial" font-size="15">MIA AUC (lower is more private)</text>',
        f'<text x="22" y="{top + plot_h / 2}" transform="rotate(-90 22 {top + plot_h / 2})" text-anchor="middle" font-family="Arial" font-size="15">Accuracy (higher is better)</text>',
    ]
    for i in range(6):
        xv = x_lo + (x_hi - x_lo) * i / 5
        yv = y_lo + (y_hi - y_lo) * i / 5
        parts.append(f'<line x1="{x(xv):.1f}" y1="{top}" x2="{x(xv):.1f}" y2="{top + plot_h}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{x(xv):.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="Arial" font-size="12">{xv:.2f}</text>')
        parts.append(f'<line x1="{left}" y1="{y(yv):.1f}" x2="{left + plot_w}" y2="{y(yv):.1f}" stroke="#e8e8e8"/>')
        parts.append(f'<text x="{left - 12}" y="{y(yv) + 4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{yv:.2f}</text>')

    front = frontier(valid)
    if len(front) >= 2:
        coords = " ".join(f'{x(p["mia_auc_mean"]):.1f},{y(p["acc_mean"]):.1f}' for p in front)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="#111" stroke-width="2.5" stroke-dasharray="5 4"/>')

    for point in valid:
        label = f'{point["method"]} eps{slug_eps(point["eps"])}'
        color = color_for(point["method"])
        cx = x(point["mia_auc_mean"])
        cy = y(point["acc_mean"])
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="{color}" opacity="0.9"/>')
        parts.append(
            f'<text x="{cx + 9:.1f}" y="{cy - 8:.1f}" font-family="Arial" font-size="11" fill="#222">'
            f'{svg_escape(label)}</text>'
        )

    if not valid:
        parts.append(
            f'<text x="{left + plot_w / 2}" y="{top + plot_h / 2}" text-anchor="middle" '
            'font-family="Arial" font-size="16" fill="#555">No valid results yet</text>'
        )

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_frontier_plots(summaries: List[Dict[str, Any]], plot_dir: Path) -> List[Path]:
    by_dataset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        by_dataset[row["dataset"]].append(row)
    paths = []
    for dataset, rows in sorted(by_dataset.items()):
        path = plot_dir / f"privacy_utility_frontier_{dataset}.svg"
        write_frontier_svg(dataset, rows, path)
        paths.append(path)
    return paths


def parse_args() -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Aggregate PDPO result JSONs into markdown and frontier plots.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--result-root", action="append", default=[str(root / "experiments/run_queue")])
    ap.add_argument("--out-md", default=str(root / "EXPERIMENT_MEGA_SUMMARY.md"))
    ap.add_argument("--plot-dir", default=str(root / "experiments/frontiers"))
    ap.add_argument("--ledger", default=str(root / "cost_ledger.csv"))
    ap.add_argument("--bootstrap-reps", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=12345)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    manifest_path = Path(args.manifest)
    result_roots = [Path(p) for p in args.result_root]
    manifest = load_manifest(manifest_path)
    expected_cells = expand_manifest(manifest)
    expected_seed_keys = {expected_seed_key(cell) for cell in expected_cells}
    expected_groups = {(mid, method, dataset, eps) for (mid, method, dataset, eps, _seed) in expected_seed_keys}
    expected_by_seed = {expected_seed_key(cell): cell for cell in expected_cells}

    results, invalid = scan_results(result_roots)
    current_sha = git_sha(root)
    manifest_mtime = datetime.fromtimestamp(manifest_path.stat().st_mtime, tz=timezone.utc)

    grouped: Dict[Tuple[str, str, str, float], List[Dict[str, Any]]] = defaultdict(list)
    valid_by_seed: Dict[Tuple[str, str, str, float, int], Dict[str, Any]] = {}
    extras: List[Dict[str, Any]] = []
    stale: List[Dict[str, Any]] = []

    for result in results:
        skey = seed_key_from_result(result)
        if skey in expected_seed_keys:
            previous = valid_by_seed.get(skey)
            if previous is None or str(result.get("timestamp", "")) > str(previous.get("timestamp", "")):
                valid_by_seed[skey] = result
        else:
            extras.append(result)

    for skey, result in sorted(valid_by_seed.items()):
        grouped[group_key_from_result(result)].append(result)
        reasons = []
        if current_sha != "unknown" and result.get("git_sha") != current_sha:
            reasons.append(f"git_sha {result.get('git_sha')} != current {current_sha}")
        result_time = parse_timestamp(str(result.get("timestamp", "")))
        if result_time is None:
            reasons.append("unparseable timestamp")
        elif result_time < manifest_mtime:
            reasons.append("older than manifest")
        if reasons:
            stale.append({"cell": cell_name_from_seed_key(skey), "path": result.get("_path"), "reasons": reasons})

    missing = sorted(key for key in expected_by_seed if key not in valid_by_seed)
    summaries = summarize_groups(expected_groups, grouped, args.bootstrap_reps, args.bootstrap_seed)
    plot_paths = write_frontier_plots(summaries, Path(args.plot_dir))
    spend_summary = ledger_spend(Path(args.ledger))
    write_markdown(
        Path(args.out_md),
        manifest_path,
        result_roots,
        len(expected_seed_keys),
        Path(args.ledger),
        spend_summary,
        summaries,
        missing,
        stale,
        extras,
        invalid,
        current_sha,
        plot_paths,
    )
    print(
        json.dumps(
            {
                "summary": args.out_md,
                "plots": [str(p) for p in plot_paths],
                "valid_results": len(results),
                "missing": len(missing),
                "stale": len(stale),
                "extras": len(extras),
                "invalid": len(invalid),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
