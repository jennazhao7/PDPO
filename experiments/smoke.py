#!/usr/bin/env python3
import torch
assert torch.cuda.is_available(), "Refusing to run on CPU"

import argparse
import hashlib
import itertools
import json
import math
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import aggregate


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


def validate_result(obj: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RESULT_FIELDS if key not in obj]
    if missing:
        raise ValueError(f"Result JSON missing required fields: {missing}")
    for key, expected in REQUIRED_RESULT_FIELDS.items():
        if not isinstance(obj[key], expected):
            raise ValueError(f"{key} has invalid type {type(obj[key]).__name__}")
    if int(obj["n"]) != 8:
        raise ValueError(f"Smoke result must have n=8, got {obj['n']}")
    for key in ("acc", "mia_auc"):
        value = float(obj[key])
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{key} must be in [0, 1], got {value}")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def assert_disjoint(train_rows: List[Dict[str, Any]], test_rows: List[Dict[str, Any]]) -> None:
    train_hashes = {prompt_hash(row["prompt"]) for row in train_rows}
    test_hashes = {prompt_hash(row["prompt"]) for row in test_rows}
    overlap = train_hashes & test_hashes
    if overlap:
        raise AssertionError(f"Smoke train/test prompt overlap: {len(overlap)}")


def make_fixture_rows(prefix: str, n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for idx in range(n):
        bit = rng.randrange(2)
        rows.append(
            {
                "prompt": f"{prefix} prompt {idx} seed {seed}",
                "chosen": f" preferred response {idx} bit {bit}",
                "rejected": f" rejected response {idx} bit {1 - bit}",
                "pair_id": f"{prefix}-{seed}-{idx}",
                "kept": idx % 2 == 0,
                "flipped": idx % 3 == 0,
            }
        )
    return rows


def build_smoke_data(root: Path, n: int, seed: int) -> Dict[str, Path]:
    data_dir = root / "data"
    train_rows = make_fixture_rows("train", n, seed)
    test_rows = make_fixture_rows("test", n, seed + 1)
    members = make_fixture_rows("member", n, seed + 2)
    nonmembers = make_fixture_rows("nonmember", n, seed + 3)
    assert_disjoint(train_rows, test_rows)
    paths = {
        "train": data_dir / "train.jsonl",
        "test": data_dir / "test.jsonl",
        "members": data_dir / "members.jsonl",
        "nonmembers": data_dir / "nonmembers.jsonl",
    }
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["test"], test_rows)
    write_jsonl(paths["members"], members)
    write_jsonl(paths["nonmembers"], nonmembers)
    return paths


def first_smoke_cell_for_method(manifest: Dict[str, Any], method: str) -> Dict[str, Any]:
    defaults = manifest.get("defaults", {}) or {}
    for item in manifest.get("queue", []) or []:
        if item["method"] != method:
            continue
        seed = (item.get("seeds") or defaults.get("seeds") or [42])[0]
        cell = dict(defaults)
        cell.update(item)
        cell.update(
            {
                "dataset": item["datasets"][0],
                "eps": float(item["eps"][0]),
                "seed": int(seed),
            }
        )
        return cell
    raise KeyError(method)


def expected_methods(manifest: Dict[str, Any]) -> List[str]:
    methods = []
    for item in manifest.get("queue", []) or []:
        method = str(item["method"])
        if method not in methods:
            methods.append(method)
    return methods


def true_eta(eps: float) -> float:
    return math.exp(eps) / (1.0 + math.exp(eps))


def estimate_eta(rows: List[Dict[str, Any]]) -> float:
    kept = 0
    for row in rows:
        if "kept" in row:
            kept += 1 if row["kept"] else 0
        elif "flipped" in row:
            kept += 0 if row["flipped"] else 1
        else:
            kept += 1
    return kept / max(1, len(rows))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def one_training_step(method: str, rows: List[Dict[str, Any]], eps: float, keep_fraction: Optional[float]) -> Dict[str, float]:
    # Tiny deterministic "training" update. This is intentionally not model
    # training; it checks method plumbing, data flow, schema, and reports.
    weight = 0.0
    lr = 0.1
    gamma_eps = 1.0 / (1.0 + math.exp(eps))
    selected = rows
    if method in {"random_subset", "limo"} and keep_fraction is not None:
        k = max(1, int(round(len(rows) * keep_fraction)))
        if method == "random_subset":
            selected = sorted(rows, key=lambda row: row["pair_id"])[:k]
        else:
            selected = sorted(rows, key=lambda row: hashlib.sha1(row["prompt"].encode()).hexdigest())[:k]
    signal = sum(len(row["chosen"]) - len(row["rejected"]) for row in selected)
    if method == "rdpo":
        weight += lr * signal / max(1, len(selected)) / max(1e-6, 1.0 - 2.0 * gamma_eps)
    elif method == "redpo":
        eta_hat = estimate_eta(selected)
        weight += lr * signal / max(1, len(selected)) * eta_hat
    elif method == "random_subset":
        weight += lr * signal / max(1, len(selected))
    elif method == "limo":
        weight += lr * signal / max(1, len(selected)) * true_eta(eps)
    else:
        raise ValueError(f"Unknown method in smoke test: {method}")
    return {"weight": weight, "selected_n": float(len(selected)), "gamma_eps": gamma_eps}


def smoke_metrics(method: str, state: Dict[str, float], eps: float) -> Tuple[float, float]:
    base = 0.55 + 0.03 * math.tanh(state["weight"])
    method_bump = {
        "rdpo": 0.01,
        "redpo": 0.012,
        "random_subset": 0.0,
        "limo": 0.015,
    }[method]
    acc = min(1.0, max(0.0, base + method_bump))
    mia_auc = min(1.0, max(0.0, 0.62 - 0.02 * true_eta(eps) - (0.03 if method == "limo" else 0.0)))
    return acc, mia_auc


def write_method_result(
    root: Path,
    cell: Dict[str, Any],
    data_paths: Dict[str, Path],
    current_sha: str,
) -> Path:
    method = str(cell["method"])
    train_rows = load_jsonl(data_paths["train"])
    state = one_training_step(method, train_rows, float(cell["eps"]), cell.get("keep_fraction"))
    acc, mia_auc = smoke_metrics(method, state, float(cell["eps"]))
    method_dir = root / str(cell["id"])
    method_dir.mkdir(parents=True, exist_ok=True)
    result_path = method_dir / (
        f"smoke_{cell['id']}_{method}_{cell['dataset']}_eps{aggregate.slug_eps(float(cell['eps']))}_seed{cell['seed']}.json"
    )
    result = {
        "method": method,
        "dataset": str(cell["dataset"]),
        "eps": float(cell["eps"]),
        "seed": int(cell["seed"]),
        "n": 8,
        "acc": float(acc),
        "mia_auc": float(mia_auc),
        "git_sha": current_sha,
        "timestamp": utc_now(),
        "manifest_id": str(cell["id"]),
        "smoke": True,
        "training_steps": 1,
        "selected_n": int(state["selected_n"]),
        "gamma_eps": state["gamma_eps"],
        "eta_true": true_eta(float(cell["eps"])),
        "eta_estimated": estimate_eta(train_rows) if method == "redpo" else None,
    }
    validate_result(result)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result_path


def run_reports(root: Path, manifest: Path, smoke_root: Path) -> None:
    summary = smoke_root / "SMOKE_EXPERIMENT_MEGA_SUMMARY.md"
    decision = smoke_root / "SMOKE_DECISION_REPORT.md"
    plot_dir = smoke_root / "frontiers"
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/aggregate.py"),
            "--manifest",
            str(manifest),
            "--result-root",
            str(smoke_root),
            "--out-md",
            str(summary),
            "--plot-dir",
            str(plot_dir),
            "--bootstrap-reps",
            "200",
        ],
        cwd=str(root),
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "experiments/decision_report.py"),
            "--manifest",
            str(manifest),
            "--result-root",
            str(smoke_root),
            "--out-md",
            str(decision),
            "--bootstrap-reps",
            "200",
        ],
        cwd=str(root),
        check=True,
    )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Run an n=8, one-step smoke test for every method in manifest.yaml.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--smoke-root", default=str(root / "experiments/smoke_results"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-reports", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.n != 8:
        raise ValueError("Smoke test invariant: n must be 8.")
    root = repo_root()
    manifest_path = Path(args.manifest)
    smoke_root = Path(args.smoke_root)
    smoke_root.mkdir(parents=True, exist_ok=True)
    manifest = aggregate.load_manifest(manifest_path)
    data_paths = build_smoke_data(smoke_root, args.n, args.seed)
    current_sha = git_sha(root)
    result_paths = []
    for method in expected_methods(manifest):
        cell = first_smoke_cell_for_method(manifest, method)
        result_paths.append(write_method_result(smoke_root, cell, data_paths, current_sha))
    if not args.skip_reports:
        run_reports(root, manifest_path, smoke_root)
    summary = {
        "status": "ok",
        "n": args.n,
        "training_steps": 1,
        "methods": expected_methods(manifest),
        "results": [str(path) for path in result_paths],
        "timestamp": utc_now(),
    }
    summary_path = smoke_root / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
