#!/usr/bin/env python3
import torch
assert torch.cuda.is_available(), "Refusing to run on CPU"

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

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

SPOT_INTERRUPTION_PATTERNS = (
    "spot interruption",
    "preempt",
    "preempted",
    "instance terminated",
    "host error",
    "sigterm",
    "signal 15",
    "job killed",
    "connection reset",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_git_sha(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


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
                if value.strip():
                    data[section] = parse_scalar(value.strip())
                else:
                    data[section] = [] if section == "queue" else {}
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


def expand_cells(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
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


def cell_id(cell: Dict[str, Any]) -> str:
    return f"{cell['id']}_{cell['method']}_{cell['dataset']}_eps{slug_eps(cell['eps'])}_seed{cell['seed']}"


def format_template(template: str, cell: Dict[str, Any], paths: Dict[str, str]) -> str:
    values = dict(cell)
    values.update(paths)
    values["eps_slug"] = slug_eps(cell["eps"])
    return template.format(**values)


def result_path(result_root: Path, cell: Dict[str, Any]) -> Path:
    return result_root / str(cell["id"]) / f"{cell_id(cell)}.json"


def model_dir(result_root: Path, cell: Dict[str, Any]) -> Path:
    return result_root / str(cell["id"]) / "models" / cell_id(cell)


def scratch_dir(result_root: Path, cell: Dict[str, Any]) -> Path:
    return result_root / str(cell["id"]) / "scratch" / cell_id(cell)


def ref_cache_path(args: argparse.Namespace, cell: Dict[str, Any]) -> Path:
    model = str(cell.get("base_model", "model")).replace("/", "--").replace(":", "_")
    return Path(args.ref_logps_dir) / f"{cell['dataset']}_{model}_ref_logps.jsonl"


def validate_result(obj: Dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_RESULT_FIELDS if key not in obj]
    if missing:
        raise ValueError(f"Result JSON missing required fields: {missing}")
    for key, expected in REQUIRED_RESULT_FIELDS.items():
        if not isinstance(obj[key], expected):
            raise ValueError(f"Result field {key!r} has invalid type: {type(obj[key]).__name__}")
    if obj["n"] < 0:
        raise ValueError("Result field 'n' must be nonnegative")
    for key in ("acc", "mia_auc"):
        if not (0.0 <= float(obj[key]) <= 1.0):
            raise ValueError(f"Result field {key!r} must be in [0, 1]")


def load_valid_result(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        validate_result(obj)
        return obj
    except Exception:
        return None


def append_log(log_path: Path, record: Dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("timestamp", utc_now())
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def prompt_hashes(jsonl_path: Path) -> set[str]:
    hashes = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "prompt" not in row:
                raise ValueError(f"Missing prompt field in {jsonl_path}")
            hashes.add(prompt_hash(str(row["prompt"])))
    return hashes


def assert_prompt_disjoint(train_jsonl: Path, test_jsonl: Path) -> None:
    train_hashes = prompt_hashes(train_jsonl)
    test_hashes = prompt_hashes(test_jsonl)
    overlap = train_hashes & test_hashes
    if overlap:
        raise ValueError(
            f"Train/test prompt overlap detected: {len(overlap)} prompt hashes overlap "
            f"between {train_jsonl} and {test_jsonl}"
        )


def latest_checkpoint(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    checkpoints = [p for p in path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def command_output_tail(path: Path, max_chars: int = 20000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def is_spot_interruption(log_text: str, returncode: int) -> bool:
    text = log_text.lower()
    return returncode in {124, 137, 143} or any(pattern in text for pattern in SPOT_INTERRUPTION_PATTERNS)


def run_command_with_spot_retries(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    max_spot_retries: int,
    env: Optional[Dict[str, str]] = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] attempt={attempt} cmd={json.dumps(cmd)}\n")
            proc = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env or os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            log.write(f"[{utc_now()}] returncode={proc.returncode}\n")
        if proc.returncode == 0:
            return
        tail = command_output_tail(log_path)
        if is_spot_interruption(tail, proc.returncode) and attempt <= max_spot_retries:
            time.sleep(min(60, 5 * attempt))
            continue
        raise RuntimeError(f"Command failed with return code {proc.returncode}; see {log_path}")


def append_cost_ledger(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell",
        "method",
        "dataset",
        "eps",
        "seed",
        "card",
        "start",
        "end",
        "gpu_hours",
        "est_cost",
        "result_json",
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def cleanup_intermediate_checkpoints(out_dir: Path) -> None:
    if "outputs" in out_dir.parts:
        raise ValueError(f"Refusing cleanup inside protected outputs path: {out_dir}")
    for ckpt in out_dir.glob("checkpoint-*"):
        if not ckpt.is_dir():
            continue
        for child in sorted(ckpt.rglob("*"), reverse=True):
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
        ckpt.rmdir()


def enforce_lora_only(out_dir: Path) -> None:
    forbidden = []
    for pattern in ("pytorch_model*.bin", "model*.safetensors", "consolidated*.pth"):
        forbidden.extend(out_dir.glob(pattern))
    if forbidden:
        names = ", ".join(str(path) for path in forbidden)
        raise ValueError(f"Refusing full-model artifacts; keep LoRA adapters only: {names}")


def true_eta(eps: float) -> float:
    return math.exp(eps) / (1.0 + math.exp(eps))


def estimate_eta_from_jsonl(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    total = 0
    kept = 0
    candidate_keys = ("label_kept", "rr_kept", "keep", "kept")
    flipped_keys = ("is_flipped", "flipped", "rr_flipped")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            found = False
            for key in candidate_keys:
                if key in row:
                    kept += 1 if bool(row[key]) else 0
                    found = True
                    break
            if found:
                continue
            for key in flipped_keys:
                if key in row:
                    kept += 0 if bool(row[key]) else 1
                    found = True
                    break
            if not found:
                return None
    if total == 0:
        return None
    return kept / total


def extract_metric(path: Path, keys: Iterable[str]) -> Optional[float]:
    if not path.exists():
        return None
    obj = read_json(path)
    for key in keys:
        if key in obj and obj[key] is not None:
            return float(obj[key])
    return None


def extract_count(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    obj = read_json(path)
    for key in ("n", "count", "n_eval", "n_examples"):
        if key in obj and obj[key] is not None:
            return int(obj[key])
    return None


def build_paths(args: argparse.Namespace, cell: Dict[str, Any], result_root: Path) -> Dict[str, str]:
    model = model_dir(result_root, cell)
    scratch = scratch_dir(result_root, cell)
    paths = {
        "result_root": str(result_root),
        "cell_id": cell_id(cell),
        "model_dir": str(model),
        "scratch_dir": str(scratch),
        "model_manifest": str(model / "M2_manifest.json"),
        "train_jsonl": "",
        "test_jsonl": "",
        "member_jsonl": "",
        "nonmember_jsonl": "",
        "resume_checkpoint": "",
        "ref_logps_cache": "",
    }
    paths["train_jsonl"] = format_template(args.train_template, cell, paths)
    paths["test_jsonl"] = resolve_test_template(args.test_template, cell, paths)
    paths["member_jsonl"] = format_template(args.member_template, cell, paths)
    paths["nonmember_jsonl"] = format_template(args.nonmember_template, cell, paths)
    paths["stage1_adapter"] = format_template(args.stage1_template, cell, paths)
    cache = ref_cache_path(args, cell)
    if cache.exists():
        paths["ref_logps_cache"] = str(cache)
    return paths


def resolve_test_template(template: str, cell: Dict[str, Any], paths: Dict[str, str]) -> str:
    if template != "auto":
        return format_template(template, cell, paths)
    dataset = str(cell["dataset"])
    auto_paths = {
        "truthy": "stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl",
        "hhrlhf": "stage2_debugging/test_pref.jsonl",
        "pku": "stage2_debugging/testsets/pku_secure/test_pref.jsonl",
    }
    if dataset not in auto_paths:
        raise KeyError(f"No automatic test path configured for dataset={dataset!r}")
    return auto_paths[dataset]


def command_from_template(template: str, cell: Dict[str, Any], paths: Dict[str, str]) -> List[str]:
    rendered = format_template(template, cell, paths)
    return shlex.split(rendered)


def train_command(args: argparse.Namespace, cell: Dict[str, Any], paths: Dict[str, str]) -> List[str]:
    method = str(cell["method"])
    custom = (args.method_command or {}).get(method)
    if custom:
        return command_from_template(custom, cell, paths)
    if method == "rdpo":
        return [
            sys.executable,
            "-u",
            "stage2_debugging/train_stage2_rdpo.py",
            "--model",
            str(cell["base_model"]),
            "--stage1_adapter",
            paths["stage1_adapter"],
            "--data",
            paths["train_jsonl"],
            "--out",
            paths["model_dir"],
            "--manifest_out",
            paths["model_manifest"],
            "--rr_epsilon",
            str(cell["eps"]),
            "--lora_r",
            str(cell.get("lora_rank", 16)),
            "--seed",
            str(cell["seed"]),
            "--bf16",
        ]
    if method == "limo":
        return [
            sys.executable,
            "-u",
            "stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py",
            "--model",
            str(cell["base_model"]),
            "--stage1_adapter",
            paths["stage1_adapter"],
            "--data",
            paths["train_jsonl"],
            "--out",
            paths["model_dir"],
            "--manifest_out",
            paths["model_manifest"],
            "--epsilon",
            str(cell["eps"]),
            "--keep_fraction",
            str(cell.get("keep_fraction", 0.5)),
            "--lora_r",
            str(cell.get("lora_rank", 16)),
            "--seed",
            str(cell["seed"]),
            "--bf16",
        ]
    raise NotImplementedError(
        f"No built-in command for method={method!r}. Add --method-command {method}=<template>."
    )


def eval_command(paths: Dict[str, str], eval_json: Path) -> List[str]:
    return [
        sys.executable,
        "-u",
        "stage2_debugging/eval_preference_accuracy.py",
        "--manifest",
        paths["model_manifest"],
        "--test_jsonl",
        paths["test_jsonl"],
        "--out_json",
        str(eval_json),
    ]


def mia_command(paths: Dict[str, str], mia_json: Path, seed: int) -> List[str]:
    return [
        sys.executable,
        "-u",
        "stage2_debugging/eval_privacy_audit.py",
        "--manifest",
        paths["model_manifest"],
        "--member_jsonl",
        paths["member_jsonl"],
        "--nonmember_jsonl",
        paths["nonmember_jsonl"],
        "--out_json",
        str(mia_json),
        "--seed",
        str(seed),
    ]


def require_files(paths: Dict[str, str], keys: Iterable[str]) -> None:
    missing = [paths[key] for key in keys if not Path(paths[key]).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def run_cell(args: argparse.Namespace, root: Path, result_root: Path, log_path: Path, cell: Dict[str, Any]) -> None:
    rid = result_path(result_root, cell)
    valid_existing = load_valid_result(rid)
    if valid_existing and not args.force:
        append_log(log_path, {"cell": cell_id(cell), "status": "skipped_existing", "result_json": str(rid)})
        return

    paths = build_paths(args, cell, result_root)
    cell_scratch = Path(paths["scratch_dir"])
    cell_scratch.mkdir(parents=True, exist_ok=True)
    Path(paths["model_dir"]).mkdir(parents=True, exist_ok=True)
    rid.parent.mkdir(parents=True, exist_ok=True)

    eta_true = true_eta(float(cell["eps"])) if cell.get("method") == "redpo" else None
    eta_est = estimate_eta_from_jsonl(Path(paths["train_jsonl"])) if cell.get("method") == "redpo" else None
    checkpoint = latest_checkpoint(Path(paths["model_dir"]))
    paths["resume_checkpoint"] = str(checkpoint) if checkpoint else ""

    append_log(
        log_path,
        {
            "cell": cell_id(cell),
            "status": "started",
            "method": cell["method"],
            "dataset": cell["dataset"],
            "eps": cell["eps"],
            "seed": cell["seed"],
            "result_json": str(rid),
            "resume_checkpoint": str(checkpoint) if checkpoint else None,
            "eta_true": eta_true,
            "eta_estimated": eta_est,
            "card": args.card,
            "ref_logps_cache": paths.get("ref_logps_cache") or None,
        },
    )

    try:
        start_ts = utc_now()
        start_time = time.time()
        require_files(paths, ("train_jsonl", "test_jsonl", "stage1_adapter"))
        assert_prompt_disjoint(Path(paths["train_jsonl"]), Path(paths["test_jsonl"]))

        train_log = cell_scratch / "train.log"
        env = os.environ.copy()
        if paths.get("ref_logps_cache"):
            env["PDPO_REF_LOGPS_CACHE"] = paths["ref_logps_cache"]
        run_command_with_spot_retries(
            train_command(args, cell, paths),
            cwd=root,
            log_path=train_log,
            max_spot_retries=args.spot_retries,
            env=env,
        )

        eval_json = cell_scratch / "eval.json"
        eval_log = cell_scratch / "eval.log"
        run_command_with_spot_retries(
            eval_command(paths, eval_json),
            cwd=root,
            log_path=eval_log,
            max_spot_retries=args.spot_retries,
            env=env,
        )
        acc = extract_metric(eval_json, ("acc", "accuracy"))
        n = extract_count(eval_json)
        if acc is None or n is None:
            raise ValueError(f"Eval output missing acc/accuracy or n: {eval_json}")

        mia_auc = 0.0
        if bool(cell.get("mia", False)):
            require_files(paths, ("member_jsonl", "nonmember_jsonl"))
            mia_json = cell_scratch / "mia.json"
            mia_log = cell_scratch / "mia.log"
            run_command_with_spot_retries(
                mia_command(paths, mia_json, int(cell["seed"])),
                cwd=root,
                log_path=mia_log,
                max_spot_retries=args.spot_retries,
                env=env,
            )
            parsed_mia_auc = extract_metric(mia_json, ("mia_auc", "auc"))
            if parsed_mia_auc is None:
                raise ValueError(f"MIA output missing mia_auc/auc: {mia_json}")
            mia_auc = parsed_mia_auc

        result = {
            "method": cell["method"],
            "dataset": cell["dataset"],
            "eps": float(cell["eps"]),
            "seed": int(cell["seed"]),
            "n": int(n),
            "acc": float(acc),
            "mia_auc": float(mia_auc),
            "git_sha": run_git_sha(root),
            "timestamp": utc_now(),
            "manifest_id": cell["id"],
            "base_model": cell.get("base_model"),
            "result_json": str(rid),
            "model_manifest": paths["model_manifest"],
            "eta_true": eta_true,
            "eta_estimated": eta_est,
            "card": args.card,
            "ref_logps_cache": paths.get("ref_logps_cache") or None,
        }
        validate_result(result)
        if rid.exists() and not args.force:
            raise FileExistsError(f"Refusing to overwrite existing result JSON without --force: {rid}")
        tmp = rid.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(rid)
        enforce_lora_only(Path(paths["model_dir"]))
        cleanup_intermediate_checkpoints(Path(paths["model_dir"]))
        end_ts = utc_now()
        gpu_hours = max(0.0, (time.time() - start_time) / 3600.0)
        append_cost_ledger(
            Path(args.cost_ledger),
            {
                "cell": cell_id(cell),
                "method": cell["method"],
                "dataset": cell["dataset"],
                "eps": cell["eps"],
                "seed": cell["seed"],
                "card": args.card,
                "start": start_ts,
                "end": end_ts,
                "gpu_hours": f"{gpu_hours:.6f}",
                "est_cost": f"{gpu_hours * args.spot_rate:.6f}",
                "result_json": str(rid),
            },
        )
        append_log(log_path, {"cell": cell_id(cell), "status": "completed", "result_json": str(rid)})
    except Exception as exc:
        append_log(log_path, {"cell": cell_id(cell), "status": "failed", "error": str(exc)})
        if not args.keep_going:
            raise


def parse_method_commands(values: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--method-command must be METHOD=TEMPLATE")
        method, template = item.split("=", 1)
        out[method.strip()] = template.strip()
    return out


def parse_args() -> argparse.Namespace:
    root = repo_root()
    ap = argparse.ArgumentParser(description="Run experiments/manifest.yaml cells sequentially.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--result-root", default=str(root / "experiments/run_queue"))
    ap.add_argument("--run-log", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--spot-retries", type=int, default=3)
    ap.add_argument("--card", default=config.PREFERRED_CARD)
    ap.add_argument("--cost-ledger", default=str(root / "cost_ledger.csv"))
    ap.add_argument("--ref-logps-dir", default=str(root / "experiments/ref_logps"))
    ap.add_argument("--method-command", action="append", default=[])
    ap.add_argument(
        "--stage1-template",
        default="stage2_debugging/stage1/results_eps{eps}_seed{seed}/{dataset}_eps{eps}_s{seed}",
    )
    ap.add_argument(
        "--train-template",
        default="stage2_debugging/preprocessing/d2_rr_flipped_{dataset}_eps{eps}_seed{seed}.jsonl",
    )
    ap.add_argument(
        "--test-template",
        default="auto",
    )
    ap.add_argument(
        "--member-template",
        default="stage2_debugging/mia_full/{dataset}/eps{eps}_seed{seed}/members.jsonl",
    )
    ap.add_argument(
        "--nonmember-template",
        default="stage2_debugging/mia_full/{dataset}/eps{eps}_seed{seed}/nonmembers.jsonl",
    )
    args = ap.parse_args()
    args.method_command = parse_method_commands(args.method_command)
    args.card = args.card.upper()
    args.spot_rate = config.get_card_config(args.card).rough_spot_usd_per_hour
    return args


def main() -> int:
    args = parse_args()
    root = repo_root()
    manifest = load_manifest(Path(args.manifest))
    cells = expand_cells(manifest)
    result_root = Path(args.result_root)
    log_path = Path(args.run_log) if args.run_log else result_root / "run_log.jsonl"
    append_log(log_path, {"status": "queue_started", "manifest": args.manifest, "cells": len(cells)})
    for cell in cells:
        run_cell(args, root, result_root, log_path, cell)
    append_log(log_path, {"status": "queue_finished", "manifest": args.manifest, "cells": len(cells)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
