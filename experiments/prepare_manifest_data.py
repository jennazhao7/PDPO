#!/usr/bin/env python3
"""Build the canonical clean/test/RR data bundle required by manifest.yaml."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import aggregate


DATASET_SOURCES = {
    "truthy": Path("preprocessing/truthydpo/dpo_train_ready.jsonl"),
    "hhrlhf": Path("preprocessing/hhrlhf/dpo_train_ready.jsonl"),
    "pku": Path("data/pku_saferlhf_secure/source_pref.jsonl"),
}

CANONICAL_DIRS = {
    "truthy": Path("data/truthydpo_secure"),
    "hhrlhf": Path("data/hhrlhf_secure"),
    "pku": Path("data/pku_saferlhf_secure"),
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for index, row in enumerate(rows):
        missing = [key for key in ("prompt", "chosen", "rejected") if key not in row]
        if missing:
            raise ValueError(f"{path}:{index + 1} missing {missing}")
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def row_id(row: Dict[str, Any]) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    payload = f"{row['prompt']}\n{row['chosen']}\n{row['rejected']}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:24]


def normalize(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        prompt = str(row["prompt"]).strip()
        chosen = str(row["chosen"]).strip()
        rejected = str(row["rejected"]).strip()
        if not prompt or not chosen or not rejected:
            continue
        normalized = {"id": row_id(row), "prompt": prompt, "chosen": chosen, "rejected": rejected}
        if normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        out.append(normalized)
    return out


def stable_uniform(seed: int, example_id: str) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(example_id.encode("utf-8"))
    return int.from_bytes(h.digest(), "big") / 2**64


def gamma_eps(eps: float) -> float:
    return 1.0 / (1.0 + math.exp(eps))


def split_clean(rows: List[Dict[str, Any]], train_n: int, test_n: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    if len(rows) < train_n + test_n:
        raise ValueError(f"Need {train_n + test_n} rows, found {len(rows)}")
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        prompt_key = hashlib.sha256(row["prompt"].encode()).hexdigest()
        groups.setdefault(prompt_key, []).append(row)
    grouped = list(groups.values())
    random.Random(seed).shuffle(grouped)

    test: List[Dict[str, Any]] = []
    train: List[Dict[str, Any]] = []
    remaining = []
    for group in grouped:
        if len(test) < test_n:
            test.extend(group)
        else:
            remaining.append(group)
    for group in remaining:
        if len(train) >= train_n:
            break
        train.extend(group)
    if len(train) < train_n:
        raise ValueError(f"Prompt-group split left only {len(train)} train rows; need {train_n}")
    train_prompts = {hashlib.sha256(row["prompt"].encode()).hexdigest() for row in train}
    test_prompts = {hashlib.sha256(row["prompt"].encode()).hexdigest() for row in test}
    overlap = train_prompts & test_prompts
    if overlap:
        raise ValueError(f"Clean split prompt overlap: {len(overlap)}")
    return train, test


def rr_flip(rows: Sequence[Dict[str, Any]], eps: float, seed: int) -> Tuple[List[Dict], Dict[str, Any]]:
    gamma = gamma_eps(eps)
    output = []
    flips = 0
    for row in rows:
        flipped = stable_uniform(seed, str(row["id"])) < gamma
        record = dict(row)
        if flipped:
            record["chosen"], record["rejected"] = row["rejected"], row["chosen"]
            flips += 1
        record["rr_flipped"] = flipped
        record["rr_gamma"] = gamma
        output.append(record)
    return output, {
        "eps": eps,
        "seed": seed,
        "gamma_eps": gamma,
        "n": len(output),
        "flips": flips,
        "observed_flip_rate": flips / len(output) if output else 0.0,
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_sizes(dataset: str, available: int, args: argparse.Namespace) -> Tuple[int, int]:
    requested_train = args.train_rows
    requested_test = args.test_rows
    if dataset == "truthy":
        requested_train = min(requested_train, 800)
        requested_test = min(requested_test, 200)
    if available < requested_train + requested_test:
        raise ValueError(
            f"{dataset}: requested train={requested_train}, test={requested_test}, available={available}"
        )
    return requested_train, requested_test


def canonical_test_alias(dataset: str) -> Path:
    return {
        "truthy": Path("stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"),
        "hhrlhf": Path("stage2_debugging/test_pref.jsonl"),
        "pku": Path("stage2_debugging/testsets/pku_secure/test_pref.jsonl"),
    }[dataset]


def download_pku_source(path: Path, rows_needed: int, seed: int) -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Downloading PKU requires the `datasets` package. Install requirements.txt "
            "or run with PYTHONPATH pointing to a local datasets installation."
        ) from exc
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=max(10_000, rows_needed * 4))
    normalized = []
    for row in dataset:
        prompt = str(row.get("prompt", "")).strip()
        r0 = str(row.get("response_0", "")).strip()
        r1 = str(row.get("response_1", "")).strip()
        better = row.get("better_response_id")
        if not prompt or not r0 or not r1 or better not in (0, 1):
            continue
        normalized.append(
            {
                "id": row_id({"prompt": prompt, "chosen": r0, "rejected": r1}),
                "prompt": prompt,
                "chosen": r0 if better == 0 else r1,
                "rejected": r1 if better == 0 else r0,
            }
        )
        if len(normalized) >= rows_needed:
            break
    if len(normalized) < rows_needed:
        raise ValueError(f"PKU download yielded {len(normalized)} usable rows; need {rows_needed}")
    write_jsonl(path, normalized[:rows_needed])


def prepare_dataset(
    dataset: str,
    eps_values: Sequence[float],
    seeds: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    source = DATASET_SOURCES[dataset]
    if dataset == "pku" and not source.exists() and args.download_pku:
        download_pku_source(source, args.train_rows + args.test_rows, args.split_seed)
    if not source.exists():
        raise FileNotFoundError(
            f"Missing source for {dataset}: {source}. "
            "For PKU run experiments/core_result_tonight/01_secure_pku_saferlhf.py "
            "or pass its normalized train_pref.jsonl as data/pku_saferlhf_secure/source_pref.jsonl."
        )
    rows = normalize(read_jsonl(source))
    train_n, test_n = dataset_sizes(dataset, len(rows), args)
    train, test = split_clean(rows, train_n, test_n, args.split_seed)
    canonical_dir = CANONICAL_DIRS[dataset]
    train_path = canonical_dir / "train_pref.jsonl"
    test_path = canonical_dir / "test_pref.jsonl"
    write_jsonl(train_path, train)
    write_jsonl(test_path, test)
    write_jsonl(canonical_test_alias(dataset), test)

    rr_outputs = []
    audit_path = Path("stage2_debugging/preprocessing") / f"rr_audit_{dataset}.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as audit:
        for eps in eps_values:
            for seed in seeds:
                noisy, stats = rr_flip(train, eps, seed)
                out = Path("stage2_debugging/preprocessing") / (
                    f"d2_rr_flipped_{dataset}_eps{eps}_seed{seed}.jsonl"
                )
                write_jsonl(out, noisy)
                stats["path"] = str(out)
                stats["sha256"] = file_sha256(out)
                audit.write(json.dumps(stats, sort_keys=True) + "\n")
                rr_outputs.append(stats)

    manifest = {
        "dataset": dataset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "source_sha256": file_sha256(source),
        "split_seed": args.split_seed,
        "train": {"path": str(train_path), "n": len(train), "sha256": file_sha256(train_path)},
        "test": {"path": str(test_path), "n": len(test), "sha256": file_sha256(test_path)},
        "rr_outputs": rr_outputs,
    }
    manifest_path = canonical_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    root = aggregate.repo_root()
    ap = argparse.ArgumentParser(description="Prepare all manifest data deterministically.")
    ap.add_argument("--manifest", default=str(root / "experiments/manifest.yaml"))
    ap.add_argument("--datasets", default="truthy,hhrlhf,pku")
    ap.add_argument("--train-rows", type=int, default=1000)
    ap.add_argument("--test-rows", type=int, default=500)
    ap.add_argument("--split-seed", type=int, default=123)
    ap.add_argument("--download-pku", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    manifest = aggregate.load_manifest(Path(args.manifest))
    cells = aggregate.expand_manifest(manifest)
    eps_by_dataset: Dict[str, set[float]] = {}
    seeds_by_dataset: Dict[str, set[int]] = {}
    for cell in cells:
        eps_by_dataset.setdefault(str(cell["dataset"]), set()).add(float(cell["eps"]))
        seeds_by_dataset.setdefault(str(cell["dataset"]), set()).add(int(cell["seed"]))
    selected = [item.strip() for item in args.datasets.split(",") if item.strip()]
    summaries = []
    for dataset in selected:
        summaries.append(
            prepare_dataset(
                dataset,
                sorted(eps_by_dataset[dataset]),
                sorted(seeds_by_dataset[dataset]),
                args,
            )
        )
    print(json.dumps({"datasets": summaries}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
