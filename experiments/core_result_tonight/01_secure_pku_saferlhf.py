#!/usr/bin/env python3
"""
Secure and reproducibly snapshot PKU-SafeRLHF for tonight's core experiment.

Outputs:
  - raw_hf_dataset/            HuggingFace dataset saved via save_to_disk()
  - train_pref.jsonl           Normalized preference pairs for training
  - test_pref.jsonl            Normalized preference pairs for held-out eval
  - manifest.json              Full provenance + integrity hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

from datasets import Dataset, DatasetDict, load_dataset


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def canonical_example_id(prompt: str, r0: str, r1: str) -> str:
    raw = f"{prompt}\n<<R0>>\n{r0}\n<<R1>>\n{r1}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def normalize_row(row: Dict) -> Optional[Dict]:
    prompt = str(row.get("prompt", "")).strip()
    r0 = str(row.get("response_0", "")).strip()
    r1 = str(row.get("response_1", "")).strip()
    better = row.get("better_response_id", None)
    safer = row.get("safer_response_id", None)

    if not prompt or not r0 or not r1:
        return None
    if better not in (0, 1):
        return None

    chosen = r0 if better == 0 else r1
    rejected = r1 if better == 0 else r0

    return {
        "id": canonical_example_id(prompt, r0, r1),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "response_0": r0,
        "response_1": r1,
        "better_response_id": int(better),
        "safer_response_id": int(safer) if safer in (0, 1) else None,
        "response_0_source": row.get("response_0_source"),
        "response_1_source": row.get("response_1_source"),
        "prompt_source": row.get("prompt_source"),
        "response_0_severity_level": row.get("response_0_severity_level"),
        "response_1_severity_level": row.get("response_1_severity_level"),
        "response_0_harm_category": row.get("response_0_harm_category"),
        "response_1_harm_category": row.get("response_1_harm_category"),
    }


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


@dataclass
class SplitResult:
    name: str
    in_rows: int
    out_rows: int
    dropped_rows: int
    path: str
    sha256: str


def normalize_split(ds: Dataset, split_name: str, out_path: str, limit: int) -> SplitResult:
    if limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    normalized: List[Dict] = []
    for row in ds:
        item = normalize_row(row)
        if item is not None:
            normalized.append(item)

    out_rows = write_jsonl(out_path, normalized)
    return SplitResult(
        name=split_name,
        in_rows=len(ds),
        out_rows=out_rows,
        dropped_rows=len(ds) - out_rows,
        path=out_path,
        sha256=sha256_file(out_path),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Securely snapshot PKU-SafeRLHF.")
    ap.add_argument("--dataset_id", default="PKU-Alignment/PKU-SafeRLHF")
    ap.add_argument("--dataset_config", default="default")
    ap.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision/commit hash for strict reproducibility.",
    )
    ap.add_argument(
        "--output_dir",
        default="data/pku_saferlhf_secure",
        help="Root output directory.",
    )
    ap.add_argument("--train_rows", type=int, default=10000, help="0 means all rows.")
    ap.add_argument("--test_rows", type=int, default=1000, help="0 means all rows.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(
        f"[Data] Loading {args.dataset_id} (config={args.dataset_config}, revision={args.revision})"
    )
    ds_dict: DatasetDict = load_dataset(
        args.dataset_id,
        name=args.dataset_config,
        revision=args.revision,
    )

    required_splits = ["train", "test"]
    for s in required_splits:
        if s not in ds_dict:
            raise ValueError(f"Dataset missing required split '{s}'. Found: {list(ds_dict.keys())}")

    raw_out = os.path.join(args.output_dir, "raw_hf_dataset")
    print(f"[Data] Saving full raw dataset snapshot to {raw_out}")
    ds_dict.save_to_disk(raw_out)

    train_path = os.path.join(args.output_dir, "train_pref.jsonl")
    test_path = os.path.join(args.output_dir, "test_pref.jsonl")

    train_res = normalize_split(ds_dict["train"], "train", train_path, args.train_rows)
    test_res = normalize_split(ds_dict["test"], "test", test_path, args.test_rows)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": args.dataset_id,
            "config": args.dataset_config,
            "revision": args.revision,
            "available_splits": list(ds_dict.keys()),
            "license_note": "PKU-SafeRLHF is published with CC-BY-NC-4.0 terms.",
        },
        "normalization": {
            "preference_rule": "chosen/rejected derived from better_response_id in {0,1}",
            "required_fields": ["prompt", "response_0", "response_1", "better_response_id"],
        },
        "splits": {
            "train": train_res.__dict__,
            "test": test_res.__dict__,
        },
        "raw_snapshot": {
            "path": raw_out,
        },
    }

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[Done] Secure data snapshot complete.")
    print(f"  - train: {train_res.out_rows} rows -> {train_res.path}")
    print(f"  - test : {test_res.out_rows} rows -> {test_res.path}")
    print(f"  - manifest: {manifest_path}")


if __name__ == "__main__":
    main()
