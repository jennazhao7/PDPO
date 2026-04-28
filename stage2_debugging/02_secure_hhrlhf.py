#!/usr/bin/env python3
"""
Secure and reproducibly snapshot Anthropic HH-RLHF for tonight's core experiment.

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


def canonical_example_id(prompt: str, chosen: str, rejected: str) -> str:
    raw = f"{prompt}\n<<C>>\n{chosen}\n<<R>>\n{rejected}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def extract_anthropic_prompt(prompt_and_response: str) -> str:
    search_term = "\n\nAssistant:"
    idx = prompt_and_response.rfind(search_term)
    if idx == -1:
        raise ValueError("Expected '\n\nAssistant:' in prompt+response text.")
    return prompt_and_response[: idx + len(search_term)]


def normalize_row(row: Dict) -> Optional[Dict]:
    chosen_full = str(row.get("chosen", "")).strip()
    rejected_full = str(row.get("rejected", "")).strip()

    if not chosen_full or not rejected_full:
        return None

    try:
        prompt = extract_anthropic_prompt(chosen_full)
        chosen = chosen_full[len(prompt):].strip()
        rejected = rejected_full[len(prompt):].strip()
    except ValueError:
        return None

    return {
        "id": canonical_example_id(prompt, chosen, rejected),
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
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
    ap = argparse.ArgumentParser(description="Securely snapshot Anthropic HH-RLHF.")
    ap.add_argument("--dataset_id", default="Anthropic/hh-rlhf")
    ap.add_argument("--dataset_config", default="default")
    ap.add_argument(
        "--revision",
        default=None,
        help="Optional dataset revision/commit hash for strict reproducibility.",
    )
    ap.add_argument(
        "--output_dir",
        default="data/hhrlhf_secure",
        help="Root output directory.",
    )
    ap.add_argument("--train_rows", type=int, default=10000, help="0 means all rows.")
    ap.add_argument("--test_rows", type=int, default=1000, help="0 means all rows.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[Data] Loading {args.dataset_id} (config={args.dataset_config}, revision={args.revision})")
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
        },
        "normalization": {
            "preference_rule": "split 'chosen' and 'rejected' by last '\\n\\nAssistant:'",
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
