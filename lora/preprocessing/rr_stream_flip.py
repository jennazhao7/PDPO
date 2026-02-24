#!/usr/bin/env python3
"""
Stream DPO examples and apply deterministic RR flipping.

This module supports streaming datasets via Hugging Face `datasets` or a plain
JSONL iterator. It never writes privatized datasets to disk; it only emits
small audit logs. Partitioning is deterministic and streaming-safe, so you
can split a stream (e.g., first half then second half) without counting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


REQUIRED_FIELDS = ("prompt", "chosen", "rejected")


def q_from_epsilon(epsilon: float) -> float:
    """Flip probability for (epsilon, 0)-LDP binary RR."""
    return 1.0 / (math.exp(epsilon) + 1.0)


def get_example_id(example: Dict, id_key: str = "id") -> str:
    """Return a stable example_id (id field or md5 hash of content)."""
    if id_key in example and example[id_key] is not None:
        return str(example[id_key])
    prompt = str(example.get("prompt", ""))
    chosen = str(example.get("chosen", ""))
    rejected = str(example.get("rejected", ""))
    raw = f"{prompt}\n{chosen}\n{rejected}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def stable_uniform(seed: int, example_id: str) -> float:
    """
    Stable U(0,1) derived from (seed, example_id) using blake2b (8 bytes).
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"|")
    h.update(str(example_id).encode("utf-8"))
    digest = h.digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value / 2**64


def apply_rr_flip(
    example: Dict,
    epsilon: float,
    seed: int,
    id_key: str = "id",
    example_id: Optional[str] = None,
) -> Tuple[Dict, bool, str, float, float]:
    """Apply RR flip deterministically to a single example."""
    if example_id is None:
        example_id = get_example_id(example, id_key=id_key)
    q = q_from_epsilon(epsilon)
    u = stable_uniform(seed, example_id)
    rr_flipped = u < q
    if rr_flipped:
        flipped = dict(example)
        flipped["chosen"], flipped["rejected"] = example["rejected"], example["chosen"]
        return flipped, True, example_id, q, u
    return example, False, example_id, q, u


def iter_jsonl(path: str) -> Iterator[Dict]:
    """Stream JSONL file line-by-line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_stream(args: argparse.Namespace) -> Iterable[Dict]:
    """
    Build a streaming dataset iterator.
    - If input_jsonl is set, stream via file iterator.
    - Else, use datasets.load_dataset(..., streaming=True).
    """
    if args.input_jsonl:
        return iter_jsonl(args.input_jsonl)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required unless --input_jsonl is provided."
        ) from exc

    dataset_name = args.dataset_name
    data_files = None
    if args.data_files:
        data_files = [s for s in args.data_files.split(",") if s]
        if dataset_name is None:
            dataset_name = "json"

    if dataset_name is None:
        raise ValueError("Provide --dataset_name or --input_jsonl.")

    return load_dataset(
        path=dataset_name,
        name=args.dataset_config,
        data_files=data_files,
        split=args.split,
        streaming=True,
        revision=args.revision,
    )


def validate_example(example: Dict) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in example]
    if missing:
        raise KeyError(f"Missing required fields: {missing}")


def in_partition(
    example_id: str,
    partition_index: Optional[int],
    partition_count: Optional[int],
    partition_seed: int,
) -> bool:
    """
    Deterministically assign examples to partitions without counting.
    If partition_count is None, include all examples.
    """
    if partition_count is None:
        return True
    h = hashlib.blake2b(digest_size=8)
    h.update(str(partition_seed).encode("utf-8"))
    h.update(b"|")
    h.update(str(example_id).encode("utf-8"))
    value = int.from_bytes(h.digest(), byteorder="big", signed=False)
    return (value % partition_count) == partition_index


def wilson_interval(p_hat: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a Bernoulli proportion."""
    if n <= 0:
        return 0.0, 1.0
    denom = 1.0 + (z**2) / n
    center = (p_hat + (z**2) / (2.0 * n)) / denom
    margin = (
        z
        * math.sqrt(
            (p_hat * (1.0 - p_hat) + (z**2) / (4.0 * n)) / n
        )
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def audit_stream(
    iterator: Iterable[Dict],
    epsilon: float,
    seed: int,
    max_audit: int,
    id_key: str,
    sample_out: Optional[str],
    sample_size: int,
    include_text: bool,
    sample_include_u: bool,
    partition_index: Optional[int],
    partition_count: Optional[int],
    partition_seed: int,
    disjoint_partition_index: Optional[int],
    disjoint_partition_count: Optional[int],
    disjoint_partition_seed: Optional[int],
    write_out: Optional[str] = None,
) -> Dict:
    """
    Stream examples, apply RR, and compute observed flip rate on first N.
    Optionally write a small sample JSONL without raw text (unless permitted).
    If write_out is set, write each (partitioned, RR-flipped) example to that JSONL.
    """
    q = q_from_epsilon(epsilon)
    total = 0
    flips = 0
    sample_flipped: List[Dict] = []
    sample_kept: List[Dict] = []
    target_per_class = max(1, sample_size // 2)

    write_f = None
    if write_out:
        os.makedirs(os.path.dirname(write_out) or ".", exist_ok=True)
        write_f = open(write_out, "w", encoding="utf-8")

    try:
        for example in iterator:
            validate_example(example)
            example_id = get_example_id(example, id_key=id_key)
            if not in_partition(
                example_id=example_id,
                partition_index=partition_index,
                partition_count=partition_count,
                partition_seed=partition_seed,
            ):
                continue
            flipped_example, rr_flipped, _, _, u = apply_rr_flip(
                example, epsilon=epsilon, seed=seed, id_key=id_key, example_id=example_id
            )

            if write_f:
                rec = {"prompt": flipped_example["prompt"], "chosen": flipped_example["chosen"], "rejected": flipped_example["rejected"]}
                if id_key in example and example[id_key] is not None:
                    rec[id_key] = example[id_key]
                write_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            total += 1
            flips += int(rr_flipped)

            if sample_out and (len(sample_flipped) < target_per_class or len(sample_kept) < target_per_class):
                record = {
                    "example_id": example_id,
                    "rr_flipped": rr_flipped,
                    "rr_q": q,
                }
                if sample_include_u:
                    record["rr_u"] = u
                if include_text:
                    record.update(
                        {
                            "prompt": example.get("prompt", ""),
                            "chosen": example.get("chosen", ""),
                            "rejected": example.get("rejected", ""),
                        }
                    )
                if rr_flipped and len(sample_flipped) < target_per_class:
                    sample_flipped.append(record)
                elif not rr_flipped and len(sample_kept) < target_per_class:
                    sample_kept.append(record)

            if not write_out and total >= max_audit:
                break

    finally:
        if write_f:
            write_f.close()

    if sample_out:
        os.makedirs(os.path.dirname(sample_out) or ".", exist_ok=True)
        with open(sample_out, "w", encoding="utf-8") as f:
            for rec in sample_flipped + sample_kept:
                f.write(json.dumps(rec) + "\n")

    observed_flip_rate = (flips / total) if total > 0 else 0.0
    ci_low, ci_high = wilson_interval(observed_flip_rate, total) if total > 0 else (0.0, 1.0)
    noise_check_pass = (ci_low <= q <= ci_high) if total > 0 else False
    disjoint_by_construction = None
    if disjoint_partition_count is not None:
        disjoint_by_construction = (
            disjoint_partition_count == partition_count
            and disjoint_partition_seed == partition_seed
            and disjoint_partition_index is not None
            and partition_index is not None
            and disjoint_partition_index != partition_index
        )
    return {
        "epsilon": float(epsilon),
        "q": q,
        "observed_flip_rate": observed_flip_rate,
        "flip_rate_error": observed_flip_rate - q,
        "flip_rate_ci_low": ci_low,
        "flip_rate_ci_high": ci_high,
        "noise_check_pass": noise_check_pass,
        "N": total,
        "seed": int(seed),
        "rr_hash": "blake2b-8",
        "example_id_strategy": f"id:{id_key} else md5(prompt\\nchosen\\nrejected)",
        "partition_index": partition_index,
        "partition_count": partition_count,
        "partition_seed": int(partition_seed),
        "partition_hash": "blake2b-8",
        "disjoint_partition_index": disjoint_partition_index,
        "disjoint_partition_count": disjoint_partition_count,
        "disjoint_partition_seed": disjoint_partition_seed,
        "disjoint_by_construction": disjoint_by_construction,
    }


def write_audit_record(path: str, record: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_sample_path(base_path: Optional[str], epsilon: float, multi: bool) -> Optional[str]:
    if not base_path:
        return None
    if not multi:
        return base_path
    root, ext = os.path.splitext(base_path)
    tag = str(epsilon).replace(".", "p")
    return f"{root}_eps{tag}{ext or '.jsonl'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream RR flipping for DPO examples.")
    parser.add_argument("--epsilon", type=float, default=None, help="Single epsilon value.")
    parser.add_argument(
        "--eps_list",
        type=str,
        default=None,
        help="Comma-separated eps list, e.g., 0.5,1,2,4,8",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed for deterministic RR.")
    parser.add_argument("--max_audit", type=int, default=50000, help="Max examples for audit.")
    parser.add_argument("--audit_out", type=str, default="rr_audit.jsonl", help="Audit log path.")

    # Data source options
    parser.add_argument("--dataset_name", type=str, default=None, help="HF dataset name.")
    parser.add_argument("--dataset_config", type=str, default=None, help="HF dataset config.")
    parser.add_argument("--data_files", type=str, default=None, help="Comma-separated data files.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--revision", type=str, default=None, help="Dataset revision.")
    parser.add_argument("--input_jsonl", type=str, default=None, help="Stream from JSONL file.")
    parser.add_argument("--id_key", type=str, default="id", help="ID field name.")
    parser.add_argument(
        "--partition_count",
        type=int,
        default=None,
        help="Total number of deterministic partitions (stream-safe).",
    )
    parser.add_argument(
        "--partition_index",
        type=int,
        default=None,
        help="Partition index in [0, partition_count).",
    )
    parser.add_argument(
        "--partition_seed",
        type=int,
        default=None,
        help="Seed for partitioning (defaults to --seed).",
    )

    # Optional sample output
    parser.add_argument(
        "--sample_out",
        type=str,
        default=None,
        help="Optional JSONL sample output path.",
    )
    parser.add_argument("--sample_size", type=int, default=100, help="Sample size (total).")
    parser.add_argument(
        "--include_text",
        action="store_true",
        help="Include raw text in samples (off by default).",
    )
    parser.add_argument(
        "--sample_include_u",
        action="store_true",
        help="Include deterministic rr_u in samples.",
    )
    parser.add_argument(
        "--write_out",
        type=str,
        default=None,
        help="Write full partition (RR-flipped) to JSONL. Requires --partition_count and --partition_index.",
    )
    parser.add_argument(
        "--disjoint_partition_count",
        type=int,
        default=None,
        help="Partition count of the disjoint stage (audit only).",
    )
    parser.add_argument(
        "--disjoint_partition_index",
        type=int,
        default=None,
        help="Partition index of the disjoint stage (audit only).",
    )
    parser.add_argument(
        "--disjoint_partition_seed",
        type=int,
        default=None,
        help="Partition seed of the disjoint stage (audit only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epsilon is None and args.eps_list is None:
        raise ValueError("Provide --epsilon or --eps_list.")
    if args.epsilon is not None and args.eps_list is not None:
        raise ValueError("Provide only one of --epsilon or --eps_list.")
    if (args.partition_count is None) != (args.partition_index is None):
        raise ValueError("Provide both --partition_count and --partition_index.")
    if args.write_out and (args.partition_count is None or args.partition_index is None):
        raise ValueError("--write_out requires --partition_count and --partition_index.")
    if args.partition_count is not None:
        if args.partition_count <= 0:
            raise ValueError("--partition_count must be > 0.")
        if args.partition_index < 0 or args.partition_index >= args.partition_count:
            raise ValueError("--partition_index must be in [0, partition_count).")
    if (args.disjoint_partition_count is None) != (args.disjoint_partition_index is None):
        raise ValueError(
            "Provide both --disjoint_partition_count and --disjoint_partition_index."
        )

    epsilons: List[float]
    if args.eps_list is not None:
        epsilons = [float(e.strip()) for e in args.eps_list.split(",") if e.strip()]
    else:
        epsilons = [float(args.epsilon)]

    multi = len(epsilons) > 1
    partition_seed = args.seed if args.partition_seed is None else args.partition_seed
    for eps in epsilons:
        iterator = build_stream(args)
        sample_path = build_sample_path(args.sample_out, eps, multi)
        record = audit_stream(
            iterator=iterator,
            epsilon=eps,
            seed=args.seed,
            max_audit=args.max_audit,
            id_key=args.id_key,
            sample_out=sample_path,
            sample_size=args.sample_size,
            include_text=args.include_text,
            sample_include_u=args.sample_include_u,
            partition_index=args.partition_index,
            partition_count=args.partition_count,
            partition_seed=partition_seed,
            disjoint_partition_index=args.disjoint_partition_index,
            disjoint_partition_count=args.disjoint_partition_count,
            disjoint_partition_seed=args.disjoint_partition_seed,
            write_out=args.write_out,
        )
        write_audit_record(args.audit_out, record)
        print(json.dumps(record))


if __name__ == "__main__":
    main()
