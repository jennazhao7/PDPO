#!/usr/bin/env python3
"""Cache base-model reference logprobs for ONE clean preference split, in fp32.

``precompute_ref_logps.py`` is manifest-driven: it expands ``experiments/manifest.yaml`` into
cells and names its output per cell. That is the wrong model here — Stage 1 trains clean DPO on a
single v3 split, there is no eps/seed cell to expand, and the manifest is not even part of the
Stage 1 source bundle. This script caches exactly one split to exactly one path.

The record schema is the one ``matched_core.load_reference_cache`` validates: ``pair_key``,
``base_model``, ``scoring_dtype``, ``truncation``, ``max_len``, ``prompt_hash``, ``chosen_hash``,
``rejected_hash``, ``chosen_ref_logp``, ``rejected_ref_logp``. Orientation hashes are what make a
cache reusable only for the exact rows it was built from.

Scoring uses the same validated tensor path as training and evaluation
(``encode_prompt_response`` + ``torch_batched_response_logprobs``), so cached reference logprobs
are bit-comparable with the policy logprobs they are subtracted from.

FP32 only: the project locks fp32 and both evaluators raise on anything else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2_debugging"))

from matched_core import pair_key, read_jsonl, text_hash  # noqa: E402
from ref_logprob_core import (  # noqa: E402
    REFERENCE_DTYPE,
    TRUNCATION_MODE,
    encode_prompt_response,
    torch_batched_response_logprobs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="clean preference split (jsonl)")
    parser.add_argument("--out", required=True, help="cache path (jsonl)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    # P1b: arm identity, written into the cache so cross-arm reuse is refused by a STATED
    # assertion rather than incidentally by a pair_key lookup miss. C3 showed the lookup-miss
    # defense is per-row and so requires at least one differing row; this does not.
    parser.add_argument("--rr-eps", type=float, required=True,
                        help="The arm this cache belongs to. Asserted on load.")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite existing reference cache: {out}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.cuda.is_available(), "Refusing to run on CPU"
    device = torch.device("cuda")

    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError(f"no rows in {args.data}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model = model.to(device).eval()

    def score(prompts_responses: List[tuple[str, str]]) -> List[float]:
        sequences, prompt_lens = [], []
        for prompt, response in prompts_responses:
            ids, prompt_len = encode_prompt_response(tokenizer, prompt, response, args.max_len)
            sequences.append(ids)
            prompt_lens.append(prompt_len)
        with torch.no_grad():
            values = torch_batched_response_logprobs(
                model, sequences, prompt_lens, tokenizer.pad_token_id, device
            )
        return [float(value.item()) for value in values]

    records: List[Dict[str, Any]] = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start : start + args.batch_size]
        chosen = score([(row["prompt"], row["chosen"]) for row in batch])
        rejected = score([(row["prompt"], row["rejected"]) for row in batch])
        for row, chosen_logp, rejected_logp in zip(batch, chosen, rejected):
            records.append(
                {
                    "pair_key": pair_key(row),
                    "cache_role": "train_reference_arm_orientation",
                    "rr_eps": args.rr_eps,
                    "base_model": args.model,
                    "scoring_dtype": REFERENCE_DTYPE,
                    "truncation": TRUNCATION_MODE,
                    "max_len": args.max_len,
                    "prompt_hash": text_hash(str(row["prompt"])),
                    "chosen_hash": text_hash(str(row["chosen"])),
                    "rejected_hash": text_hash(str(row["rejected"])),
                    "chosen_ref_logp": chosen_logp,
                    "rejected_ref_logp": rejected_logp,
                }
            )
        if (start // args.batch_size) % 25 == 0:
            print(f"  [{min(start + args.batch_size, len(rows))}/{len(rows)}]", flush=True)

    # Written atomically so a preemption cannot leave a truncated cache that later validates
    # row-by-row until it runs out.
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(out)
    print(f"wrote {len(records)} reference records to {out} (scoring_dtype={REFERENCE_DTYPE})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
