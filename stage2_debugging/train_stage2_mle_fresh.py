#!/usr/bin/env python3
"""
MLE-FreshRetrain (architectural control).

Fresh LoRA on base model + standard DPO on raw noisy D2 labels.
No M1 scoring, no MAP relabeling, no Soft-Bayes weights.

This isolates pure architecture effect:
  - Policy: base + fresh LoRA
  - Reference: base (implicit PEFT ref with ref_model=None)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLE-FreshRetrain on raw noisy D2")
    p.add_argument("--model", required=True, help="Base model ID/path")
    p.add_argument("--data", required=True, help="Noisy D2 JSONL with prompt/chosen/rejected")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--epsilon", type=float, default=1.0, help="RR epsilon (for metadata only)")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--bsz", type=int, default=1)
    p.add_argument("--ga", type=int, default=16)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("=== MLE-FreshRetrain (raw noisy D2) ===")
    print(f"model={args.model}")
    print(f"data={args.data}")
    print(f"out={args.out}")
    print(f"epsilon={args.epsilon} beta={args.beta} lr={args.lr}")

    rows = [json.loads(l) for l in open(args.data, "r", encoding="utf-8") if l.strip()]
    if not rows:
        raise ValueError(f"Empty data file: {args.data}")
    for i, r in enumerate(rows[:5]):
        for k in ("prompt", "chosen", "rejected"):
            if k not in r:
                raise KeyError(f"Row {i} missing key '{k}' in {args.data}")
    ds = Dataset.from_list(
        [{"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]
    )
    print(f"[Data] n_pairs={len(ds)}")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.bf16:
        dtype = torch.bfloat16
    else:
        # Keep default auto behavior when bf16 is not requested.
        dtype = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)

    train_args = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=10,
        warmup_steps=10,
        bf16=args.bf16,
        beta=args.beta,
        max_length=args.max_len,
        max_prompt_length=args.max_len // 2,
        seed=args.seed,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT implicit reference = base model
        train_dataset=ds,
        processing_class=tok,
        args=train_args,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.out, "fresh_lora"))

    run_stats = {
        "method": "mle_fresh_retrain",
        "n_pairs": len(ds),
        "epsilon": args.epsilon,
        "beta": args.beta,
        "lr": args.lr,
        "seed": args.seed,
    }
    with open(os.path.join(args.out, "run_stats.json"), "w", encoding="utf-8") as f:
        json.dump(run_stats, f, indent=2)

    manifest = {
        "base_model": args.model,
        "method": "mle_fresh_single_lora",
        "adapters": [
            {"name": "fresh_lora", "path": os.path.abspath(os.path.join(args.out, "fresh_lora"))},
        ],
        "note": "Single LoRA from base model on raw noisy D2. Eval uses base as reference.",
    }
    with open(os.path.join(args.out, "M2_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ MLE-FreshRetrain complete: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

