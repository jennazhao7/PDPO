#!/usr/bin/env python3
"""
Stage-2 LoRA DPO training on D2.

Loads a base model, attaches and freezes the Stage-1 adapter, then adds a new
Stage-2 adapter and trains it using DPO (not SFT) on the chosen/rejected pairs.
This enables preference signal recovery from noisy RR-flipped labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


DEFAULT_DATA = "lora/preprocessing/d2_rr_flipped.jsonl"


def load_causal_lm_compat(model_id: str, **kwargs):
    """Load CausalLM with safetensors-first strategy and torch<2.6 fallback."""
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True, **kwargs)
    except Exception as safe_exc:
        original_import_check = None
        original_modeling_check = None
        try:
            import transformers.utils.import_utils as import_utils
            import transformers.modeling_utils as modeling_utils
            original_import_check = import_utils.check_torch_load_is_safe
            original_modeling_check = modeling_utils.check_torch_load_is_safe
            import_utils.check_torch_load_is_safe = lambda: None
            modeling_utils.check_torch_load_is_safe = lambda: None
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            print(
                f"[Load] WARNING: safetensors unavailable ({safe_exc}); "
                "loaded via torch.load compatibility fallback."
            )
            return model
        finally:
            try:
                if original_import_check is not None:
                    import_utils.check_torch_load_is_safe = original_import_check
                if original_modeling_check is not None:
                    modeling_utils.check_torch_load_is_safe = original_modeling_check
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-2 LoRA DPO training")
    ap.add_argument("--model", required=True, help="Base model name or path")
    ap.add_argument("--stage1_adapter", required=True, help="Path to Stage-1 LoRA adapter")
    ap.add_argument("--stage1_subfolder", default=None)
    ap.add_argument("--data", default=DEFAULT_DATA, help="D2 JSONL with prompt/chosen/rejected")
    ap.add_argument("--out", default=None, help="Output dir for Stage-2 adapter")
    ap.add_argument("--manifest_out", default=None, help="Manifest JSON path")
    ap.add_argument("--output_suffix", default="")

    # DPO hyperparameters
    ap.add_argument("--beta", type=float, default=0.1, help="DPO beta (KL penalty)")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_prompt_length", type=int, default=256)
    ap.add_argument("--max_target_length", type=int, default=256)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=300)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules for stage2",
    )

    # Precision
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")

    # Smoke test
    ap.add_argument("--no_smoke_test", action="store_true")

    return ap.parse_args()


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    # Auto-derive output path
    if args.out is None:
        safe_model = args.model.replace("/", "--")
        suffix = args.output_suffix or "stage2_dpo"
        args.out = f"./outputs/{safe_model}_{suffix}"
    if args.manifest_out is None:
        args.manifest_out = os.path.join(args.out, "M2_manifest.json")

    os.makedirs(args.out, exist_ok=True)
    print(f"[Config] out={args.out} manifest={args.manifest_out}")

    # Load D2 data with prompt, chosen, rejected columns
    print(f"[Data] loading {args.data}")
    raw = load_dataset("json", data_files=args.data)["train"]

    # Ensure required columns exist
    cols = raw.column_names
    for req in ["prompt", "chosen", "rejected"]:
        if req not in cols:
            raise ValueError(f"D2 data missing required column '{req}'. Found: {cols}")

    n_total = len(raw)
    print(f"[Data] loaded {n_total} rows with columns {cols}")

    # Tokenizer
    use_fast = "open_llama" not in args.model.lower()
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Device & dtype
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        dtype = torch.float32
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Load base model
    print("[Model] Loading base model...")
    model = load_causal_lm_compat(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=None,
    )
    model.config.use_cache = False

    # Attach Stage-1 adapter and freeze it
    if args.stage1_subfolder:
        model = PeftModel.from_pretrained(
            model, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1"
        )
    else:
        model = PeftModel.from_pretrained(model, args.stage1_adapter, adapter_name="stage1")

    print("[Model] Merging Stage-1 adapter into base weights...")
    model = model.merge_and_unload()

    # Add Stage-2 adapter
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    stage2_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    from peft import get_peft_model
    model = get_peft_model(model, stage2_config, adapter_name="stage2")

    # Verify trainable params
    trainable = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    stage1_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ("stage1" in n and p.requires_grad)
    )
    print(f"[Trainable] trainable={trainable} total={total} ({100*trainable/max(1,total):.4f}%)")
    assert trainable > 0, "No trainable parameters found."
    assert stage1_trainable == 0, "Stage-1 adapter has trainable params — should be frozen."

    # DPO training config
    dpo_config = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=bool(cuda_ok and args.fp16),
        bf16=bool(cuda_ok and args.bf16),
        max_grad_norm=args.max_grad_norm,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        beta=args.beta,
        loss_type="sigmoid",
        max_length=args.max_len,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_target_length,
        seed=args.seed,
    )

    # DPO Trainer — ref_model=None uses implicit reference (PEFT)
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tok,
        args=dpo_config,
        train_dataset=raw,
    )

    print("[Train] Stage-2 LoRA DPO")
    print(f"  bsz={args.bsz} GA={args.ga} beta={args.beta}")
    print(f"  max_len={args.max_len} (prompt={args.max_prompt_length}, target={args.max_target_length})")
    trainer.train()

    # Save Stage-2 adapter only
    model.save_pretrained(args.out, adapter_name="stage2")

    # Write manifest
    manifest = {
        "base_model": args.model,
        "adapters": [
            {"name": "stage1", "path": os.path.abspath(args.stage1_adapter)},
            {"name": "stage2", "path": os.path.abspath(args.out)},
        ],
        "adapter_order": ["stage1", "stage2"],
        "notes": "M2 = base + stage1 + stage2 (Stage-2 trained with DPO)",
    }
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Saved Stage-2 (DPO) adapter to {args.out}")
    print(f"✅ Wrote manifest to {args.manifest_out}")

    # Save data stats
    stats = {"N": n_total, "method": "DPO", "beta": args.beta}
    with open(os.path.join(args.out, "stage2_data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
