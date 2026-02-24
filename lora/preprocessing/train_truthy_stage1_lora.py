#!/usr/bin/env python3
"""
Stage-1 LoRA DPO training on the Truthy-DPO 1016-pair subset.

Deterministically selects the first half (~508) of the subset, and optionally
applies Randomized Response (RR) swaps on the fly. No privatized dataset is
written to disk; only LoRA adapters and minimal logs are saved.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


ALLOWED_PROMPT_KEYS = ["prompt", "question", "instruction"]
ALLOWED_CHOSEN_KEYS = ["chosen", "chosen_response", "better", "accepted"]
ALLOWED_REJECTED_KEYS = ["rejected", "rejected_response", "worse", "rejected_text"]
# Default: D1 output from rr_stream_flip (partition 0, RR-flipped). Run rr_stream_flip first.
DEFAULT_DATA = "lora/preprocessing/d1_rr_flipped.jsonl"


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


def find_first_key(cols: List[str], candidates: List[str]) -> str:
    for key in candidates:
        if key in cols:
            return key
    return ""


def remap_columns(ds: Dataset) -> Dataset:
    cols = ds.column_names
    p = find_first_key(cols, ALLOWED_PROMPT_KEYS)
    c = find_first_key(cols, ALLOWED_CHOSEN_KEYS)
    r = find_first_key(cols, ALLOWED_REJECTED_KEYS)

    missing = []
    if not p:
        missing.append(f"prompt∈{ALLOWED_PROMPT_KEYS}")
    if not c:
        missing.append(f"chosen∈{ALLOWED_CHOSEN_KEYS}")
    if not r:
        missing.append(f"rejected∈{ALLOWED_REJECTED_KEYS}")
    if missing:
        raise ValueError(
            "Dataset columns not found. Need prompt/chosen/rejected.\n"
            f"Found columns: {cols}\n"
            f"Missing: {', '.join(missing)}"
        )

    rename_map = {}
    if p != "prompt":
        rename_map[p] = "prompt"
    if c != "chosen":
        rename_map[c] = "chosen"
    if r != "rejected":
        rename_map[r] = "rejected"
    if rename_map:
        ds = ds.rename_columns(rename_map)
    keep = ["prompt", "chosen", "rejected"]
    drop = [x for x in ds.column_names if x not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


def parse_args():
    ap = argparse.ArgumentParser(
        description="Stage-1 LoRA DPO training on Truthy-DPO 1016 subset."
    )
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-prompt", type=int, default=256)
    ap.add_argument("--max-target", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--id-key", type=str, default="id")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--one-step", action="store_true")
    return ap.parse_args()


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    # Output path (auto-derive if not provided)
    if args.out is None:
        safe_model = args.model.replace("/", "--")
        args.out = f"./models/{safe_model}-truthy-stage1-lora"

    # Dataset: D1 output from rr_stream_flip (already partitioned + RR-flipped)
    data_path = args.data
    print(f"[Data] loading {data_path} (expect D1 from rr_stream_flip)")
    raw = load_dataset("json", data_files=data_path)["train"]
    ds = remap_columns(raw)
    print(f"[Data] N={len(ds)} (D1 partition, RR-flipped by rr_stream_flip)")

    # Tokenizer
    # OpenLLaMA model card recommends slow tokenizer to avoid tokenization mismatch.
    use_fast = False if "open_llama" in args.model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Device
    cuda_ok = torch.cuda.is_available()
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Device] Distributed: local_rank={local_rank} device={device}")
    else:
        device = torch.device("cuda:0" if cuda_ok else "cpu")
        print(f"[Device] Single: cuda={cuda_ok} device={device}")

    # Model
    dtype = torch.float16 if cuda_ok else torch.float32
    print("[Model] loading…")
    model = load_causal_lm_compat(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=None,
    )
    model.config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass

    # LoRA setup (default attention-only for Llama-3.2-1B sanity baseline)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--target-modules must include at least one module.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # DPO training args
    per_device_bsz = 1 if args.debug else args.bsz
    ga = 1 if args.debug else args.ga
    epochs = 0.1 if args.debug else args.epochs

    targs = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=per_device_bsz,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=bool(cuda_ok),
        max_grad_norm=1.0,
        save_strategy="no",
        logging_steps=10,
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
        beta=0.1,
        loss_type="sigmoid",
        max_length=args.max_len,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_target,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        processing_class=tok,
        args=targs,
        train_dataset=ds,
    )

    if args.one_step:
        print("[Debug] ONE-STEP: running a single training step…")
        dl = trainer.get_train_dataloader()
        batch = next(iter(dl))
        trainer.training_step(model, batch)
        print("✅ One-step dry run passed. Exiting.")
        return 0

    print("[Train] Stage-1 LoRA DPO")
    print(f"  bsz={per_device_bsz} GA={ga}")
    print(f"  max_len={args.max_len} (prompt={args.max_prompt}, target={args.max_target})")
    trainer.train()

    trainer.save_model(args.out)
    print(f"✅ Saved LoRA adapters to {args.out}")
    return 0


if __name__ == "__main__":
    if os.getenv("DEBUG", "") == "1":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    sys.exit(main())
