# train_dpo_stage1.py
import os
import sys
import argparse
from typing import List
#!/usr/bin/env python3
# train_dpo_stage1.py
import os
import sys
import argparse
from typing import List

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import DPOTrainer, DPOConfig


# ---------- Helpers ----------
ALLOWED_PROMPT_KEYS   = ["prompt", "question", "instruction"]
ALLOWED_CHOSEN_KEYS   = ["chosen", "chosen_response", "better, accepted", "accepted"]
ALLOWED_REJECTED_KEYS = ["rejected", "rejected_response", "worse", "rejected_text"]


def find_first_key(cols: List[str], candidates: List[str]) -> str:
    for k in candidates:
        if k in cols:
            return k
    return ""


def remap_columns(ds: Dataset) -> Dataset:
    cols = ds.column_names
    p = find_first_key(cols, ALLOWED_PROMPT_KEYS)
    c = find_first_key(cols, ALLOWED_CHOSEN_KEYS)
    r = find_first_key(cols, ALLOWED_REJECTED_KEYS)

    missing = []
    if not p: missing.append(f"prompt∈{ALLOWED_PROMPT_KEYS}")
    if not c: missing.append(f"chosen∈{ALLOWED_CHOSEN_KEYS}")
    if not r: missing.append(f"rejected∈{ALLOWED_REJECTED_KEYS}")
    if missing:
        raise ValueError(
            "Dataset columns not found. Need prompt/chosen/rejected.\n"
            f"Found columns: {cols}\n"
            f"Missing: {', '.join(missing)}"
        )

    rename_map = {}
    if p != "prompt":   rename_map[p] = "prompt"
    if c != "chosen":   rename_map[c] = "chosen"
    if r != "rejected": rename_map[r] = "rejected"
    if rename_map:
        ds = ds.rename_columns(rename_map)
    keep = ["prompt", "chosen", "rejected"]
    drop = [x for x in ds.column_names if x not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


# ---------- Main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2-large", help="Base policy to DPO fine-tune.")
    ap.add_argument("--ref-model", default=None, help="Optional reference model path/name. If omitted, runs reference-free.")
    ap.add_argument("--data", required=True, help="Path to DPO JSONL with prompt/chosen/rejected.")
    ap.add_argument("--out", default="./models/M1", help="Output dir for checkpoints.")
    ap.add_argument("--epochs", type=float, default=1.0, help="Number of epochs.")
    ap.add_argument("--bsz", type=int, default=1, help="Per-device train batch size.")
    ap.add_argument("--ga", type=int, default=1, help="Gradient accumulation steps.")
    ap.add_argument("--max-prompt", type=int, default=64, help="Max prompt tokens.")
    ap.add_argument("--max-target", type=int, default=64, help="Max completion tokens.")
    ap.add_argument("--max-len", type=int, default=128, help="Max total sequence length.")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (set 0 if CPU-bound).")
    ap.add_argument("--pin-memory", action="store_true", help="Pin DataLoader memory (recommended for GPU).")
    ap.add_argument("--device-map", default="auto", help='Hf device_map (e.g., "auto", "balanced", or "none").')
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype load and AMP choice.")
    ap.add_argument("--debug", action="store_true", help="Debug mode with tiny subset and short run.")
    ap.add_argument("--tiny", type=int, default=128, help="Subset size for debug.")
    ap.add_argument("--one-step", action="store_true", help="Run a single training step then exit.")
    return ap.parse_args()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    # Device
    cuda_ok = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_ok else 0
    device = torch.device("cuda:0" if (cuda_ok and n_gpus >= 1) else "cpu")
    print(f"[Device] cuda_available={cuda_ok} visible_gpus={n_gpus} → using {device}")

    # Dataset
    print(f"[Data] loading {args.data}")
    raw = load_dataset("json", data_files=args.data)["train"]
    ds = remap_columns(raw)
    if args.debug:
        ds = ds.select(range(min(args.tiny, len(ds))))
        print(f"[Data] debug mode: subset to {len(ds)} examples")
    else:
        print(f"[Data] full size: {len(ds)} examples")

    ex0 = ds[0]
    print(f"[Data] sample[0] keys={list(ex0.keys())}")
    for k in ["prompt", "chosen", "rejected"]:
        txt = ex0[k]
        preview = txt[:120].replace("\n", " ")
        ellipsis = "..." if len(txt) > 120 else ""
        print(f"  {k}: {preview}{ellipsis}")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Model
    print("[Model] loading…")
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]
    
    # Use device_map only if explicitly set (not "none")
    use_device_map = args.device_map != "none"
    
    # Load policy in FP32 when training with autocast to avoid FP16 grad-scaler issues
    if args.dtype == "float16":
        load_dtype = torch.float32
    elif args.dtype == "bfloat16":
        load_dtype = torch.float32
    elif args.dtype == "float32":
        load_dtype = torch.float32
    else:  # auto
        load_dtype = None
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=args.device_map if use_device_map else None,
    )

    model.config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    
    # Only move to device if not using device_map (device_map handles placement)
    if not use_device_map:
        model.to(device)
    
    print(f"[Model] device={next(model.parameters()).device}  "
          f"n_params≈{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # DPO training arguments (using DPOConfig instead of TrainingArguments)
    per_device_bsz = 1 if args.debug else args.bsz
    ga = 1 if args.debug else args.ga
    epochs = 0.1 if args.debug else args.epochs

    # Mixed precision flags aligned to requested dtype
    # When using device_map, we load model in FP32 and use autocast for FP16/BF16
    use_fp16 = args.dtype == "float16"
    use_bf16 = args.dtype == "bfloat16"

    targs = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=per_device_bsz,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=1.0,
        save_strategy="epoch",
        save_total_limit=1,  # Keep only final checkpoint to save disk space
        logging_steps=10,
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        # DPO-specific parameters
        beta=0.1,
        loss_type="sigmoid",
        max_length=args.max_len,
        max_prompt_length=args.max_prompt,
        # TRL version here expects max_completion_length (not max_target_length)
        max_completion_length=args.max_target,
    )

    # DPO trainer
    # Note: Newer versions of trl use 'processing_class' instead of 'tokenizer'
    # and 'ref_model=None' instead of 'reference_free=True'
    # DPO-specific parameters are now set in DPOConfig (targs) instead of DPOTrainer
    ref_model = None
    if args.ref_model:
        print(f"[Model] loading ref model: {args.ref_model}")
        # Use same dtype logic as main model
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            dtype=load_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            device_map=args.device_map if use_device_map else None,
        )
        ref_model.config.use_cache = False
        try:
            ref_model.config.attn_implementation = "eager"
        except Exception:
            pass
        # Only move to device if not using device_map
        if not use_device_map:
            ref_model.to(device)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # reference-free if None
        processing_class=tok,  # newer API uses processing_class instead of tokenizer
        args=targs,
        train_dataset=ds,
    )
    
    # Disable model card creation to avoid disk space issues
    def noop_create_model_card(*args, **kwargs):
        pass
    trainer.create_model_card = noop_create_model_card

    # One-step dry run
    if args.one_step:
        print("[Debug] ONE-STEP: building dataloader…")
        dl = trainer.get_train_dataloader()
        b = next(iter(dl))
        print(f"[Debug] batch keys: {list(b.keys())}")
        for k, v in b.items():
            if torch.is_tensor(v):
                assert v.ndim == 2, f"{k} must be [B,T], got {v.shape}"
                assert not torch.isnan(v.float()).any(), f"NaN in {k}"
        print("[Debug] running a single training step…")
        trainer.training_step(model, b)
        print("✅ One-step dry run passed. Exiting.")
        return 0

    # Train
    eff_bsz = per_device_bsz * ga * max(1, n_gpus)
    print(f"[Train] bsz={per_device_bsz} GA={ga} GPUs={n_gpus} eff_bsz={eff_bsz} "
          f"max_len={args.max_len} (prompt={args.max_prompt}, target={args.max_target})")

    try:
        trainer.train()
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"⚠️  Disk space error during training: {e}")
            print("Attempting to save model despite error...")
        else:
            raise
    
    # Save model (skip model card creation which is already disabled)
    try:
        trainer.save_model(args.out)
        print(f"✅ Saved to {args.out}")
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"⚠️  Disk space error when saving model: {e}")
            print(f"⚠️  Model may be partially saved. Check {args.out}")
            return 1
        else:
            raise
    
    return 0


if __name__ == "__main__":
    if os.getenv("DEBUG", "") == "1":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    sys.exit(main())
import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from trl import DPOTrainer, DPOConfig


# ---------- Helpers ----------
ALLOWED_PROMPT_KEYS   = ["prompt", "question", "instruction"]
ALLOWED_CHOSEN_KEYS   = ["chosen", "chosen_response", "better", "accepted"]
ALLOWED_REJECTED_KEYS = ["rejected", "rejected_response", "worse", "rejected_text"]


def find_first_key(cols: List[str], candidates: List[str]) -> str:
    for k in candidates:
        if k in cols:
            return k
    return ""


def remap_columns(ds: Dataset) -> Dataset:
    cols = ds.column_names
    p = find_first_key(cols, ALLOWED_PROMPT_KEYS)
    c = find_first_key(cols, ALLOWED_CHOSEN_KEYS)
    r = find_first_key(cols, ALLOWED_REJECTED_KEYS)

    missing = []
    if not p: missing.append(f"prompt∈{ALLOWED_PROMPT_KEYS}")
    if not c: missing.append(f"chosen∈{ALLOWED_CHOSEN_KEYS}")
    if not r: missing.append(f"rejected∈{ALLOWED_REJECTED_KEYS}")
    if missing:
        raise ValueError(
            "Dataset columns not found. Need prompt/chosen/rejected.\n"
            f"Found columns: {cols}\n"
            f"Missing: {', '.join(missing)}"
        )

    rename_map = {}
    if p != "prompt":   rename_map[p] = "prompt"
    if c != "chosen":   rename_map[c] = "chosen"
    if r != "rejected": rename_map[r] = "rejected"
    if rename_map:
        ds = ds.rename_columns(rename_map)
    keep = ["prompt", "chosen", "rejected"]
    drop = [x for x in ds.column_names if x not in keep]
    if drop:
        ds = ds.remove_columns(drop)
    return ds


# ---------- Main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai-community/gpt2-large", help="Base policy to DPO fine-tune.")
    ap.add_argument("--ref-model", default=None, help="Optional reference model path/name. If omitted, runs reference-free.")
    ap.add_argument("--data", required=True, help="Path to DPO JSONL with prompt/chosen/rejected.")
    ap.add_argument("--out", default="./models/M1", help="Output dir for checkpoints.")
    ap.add_argument("--epochs", type=float, default=1.0, help="Number of epochs.")
    ap.add_argument("--bsz", type=int, default=1, help="Per-device train batch size.")
    ap.add_argument("--ga", type=int, default=1, help="Gradient accumulation steps.")
    ap.add_argument("--max-prompt", type=int, default=64, help="Max prompt tokens.")
    ap.add_argument("--max-target", type=int, default=64, help="Max completion tokens.")
    ap.add_argument("--max-len", type=int, default=128, help="Max total sequence length.")
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (set 0 if CPU-bound).")
    ap.add_argument("--pin-memory", action="store_true", help="Pin DataLoader memory (recommended for GPU).")
    ap.add_argument("--device-map", default="auto", help='Hf device_map (e.g., "auto", "balanced", or "none").')
    ap.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype load and AMP choice.")
    ap.add_argument("--debug", action="store_true", help="Debug mode with tiny subset and short run.")
    ap.add_argument("--tiny", type=int, default=128, help="Subset size for debug.")
    ap.add_argument("--one-step", action="store_true", help="Run a single training step then exit.")
    return ap.parse_args()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    # Device
    cuda_ok = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_ok else 0
    device = torch.device("cuda:0" if (cuda_ok and n_gpus >= 1) else "cpu")
    print(f"[Device] cuda_available={cuda_ok} visible_gpus={n_gpus} → using {device}")

    # Dataset
    print(f"[Data] loading {args.data}")
    raw = load_dataset("json", data_files=args.data)["train"]
    ds = remap_columns(raw)
    if args.debug:
        ds = ds.select(range(min(args.tiny, len(ds))))
        print(f"[Data] debug mode: subset to {len(ds)} examples")
    else:
        print(f"[Data] full size: {len(ds)} examples")

    ex0 = ds[0]
    print(f"[Data] sample[0] keys={list(ex0.keys())}")
    for k in ["prompt", "chosen", "rejected"]:
        txt = ex0[k]
        preview = txt[:120].replace("\n", " ")
        ellipsis = "..." if len(txt) > 120 else ""
        print(f"  {k}: {preview}{ellipsis}")



    # Tokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Model
    print("[Model] loading…")
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]
    
    # Use device_map only if explicitly set (not "none")
    use_device_map = args.device_map != "none"
    
    # When using device_map with FP16/BF16, load in FP32 to avoid gradient scaling issues
    # Autocast will handle the conversion during forward pass
    if use_device_map and args.dtype in ["float16", "bfloat16"]:
        print(f"[Model] Using device_map with {args.dtype} - loading in FP32, autocast will handle conversion")
        load_dtype = torch.float32
    else:
        load_dtype = model_dtype if model_dtype != "auto" else None
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=load_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=args.device_map if use_device_map else None,
    )

    model.config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    
    # Only move to device if not using device_map (device_map handles placement)
    if not use_device_map:
        model.to(device)
    
    print(f"[Model] device={next(model.parameters()).device}  "
          f"n_params≈{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # DPO training arguments (using DPOConfig instead of TrainingArguments)
    per_device_bsz = 1 if args.debug else args.bsz
    ga = 1 if args.debug else args.ga
    epochs = 0.1 if args.debug else args.epochs

    # Mixed precision flags aligned to requested dtype
    # When using device_map, we load model in FP32 and use autocast for FP16/BF16
    use_fp16 = args.dtype == "float16"
    use_bf16 = args.dtype == "bfloat16"

    targs = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=per_device_bsz,
        gradient_accumulation_steps=ga,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=1.0,
        save_strategy="epoch",
        save_total_limit=1,  # Keep only final checkpoint to save disk space
        logging_steps=10,
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=args.pin_memory,
        # DPO-specific parameters
        beta=0.1,
        loss_type="sigmoid",
        max_length=args.max_len,
        max_prompt_length=args.max_prompt,
        # TRL version here expects max_completion_length (not max_target_length)
        max_completion_length=args.max_target,
    )

    # DPO trainer
    # Note: Newer versions of trl use 'processing_class' instead of 'tokenizer'
    # and 'ref_model=None' instead of 'reference_free=True'
    # DPO-specific parameters are now set in DPOConfig (targs) instead of DPOTrainer
    ref_model = None
    if args.ref_model:
        print(f"[Model] loading ref model: {args.ref_model}")
        # Use same dtype logic as main model
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            dtype=load_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            device_map=args.device_map if use_device_map else None,
        )
        ref_model.config.use_cache = False
        try:
            ref_model.config.attn_implementation = "eager"
        except Exception:
            pass
        # Only move to device if not using device_map
        if not use_device_map:
            ref_model.to(device)

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,  # reference-free if None
        processing_class=tok,  # newer API uses processing_class instead of tokenizer
        args=targs,
        train_dataset=ds,
    )
    
    # Disable model card creation to avoid disk space issues
    def noop_create_model_card(*args, **kwargs):
        pass
    trainer.create_model_card = noop_create_model_card

    # One-step dry run
    if args.one_step:
        print("[Debug] ONE-STEP: building dataloader…")
        dl = trainer.get_train_dataloader()
        b = next(iter(dl))
        print(f"[Debug] batch keys: {list(b.keys())}")
        for k, v in b.items():
            if torch.is_tensor(v):
                assert v.ndim == 2, f"{k} must be [B,T], got {v.shape}"
                assert not torch.isnan(v.float()).any(), f"NaN in {k}"
        print("[Debug] running a single training step…")
        trainer.training_step(model, b)
        print("✅ One-step dry run passed. Exiting.")
        return 0

    # Train
    eff_bsz = per_device_bsz * ga * max(1, n_gpus)
    print(f"[Train] bsz={per_device_bsz} GA={ga} GPUs={n_gpus} eff_bsz={eff_bsz} "
          f"max_len={args.max_len} (prompt={args.max_prompt}, target={args.max_target})")

    try:
        trainer.train()
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"⚠️  Disk space error during training: {e}")
            print("Attempting to save model despite error...")
        else:
            raise
    
    # Save model (skip model card creation which is already disabled)
    try:
        trainer.save_model(args.out)
        print(f"✅ Saved to {args.out}")
    except OSError as e:
        if "No space left on device" in str(e):
            print(f"⚠️  Disk space error when saving model: {e}")
            print(f"⚠️  Model may be partially saved. Check {args.out}")
            return 1
        else:
            raise
    
    return 0


if __name__ == "__main__":
    if os.getenv("DEBUG", "") == "1":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    sys.exit(main())
