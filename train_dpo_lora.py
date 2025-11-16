# train_dpo_lora.py
# DPO training script with LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
# Specifically designed for larger models like pythia-1b
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
from peft import LoraConfig, get_peft_model, TaskType
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
    ap = argparse.ArgumentParser(description="DPO training with LoRA")
    ap.add_argument("--model", default="EleutherAI/pythia-1b")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./models/pythia-1b-dpo-lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--max-prompt", type=int, default=256)
    ap.add_argument("--max-target", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (scaling factor)")
    ap.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--tiny", type=int, default=128)
    ap.add_argument("--one-step", action="store_true")
    return ap.parse_args()


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    # Device - Use single GPU for LoRA training (memory-efficient enough)
    # For multi-GPU, use torchrun instead
    cuda_ok = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if cuda_ok else 0
    
    # Check if we're in a distributed training environment
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        # Distributed training via torchrun
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Device] Distributed training: local_rank={local_rank}, using {device}")
    else:
        # Single GPU training
        device = torch.device("cuda:0" if (cuda_ok and n_gpus >= 1) else "cpu")
        print(f"[Device] Single GPU: cuda_available={cuda_ok} visible_gpus={n_gpus} → using {device}")

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
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        device_map=None,
    )
    model.config.use_cache = False
    try:
        model.config.attn_implementation = "eager"
    except Exception:
        pass
    
    # Configure LoRA
    print(f"[LoRA] Configuring LoRA adapters...")
    print(f"  Rank (r): {args.lora_r}")
    print(f"  Alpha: {args.lora_alpha}")
    print(f"  Dropout: {args.lora_dropout}")
    
    # Determine target modules based on model architecture
    # First, check the actual model type
    model_type = model.config.model_type.lower()
    print(f"  Model type: {model_type}")
    
    # Auto-detect target modules by inspecting the model structure
    layer_names = [name for name, _ in model.named_modules()]
    
    # Check for GPTNeoX/Pythia architecture (query_key_value)
    if any("query_key_value" in name for name in layer_names):
        target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        print(f"  Detected GPTNeoX/Pythia architecture")
    # Check for GPT-2 architecture (c_attn)
    elif any("c_attn" in name for name in layer_names):
        target_modules = ["c_attn", "c_proj", "c_fc"]
        print(f"  Detected GPT-2 architecture")
    # Check for LLaMA/GPT-Neo architecture (separate q/k/v)
    elif any("q_proj" in name for name in layer_names):
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
        # Add MLP layers if they exist
        if any("gate_proj" in name for name in layer_names):
            target_modules.extend(["gate_proj", "up_proj", "down_proj"])
        print(f"  Detected LLaMA/GPT-Neo architecture")
    else:
        # Fallback: try model_type string matching
        if "gptneox" in model_type or "pythia" in model_type:
            target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
        elif "gpt2" in model_type:
            target_modules = ["c_attn", "c_proj", "c_fc"]
        else:
            # Last resort: show available modules and suggest
            print(f"  Warning: Could not auto-detect modules. Available layer patterns:")
            unique_patterns = set()
            for name in layer_names[:50]:  # Show first 50
                parts = name.split('.')
                if len(parts) > 1:
                    unique_patterns.add(parts[-1])
            print(f"    Sample patterns: {sorted(list(unique_patterns))[:10]}")
            raise ValueError(f"Could not determine target modules for model type '{model_type}'. Please specify manually.")
    
    print(f"  Target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    model.to(device)
    print(f"[Model] device={next(model.parameters()).device}  "
          f"n_params≈{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # DPO training arguments (using DPOConfig instead of TrainingArguments)
    per_device_bsz = 1 if args.debug else args.bsz
    ga = 1 if args.debug else args.ga
    epochs = 0.1 if args.debug else args.epochs

    targs = DPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.ga,
        num_train_epochs=args.epochs,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=True,  # Use mixed precision
        max_grad_norm=1.0,
        save_strategy="epoch",
        save_total_limit=2,
        logging_steps=10,
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Reduced from 2 to save memory
        dataloader_pin_memory=False,  # Disabled to save memory
        ddp_find_unused_parameters=False,  # Optimize for DDP (LoRA only trains adapter params)
        # DPO-specific parameters
        beta=0.1,
        loss_type="sigmoid",
        max_length=args.max_len,
        max_prompt_length=args.max_prompt,
        max_completion_length=args.max_target,  # Note: max_completion_length instead of max_target_length
    )

    # DPO trainer
    # Note: Newer versions of trl (0.25.1+) use 'processing_class' instead of 'tokenizer'
    # and 'ref_model=None' instead of 'reference_free=True'
    # DPO-specific parameters are now set in DPOConfig (targs) instead of DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # reference-free mode
        processing_class=tok,
        args=targs,
        train_dataset=ds,
    )

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
    print(f"[Train] LoRA DPO Training")
    print(f"  bsz={per_device_bsz} GA={ga} GPUs={n_gpus} eff_bsz={eff_bsz}")
    print(f"  max_len={args.max_len} (prompt={args.max_prompt}, target={args.max_target})")
    print(f"  LoRA r={args.lora_r}, alpha={args.lora_alpha}")

    trainer.train()
    
    # Save model (LoRA adapters will be saved)
    trainer.save_model(args.out)
    print(f"✅ Saved LoRA adapters to {args.out}")
    print(f"💡 To load: model = PeftModel.from_pretrained(base_model, '{args.out}')")
    return 0


if __name__ == "__main__":
    if os.getenv("DEBUG", "") == "1":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    sys.exit(main())

