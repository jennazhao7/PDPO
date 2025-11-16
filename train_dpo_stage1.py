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
    TrainingArguments,
)
from trl import DPOTrainer     # ✅ no DPOConfig import


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
    ap.add_argument("--model", default="openai-community/gpt2-large")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="./models/M1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=1)
    ap.add_argument("--max-prompt", type=int, default=64)
    ap.add_argument("--max-target", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--tiny", type=int, default=128)
    ap.add_argument("--one-step", action="store_true")
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
    model.to(device)
    print(f"[Model] device={next(model.parameters()).device}  "
          f"n_params≈{sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    # Training arguments
    per_device_bsz = 1 if args.debug else args.bsz
    ga = 1 if args.debug else args.ga
    epochs = 0.1 if args.debug else args.epochs

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.bsz,          # e.g., 2
        gradient_accumulation_steps=args.ga,           # e.g., 4
        num_train_epochs=args.epochs,                  # e.g., 3
        learning_rate=1e-5,                            # safer LR for DPO
        lr_scheduler_type="cosine",
        warmup_steps=50,
        optim="adamw_torch",
        gradient_checkpointing=True,
        fp16=True,                                     # use mixed precision
        max_grad_norm=1.0,
        save_strategy="epoch",                         # save after each epoch
        save_total_limit=2,
        logging_steps=10,
        report_to=None,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )


    # DPO trainer
    trainer = DPOTrainer(
        model=model,
        tokenizer=tok,
        args=targs,
        train_dataset=ds,
        reference_free=True,
        beta=0.1,
        loss_type="sigmoid",
        label_smoothing=0.0,
        max_length=args.max_len,
        max_prompt_length=args.max_prompt,
        max_target_length=args.max_target,
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
    print(f"[Train] bsz={per_device_bsz} GA={ga} GPUs={n_gpus} eff_bsz={eff_bsz} "
          f"max_len={args.max_len} (prompt={args.max_prompt}, target={args.max_target})")

    trainer.train()
    trainer.save_model(args.out)
    print(f"✅ Saved to {args.out}")
    return 0


if __name__ == "__main__":
    if os.getenv("DEBUG", "") == "1":
        os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
        os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")
    sys.exit(main())
