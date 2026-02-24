#!/usr/bin/env python3
"""
Plain LoRA fine-tuning on a preference (DPO) or instruction (SFT) dataset.

Produces a reproducible, non-DP baseline that is directly comparable to a
DP-LoRA run using the same splits, tokenisation, and evaluation.

Outputs written to --output_dir:
    adapter_model.safetensors   LoRA weights
    adapter_config.json         PEFT config
    config.json                 Full run config + git hash
    split_indices.json          Train / val index lists (for DP fair comparison)
    metrics.jsonl               Per-step metrics
    summary.json                Final metrics
    generations.jsonl           Fixed-prompt generation samples
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMPT_CANDIDATES = ["prompt", "question", "instruction"]
CHOSEN_CANDIDATES = ["chosen", "chosen_response", "better", "accepted"]
REJECTED_CANDIDATES = ["rejected", "rejected_response", "worse", "rejected_text"]
RESPONSE_CANDIDATES = ["response", "answer", "output", "completion"]

DEFAULT_EVAL_PROMPTS = [
    "Explain quantum entanglement in simple terms.",
    "Write a short poem about the ocean.",
    "What are three tips for effective time management?",
    "Translate 'Good morning, how are you?' into French.",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "What is the difference between machine learning and deep learning?",
    "Give me a healthy breakfast recipe.",
    "Why is the sky blue?",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set seed for reproducibility across python, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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


def get_git_hash() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        return None


def find_key(columns: List[str], candidates: List[str]) -> str:
    for c in candidates:
        if c in columns:
            return c
    return ""


def detect_dataset_type(columns: List[str]) -> str:
    """Return 'dpo' if preference columns found, else 'sft'."""
    has_chosen = bool(find_key(columns, CHOSEN_CANDIDATES))
    has_rejected = bool(find_key(columns, REJECTED_CANDIDATES))
    if has_chosen and has_rejected:
        return "dpo"
    return "sft"


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def stable_split_indices(
    n: int, val_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    """Deterministic train/val split by index."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_frac))
    return indices[n_val:], indices[:n_val]


# ---------------------------------------------------------------------------
# Metrics callback — writes per-step to metrics.jsonl
# ---------------------------------------------------------------------------

class MetricsLogger(TrainerCallback):
    def __init__(self, output_dir: str):
        self.path = os.path.join(output_dir, "metrics.jsonl")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, "epoch": state.epoch}
        record.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        append_jsonl(self.path, record)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequence_logprob(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    device: torch.device,
    max_len: int,
) -> float:
    """Mean log P(response | prompt) over response tokens."""
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    response_ids = tok.encode(response, add_special_tokens=False)
    input_ids = (prompt_ids + response_ids)[:max_len]
    input_ids_t = torch.tensor([input_ids], device=device)

    logits = model(input_ids=input_ids_t).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids_t[:, 1:]

    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = len(prompt_ids)
    if prompt_len >= input_ids_t.shape[1]:
        return 0.0
    response_lp = token_lp[0, prompt_len - 1:]
    if response_lp.numel() == 0:
        return 0.0
    return response_lp.mean().item()


def eval_pairwise_accuracy(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    ds: Dataset,
    prompt_key: str,
    chosen_key: str,
    rejected_key: str,
    device: torch.device,
    max_len: int,
    max_eval: int = 200,
) -> Dict[str, float]:
    """Compute chosen-vs-rejected pairwise accuracy on held-out split."""
    model.eval()
    correct = 0
    total = 0
    chosen_lps: List[float] = []
    rejected_lps: List[float] = []

    n = min(len(ds), max_eval)
    for i in range(n):
        row = ds[i]
        c_lp = sequence_logprob(model, tok, row[prompt_key], row[chosen_key], device, max_len)
        r_lp = sequence_logprob(model, tok, row[prompt_key], row[rejected_key], device, max_len)
        chosen_lps.append(c_lp)
        rejected_lps.append(r_lp)
        if c_lp > r_lp:
            correct += 1
        total += 1
        if (i + 1) % 50 == 0:
            print(f"  [eval] {i+1}/{n}  acc={correct/total:.4f}")

    acc = correct / max(1, total)
    return {
        "pairwise_accuracy": acc,
        "chosen_logp_mean": sum(chosen_lps) / max(1, len(chosen_lps)),
        "rejected_logp_mean": sum(rejected_lps) / max(1, len(rejected_lps)),
        "eval_n": total,
    }


@torch.no_grad()
def generate_samples(
    model: torch.nn.Module,
    tok: PreTrainedTokenizerBase,
    prompts: List[str],
    device: torch.device,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> List[Dict[str, str]]:
    """Generate completions for a fixed prompt set."""
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tok.pad_token_id,
        )
        generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "generation": generated})
    return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Plain LoRA fine-tuning (DPO or SFT) — non-DP baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / data
    ap.add_argument("--model_id", required=True, help="HF model name or local path.")
    ap.add_argument("--dataset_path", required=True, help="JSONL file or HF dataset name.")
    ap.add_argument("--output_dir", required=True, help="Directory for all outputs.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_fraction", type=float, default=0.1, help="Fraction held out for eval.")

    # Training schedule
    ap.add_argument("--max_steps", type=int, default=-1, help="-1 uses num_train_epochs.")
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--max_seq_length", type=int, default=512)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated LoRA target modules (use c_attn,c_proj,c_fc for GPT-2).",
    )

    # DPO specific
    ap.add_argument("--dpo_beta", type=float, default=0.1, help="DPO beta parameter.")
    ap.add_argument(
        "--dpo_loss_type", type=str, default="sigmoid", help="DPO loss type."
    )

    # Precision / quantisation
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # Logging / saving
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_every_steps", type=int, default=0, help="0 = eval only at end.")
    ap.add_argument("--save_every_steps", type=int, default=0, help="0 = save only at end.")

    # Eval
    ap.add_argument(
        "--eval_prompts_json",
        type=str,
        default=None,
        help="JSON list of prompts for generation eval (uses built-in defaults if not set).",
    )
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument("--gen_max_new_tokens", type=int, default=128)

    # Config file shortcut
    ap.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON config file; CLI args override file values.",
    )

    args = ap.parse_args()

    # Merge config file if provided
    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if not hasattr(args, k):
                ap.error(f"Unknown config key: {k}")
            # CLI default detection: only override if user didn't pass it on CLI
            # Simple heuristic: if the value equals the parser default, use config
            default = ap.get_default(k)
            if getattr(args, k) == default:
                setattr(args, k, v)

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Record full config
    # -----------------------------------------------------------------------
    config = vars(args).copy()
    config["git_hash"] = get_git_hash()
    config["timestamp"] = datetime.datetime.now().isoformat()
    config["torch_version"] = torch.__version__
    config["cuda_available"] = torch.cuda.is_available()
    write_json(os.path.join(args.output_dir, "config.json"), config)
    print("=" * 60)
    print("Plain LoRA Training")
    print("=" * 60)
    for k, v in sorted(config.items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 2. Load dataset
    # -----------------------------------------------------------------------
    if os.path.isfile(args.dataset_path):
        raw = load_dataset("json", data_files=args.dataset_path)["train"]
    else:
        raw = load_dataset(args.dataset_path, split="train")

    cols = raw.column_names
    ds_type = detect_dataset_type(cols)
    print(f"[Data] detected_type={ds_type} columns={cols} N={len(raw)}")

    prompt_key = find_key(cols, PROMPT_CANDIDATES)
    if not prompt_key:
        raise ValueError(f"No prompt column found in {cols}")

    if ds_type == "dpo":
        chosen_key = find_key(cols, CHOSEN_CANDIDATES)
        rejected_key = find_key(cols, REJECTED_CANDIDATES)
        if not chosen_key or not rejected_key:
            raise ValueError(f"DPO dataset missing chosen/rejected in {cols}")
    else:
        response_key = find_key(cols, RESPONSE_CANDIDATES + CHOSEN_CANDIDATES)
        if not response_key:
            raise ValueError(f"SFT dataset missing response column in {cols}")

    # -----------------------------------------------------------------------
    # 3. Train / val split
    # -----------------------------------------------------------------------
    train_idx, val_idx = stable_split_indices(len(raw), args.val_fraction, args.seed)
    write_json(
        os.path.join(args.output_dir, "split_indices.json"),
        {"train": train_idx, "val": val_idx, "seed": args.seed, "val_fraction": args.val_fraction},
    )
    train_ds = raw.select(train_idx)
    val_ds = raw.select(val_idx)
    print(f"[Data] train={len(train_ds)} val={len(val_ds)}")

    # Normalise column names for DPO
    if ds_type == "dpo":
        rename = {}
        if prompt_key != "prompt":
            rename[prompt_key] = "prompt"
        if chosen_key != "chosen":
            rename[chosen_key] = "chosen"
        if rejected_key != "rejected":
            rename[rejected_key] = "rejected"
        if rename:
            train_ds = train_ds.rename_columns(rename)
            val_ds = val_ds.rename_columns(rename)
            prompt_key, chosen_key, rejected_key = "prompt", "chosen", "rejected"

        keep = ["prompt", "chosen", "rejected"]
        drop = [c for c in train_ds.column_names if c not in keep]
        if drop:
            train_ds = train_ds.remove_columns(drop)
            val_ds = val_ds.remove_columns(drop)

    # -----------------------------------------------------------------------
    # 4. Tokenizer
    # -----------------------------------------------------------------------
    # OpenLLaMA model card recommends slow tokenizer to avoid tokenization mismatch.
    use_fast = False if "open_llama" in args.model_id.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # -----------------------------------------------------------------------
    # 5. Model
    # -----------------------------------------------------------------------
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")

    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("Use only one of --load_in_4bit or --load_in_8bit.")

    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "device_map": None,
    }
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16 or cuda_ok:
        model_kwargs["torch_dtype"] = torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    print("[Model] loading...")
    model = load_causal_lm_compat(args.model_id, **model_kwargs)
    model.config.use_cache = False

    # -----------------------------------------------------------------------
    # 6. LoRA
    # -----------------------------------------------------------------------
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--target_modules must include at least one module.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    if ds_type == "sft":
        model = get_peft_model(model, lora_config)
        if args.gradient_checkpointing:
            # Required for LoRA + gradient checkpointing so loss keeps grad_fn.
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # 7. Training
    # -----------------------------------------------------------------------
    metrics_logger = MetricsLogger(args.output_dir)
    start_time = time.time()

    save_strategy = "steps" if args.save_every_steps > 0 else "no"
    save_steps = args.save_every_steps if args.save_every_steps > 0 else 500

    if ds_type == "dpo":
        # ---- DPO contrastive loss ----
        training_args = DPOConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            max_grad_norm=args.max_grad_norm,
            fp16=bool(cuda_ok and args.fp16),
            bf16=bool(cuda_ok and args.bf16),
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            seed=args.seed,
            beta=args.dpo_beta,
            loss_type=args.dpo_loss_type,
            max_length=args.max_seq_length,
            max_prompt_length=args.max_seq_length // 2,
            max_completion_length=args.max_seq_length // 2,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            processing_class=tok,
            args=training_args,
            train_dataset=train_ds,
            peft_config=lora_config,
            callbacks=[metrics_logger],
        )

    else:
        # ---- SFT on instruction data (prompt + response) ----
        from transformers import DataCollatorForLanguageModeling, Trainer

        def to_text(example):
            return {"text": f"{example[prompt_key]}\n\n{example[response_key]}"}

        train_ds = train_ds.map(to_text, remove_columns=train_ds.column_names)

        def tokenize(batch):
            return tok(batch["text"], truncation=True, max_length=args.max_seq_length)

        train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])

        model.to(device)
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            lr_scheduler_type=args.lr_scheduler_type,
            max_steps=args.max_steps,
            num_train_epochs=args.num_train_epochs,
            max_grad_norm=args.max_grad_norm,
            fp16=bool(cuda_ok and args.fp16),
            bf16=bool(cuda_ok and args.bf16),
            gradient_checkpointing=args.gradient_checkpointing,
            logging_steps=args.logging_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            report_to="none",
            remove_unused_columns=False,
            dataloader_num_workers=0,
            seed=args.seed,
        )

        data_collator = DataCollatorForLanguageModeling(tok, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
            callbacks=[metrics_logger],
        )

    print(f"[Train] type={ds_type} starting...")
    trainer.train()
    train_time = time.time() - start_time
    print(f"[Train] done in {train_time:.1f}s")

    # -----------------------------------------------------------------------
    # 8. Save adapter
    # -----------------------------------------------------------------------
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[Save] adapter -> {args.output_dir}")

    # -----------------------------------------------------------------------
    # 9. Evaluation
    # -----------------------------------------------------------------------
    # Make sure model is on GPU for eval
    trained_model = trainer.model
    device = next(trained_model.parameters()).device
    summary: Dict[str, Any] = {
        "train_time_seconds": train_time,
        "dataset_type": ds_type,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
    }

    # Pairwise accuracy (DPO only)
    if ds_type == "dpo" and len(val_ds) > 0:
        print("[Eval] pairwise accuracy on val split...")
        pair_metrics = eval_pairwise_accuracy(
            model=trained_model,
            tok=tok,
            ds=val_ds,
            prompt_key="prompt",
            chosen_key="chosen",
            rejected_key="rejected",
            device=device,
            max_len=args.max_seq_length,
            max_eval=args.max_eval_samples,
        )
        summary.update(pair_metrics)
        print(f"  pairwise_accuracy={pair_metrics['pairwise_accuracy']:.4f}")
        print(f"  chosen_logp_mean={pair_metrics['chosen_logp_mean']:.4f}")
        print(f"  rejected_logp_mean={pair_metrics['rejected_logp_mean']:.4f}")

    # Generation eval
    if args.eval_prompts_json:
        with open(args.eval_prompts_json, "r") as f:
            eval_prompts = json.load(f)
    else:
        eval_prompts = DEFAULT_EVAL_PROMPTS

    print(f"[Eval] generating on {len(eval_prompts)} prompts...")
    generations = generate_samples(
        model=trained_model,
        tok=tok,
        prompts=eval_prompts,
        device=device,
        max_new_tokens=args.gen_max_new_tokens,
    )
    gen_path = os.path.join(args.output_dir, "generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for g in generations:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"[Eval] generations -> {gen_path}")

    # Training loss from logs
    log_history = trainer.state.log_history
    train_losses = [x["loss"] for x in log_history if "loss" in x]
    if train_losses:
        summary["train_loss_final"] = train_losses[-1]
        summary["train_loss_first"] = train_losses[0]

    summary["seed"] = args.seed
    summary["model_id"] = args.model_id
    summary["lora_r"] = args.lora_r
    summary["lora_alpha"] = args.lora_alpha
    summary["learning_rate"] = args.learning_rate
    summary["max_steps"] = args.max_steps
    summary["num_train_epochs"] = args.num_train_epochs

    write_json(os.path.join(args.output_dir, "summary.json"), summary)
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
