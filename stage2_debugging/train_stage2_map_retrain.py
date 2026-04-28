#!/usr/bin/env python3
"""
Option C: Single LoRA Retrain on MAP-relabeled D2.

Instead of stacking a second LoRA adapter on top of frozen Stage1,
train a FRESH single LoRA from the base model on MAP-cleaned D2 data.

Pipeline:
  1. Score D2 pairs with M1 (base+stage1) → binary model predictions
  2. MAP estimator → hard relabel
  3. Train a FRESH LoRA from base model (no stacking) with DPOTrainer
     Reference = bare base model, Policy = base + fresh LoRA

This avoids the "correction on top of correction" LoRA stacking problem.

Usage:
  CUDA_VISIBLE_DEVICES=0 python -u lora/train_stage2_map_retrain.py \
    --model Qwen/Qwen2.5-3B \
    --stage1_adapter outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed42 \
    --data outputs_new_models/preprocessing/d2_rr_flipped_eps1.0_seed42.jsonl \
    --out outputs_overnight/map_retrain_qwen_s42 \
    --epsilon 1.0 --seed 42
"""
import argparse, json, math, os, sys, gc
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Base model ID")
    p.add_argument("--stage1_adapter", required=True, help="Path to Stage1 LoRA (for M1 scoring only)")
    p.add_argument("--data", required=True, help="D2 noisy preference JSONL")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--epsilon", type=float, default=1.0)
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
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    return p.parse_args()


@torch.no_grad()
def score_pair(model, tok, prompt, resp, device, max_len=512):
    """Sum of token log-probs over response only."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(resp, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[-max_len:]
    if len(ids) < 2:
        return 0.0
    p_len = min(len(p_ids), len(ids) - 1)
    x = torch.tensor([ids], device=device)
    logits = model(x).logits
    logp = F.log_softmax(logits[:, :-1, :], dim=-1)
    tgt = x[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    return float(tok_lp[max(p_len - 1, 0):].sum())


@torch.no_grad()
def dpo_reward(m1, tok, prompt, resp, device, max_len=512):
    """DPO implicit reward: log π_M1(y|x) - log π_base(y|x).

    Uses PeftModel.disable_adapter() to score with base model,
    avoiding a second model copy. Cancels length bias.
    """
    m1.set_adapter("stage1")
    s_m1 = score_pair(m1, tok, prompt, resp, device, max_len)
    with m1.disable_adapter():
        s_base = score_pair(m1, tok, prompt, resp, device, max_len)
    return s_m1 - s_base


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_keep = math.exp(args.epsilon) / (math.exp(args.epsilon) + 1)
    noise_flip = 1 - p_keep

    print(f"=== Option C: Single LoRA Retrain on MAP-relabeled D2 ===")
    print(f"p_keep={p_keep:.4f}  noise_flip={noise_flip:.4f}")

    # ─── Load D2 ───
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    N = len(rows)

    # ─── Load M1 for scoring ───
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    cache_path = os.path.join(args.out, "MAP_scorer_cache.pt")
    if os.path.exists(cache_path):
        print(f"[Cache] Found {cache_path}, skipping M1 scoring...", flush=True)
        model_labels = torch.load(cache_path, map_location="cpu", weights_only=False)["model_labels"]
        agree_rate = sum(model_labels) / len(model_labels)
        disagree_rate = 1 - agree_rate
        if abs(1 - 2 * noise_flip) > 1e-8:
            model_flip = max(0.01, min(0.49, (disagree_rate - noise_flip) / (1 - 2 * noise_flip)))
        else:
            model_flip = 0.5
        print(f"  agree={agree_rate:.3f}  model_flip={model_flip:.3f}")
    else:
        print("Loading M1 (base + stage1) for scoring...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
        )
        base.config.use_cache = False
        m1 = PeftModel.from_pretrained(base, args.stage1_adapter, adapter_name="stage1")
        m1.set_adapter("stage1")
        m1 = m1.to(device).eval()
        for p in m1.parameters():
            p.requires_grad = False

        # ─── Step 1: Score D2 with M1 using DPO implicit reward ───
        # DPO reward = log π_M1(y|x) - log π_base(y|x)
        # This isolates the LoRA's preference signal and cancels length bias.
        print(f"\n[Step 1/3] Scoring D2 with M1 (DPO implicit reward)...", flush=True)
        model_labels = []
        for i, r in enumerate(rows):
            dr_c = dpo_reward(m1, tok, r["prompt"], r["chosen"], device, args.max_len)
            dr_r = dpo_reward(m1, tok, r["prompt"], r["rejected"], device, args.max_len)
            model_labels.append(1 if dr_c > dr_r else 0)
            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{N}] agree={sum(model_labels)/len(model_labels):.3f}", flush=True)

        agree_rate = sum(model_labels) / len(model_labels)
        disagree_rate = 1 - agree_rate
        if abs(1 - 2 * noise_flip) > 1e-8:
            model_flip = max(0.01, min(0.49, (disagree_rate - noise_flip) / (1 - 2 * noise_flip)))
        else:
            model_flip = 0.5
        print(f"  agree={agree_rate:.3f}  model_flip={model_flip:.3f}")

        torch.save({"model_labels": model_labels}, cache_path)
        print(f"[Cache] Saved scoring data to {cache_path}")

        # Free M1
        del m1, base; gc.collect(); torch.cuda.empty_cache()

    # ─── Step 2: MAP relabel ───
    print(f"\n[Step 2/3] MAP relabeling...", flush=True)
    log_rr = math.log((1 - noise_flip) / noise_flip) if 0 < noise_flip < 1 else 0
    log_model = math.log((1 - model_flip) / model_flip) if 0 < model_flip < 1 else 0
    relabeled = []
    n_flipped = 0
    for i, r in enumerate(rows):
        decision = (1 - 2 * 1) * log_rr + (1 - 2 * model_labels[i]) * log_model
        if decision > 0:
            # Swap chosen/rejected
            relabeled.append({"prompt": r["prompt"], "chosen": r["rejected"], "rejected": r["chosen"]})
            n_flipped += 1
        else:
            relabeled.append({"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]})

    print(f"  MAP flipped {n_flipped}/{N} pairs ({100*n_flipped/N:.1f}%)")
    ds = Dataset.from_list(relabeled)

    # Save stats
    stats = {"method": "map_retrain", "n_pairs": N, "p_keep": p_keep,
             "m1_agree_rate": agree_rate, "model_flip": model_flip,
             "n_map_flipped": n_flipped}
    with open(os.path.join(args.out, "map_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ─── Step 3: Train FRESH LoRA from base (no stacking) ───
    print(f"\n[Step 3/3] Training fresh LoRA from base model...", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16 if args.bf16 else torch.float32
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    training_args = DPOConfig(
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
        ref_model=None,  # PEFT implicit reference = base model (no stage1!)
        train_dataset=ds,
        processing_class=tok,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(os.path.join(args.out, "fresh_lora"))
    print(f"\n✅ Option C (MAP retrain) complete. Saved to {args.out}")

    # Manifest — note: this model has a SINGLE adapter from base (no stage1!)
    manifest = {
        "base_model": args.model,
        "method": "map_retrain_single_lora",
        "adapters": [
            {"name": "fresh_lora", "path": os.path.abspath(os.path.join(args.out, "fresh_lora"))},
        ],
        "note": "Single LoRA from base model. Eval uses base as reference (not stage1)."
    }
    with open(os.path.join(args.out, "M2_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
