#!/usr/bin/env python3
"""
Job 4 — Oracle SB, Gated Corrected Reference.

Identical to Job 3 except the interpolation weight is:

    v_i = w_i * sigmoid(|delta_m1| / tau) * 1[delta_m1 > 0]

This gates M1's contribution on JOINT trust:
  (a) label trust (w_i from SB posterior)
  (b) M1 confidence on THIS pair: sigmoid(|delta_m1| / tau)
  (c) M1 direction correct: delta_m1 > 0 (M1 agrees with oracle label)

When any factor is zero (e.g., M1 disagrees with label), the reference
collapses to base, exactly like Job 2.  When all three are high, the
reference approaches M1, like intended in Job 3.

Expected outcome: accuracy > Job 2 (66.6%) if gating successfully filters
out the 44% of pairs where M1 has wrong sign.

Usage:
  CUDA_VISIBLE_DEVICES=0 python train_job4_oracle_sb_gatedref.py \\
    --model   Qwen/Qwen2.5-3B-Instruct \\
    --m1      ../../stage1/results_instruct_eps1.0_seed42/pku_eps1.0_s42 \\
    --data    d2_pku_with_flipped_eps1.0_seed42.jsonl \\
    --out     models/pku_oracle_sb_gatedref \\
    --epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --tau 1.0
"""
from __future__ import annotations

import argparse, gc, json, math, os, shutil, sys
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Utilities ─────────────────────────────────────────────────────────────────

def compute_oracle_weights(
    was_flipped_list: List[bool],
    epsilon: float,
    oracle_delta: float = 10.0,
) -> List[float]:
    gamma_eps = 1.0 / (1.0 + math.exp(epsilon))
    p_keep    = 1.0 - gamma_eps
    weights   = []
    for flipped in was_flipped_list:
        delta = -oracle_delta if flipped else +oracle_delta
        q_i   = 1.0 / (1.0 + math.exp(-delta))
        num   = p_keep * q_i
        den   = p_keep * q_i + gamma_eps * (1.0 - q_i)
        w_i   = max(0.01, min(0.99, num / den if den > 1e-12 else 0.5))
        weights.append(w_i)
    return weights


def compute_response_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    out = model(input_ids=input_ids)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    tok_lp       = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    B, L         = tok_lp.shape
    pos  = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    mask = (pos >= (prompt_lens.unsqueeze(1) - 1)) & (shift_labels != pad_id)
    return (tok_lp * mask.float()).sum(dim=1)


# ── Gated reference ───────────────────────────────────────────────────────────

def compute_joint_weight(
    w: torch.Tensor,        # label trust, shape [B], ∈ [0.01, 0.99]
    delta_m1: torch.Tensor, # M1 log-ratio (chosen - rejected), shape [B]
    tau: float,             # temperature for confidence: higher → less aggressive gating
    threshold: float = 0.0, # direction threshold (0 = strict agreement with label)
) -> torch.Tensor:
    """
    v_i = w_i * sigmoid(|delta_m1| / tau) * 1[delta_m1 > threshold]

    Components:
      w_i               : label trust from SB posterior
      sigmoid(|δ|/τ)    : how strong/confident is M1's signal on this pair
      1[δ > threshold]  : M1's direction agrees with noisy label

    When M1 disagrees (δ ≤ 0) → v_i = 0 → ref = base_log_ratio.
    When M1 is confident and agrees → v_i ≈ w_i → ref ≈ M1_log_ratio.
    """
    m1_confidence        = torch.sigmoid(delta_m1.abs() / tau)
    m1_direction_correct = (delta_m1 > threshold).float()
    return w * m1_confidence * m1_direction_correct


def gated_ref_log_ratio(
    m1_c: torch.Tensor,
    m1_r: torch.Tensor,
    base_c: torch.Tensor,
    base_r: torch.Tensor,
    v: torch.Tensor,          # gated joint weight, shape [B]
) -> torch.Tensor:
    """
    ref = v_i * (log π_M1(c) - log π_M1(r)) + (1-v_i) * (log π_base(c) - log π_base(r))

    v=1 → trust M1 fully; v=0 → fall back to base.
    v_i ≤ w_i always (gating is conservative).
    """
    m1_ratio   = m1_c   - m1_r
    base_ratio = base_c - base_r
    return v * m1_ratio + (1.0 - v) * base_ratio


def sb_gated_ref_loss(
    pi_c:   torch.Tensor,
    pi_r:   torch.Tensor,
    m1_c:   torch.Tensor,
    m1_r:   torch.Tensor,
    base_c: torch.Tensor,
    base_r: torch.Tensor,
    w:      torch.Tensor,    # SB label-trust weight
    beta:   float,
    tau:    float,
    step:   int,
    log_every: int = 50,
) -> torch.Tensor:
    delta_m1  = m1_c - m1_r
    v         = compute_joint_weight(w, delta_m1, tau)
    ref_ratio = gated_ref_log_ratio(m1_c, m1_r, base_c, base_r, v)
    margin    = beta * ((pi_c - pi_r) - ref_ratio)
    loss      = -(w * F.logsigmoid(margin) + (1.0 - w) * F.logsigmoid(-margin))

    if step % log_every == 0:
        base_ratio_mean = (base_c - base_r).mean().item()
        m1_ratio_mean   = (m1_c   - m1_r).mean().item()
        print(
            f"    [gated step={step}] "
            f"v_mean={v.mean():.3f}  w_mean={w.mean():.3f}  "
            f"delta_m1_mean={delta_m1.mean():.4f}  "
            f"ref_mean={ref_ratio.mean():.4f}  "
            f"(base={base_ratio_mean:.4f}  m1={m1_ratio_mean:.4f})  "
            f"margin_mean={margin.mean():.4f}",
            flush=True,
        )
        frac_gated_off = (v < 0.01).float().mean().item()
        print(f"    [gated] fraction using base only (v<0.01): {frac_gated_off:.3f}  "
              f"fraction using M1 (v>0.5): {(v > 0.5).float().mean():.3f}", flush=True)

    return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairDataset(TorchDataset):
    def __init__(self, prompts, chosen, rejected, weights, tok, max_len):
        self.prompts  = prompts
        self.chosen   = chosen
        self.rejected = rejected
        self.weights  = weights
        self.tok      = tok
        self.max_len  = max_len

    def __len__(self): return len(self.prompts)

    def __getitem__(self, idx):
        p_ids = self.tok.encode(self.prompts[idx],  add_special_tokens=True)
        c_ids = self.tok.encode(self.chosen[idx],   add_special_tokens=False)
        r_ids = self.tok.encode(self.rejected[idx], add_special_tokens=False)
        return {
            "chosen_ids":   (p_ids + c_ids)[: self.max_len],
            "rejected_ids": (p_ids + r_ids)[: self.max_len],
            "prompt_len":   len(p_ids),
            "w":            self.weights[idx],
        }


def collate_fn(batch, pad_id):
    def pad(seqs):
        ml = max(len(s) for s in seqs)
        return torch.tensor([s + [pad_id] * (ml - len(s)) for s in seqs], dtype=torch.long)
    return {
        "chosen_ids":   pad([b["chosen_ids"]   for b in batch]),
        "rejected_ids": pad([b["rejected_ids"] for b in batch]),
        "prompt_lens":  torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "w":            torch.tensor([b["w"]          for b in batch], dtype=torch.float32),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True, help="Base model ID or path")
    p.add_argument("--m1",      required=True, help="Stage1 LoRA adapter path (M1)")
    p.add_argument("--data",    required=True, help="D2 JSONL with was_flipped")
    p.add_argument("--out",     required=True)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--beta",    type=float, default=0.5)
    p.add_argument("--tau",     type=float, default=1.0,
                   help="Temperature for M1 confidence gating. "
                        "Higher tau → softer gating (less aggressive filtering).")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--lr",      type=float, default=2.5e-5)
    p.add_argument("--epochs",  type=int,   default=3)
    p.add_argument("--bsz",     type=int,   default=4)
    p.add_argument("--ga",      type=int,   default=4)
    p.add_argument("--max_len", type=int,   default=512)
    p.add_argument("--max_grad_norm",    type=float, default=1.0)
    p.add_argument("--save_steps",       type=int,   default=100)
    p.add_argument("--save_total_limit", type=int,   default=2)
    p.add_argument("--max_steps",        type=int,   default=-1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--bf16",    action="store_true")
    p.add_argument("--oracle_delta", type=float, default=10.0)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"=== Job 4: Oracle SB — Gated Corrected Reference ===")
    print(f"model={args.model}  m1={args.m1}")
    print(f"data={args.data}  out={args.out}")
    print(f"epsilon={args.epsilon}  beta={args.beta}  tau={args.tau}  lr={args.lr}")
    print(f"Key: v_i = w_i * sigmoid(|delta_m1| / tau) * 1[delta_m1 > 0]")

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if "was_flipped" not in rows[0]:
        raise ValueError("Data missing 'was_flipped' — run gen_oracle_labels.py first")

    prompts     = [r["prompt"]      for r in rows]
    chosen      = [r["chosen"]      for r in rows]
    rejected    = [r["rejected"]    for r in rows]
    was_flipped = [r["was_flipped"] for r in rows]
    N = len(rows)
    print(f"[Data] N={N}")

    weights = compute_oracle_weights(was_flipped, args.epsilon, args.oracle_delta)
    w_t = torch.tensor(weights)
    n_flip = sum(was_flipped)
    print(f"\n=== Weight Sanity ===")
    print(f"  w min={w_t.min():.3f}  max={w_t.max():.3f}  mean={w_t.mean():.3f}")
    print(f"  w > 0.9: {(w_t > 0.9).sum()}/{N}  (expect ~73%)")
    print(f"  w < 0.1: {(w_t < 0.1).sum()}/{N}  (expect ~27%)")
    print(f"  Actual flipped: {n_flip}/{N} = {n_flip/N:.3f}")

    cuda_ok = torch.cuda.is_available()
    device  = torch.device("cuda" if cuda_ok else "cpu")
    dtype   = torch.bfloat16 if (cuda_ok and args.bf16) else (torch.float16 if cuda_ok else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    # ── Load M1 (frozen) ───────────────────────────────────────────────────
    print(f"\n[M1] Loading {args.model} + stage1 adapter {args.m1} (frozen)...")
    base_for_m1 = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    base_for_m1.config.use_cache = True
    m1 = PeftModel.from_pretrained(base_for_m1, args.m1, adapter_name="stage1")
    m1.set_adapter("stage1")
    m1 = m1.to(device).eval()
    for p in m1.parameters():
        p.requires_grad = False
    print("[M1] Loaded and frozen.")

    # ── Load base (frozen) ─────────────────────────────────────────────────
    print(f"\n[Base] Loading {args.model} (frozen)...")
    base_ref = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    base_ref.config.use_cache = True
    base_ref = base_ref.to(device).eval()
    for p in base_ref.parameters():
        p.requires_grad = False
    print("[Base] Loaded and frozen.")

    # ── Dataset & DataLoader ───────────────────────────────────────────────
    ds = PairDataset(prompts, chosen, rejected, weights, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, pad_id))

    # ── Load M2 = fresh LoRA on base ───────────────────────────────────────
    print("\n[M2] Loading fresh base model + LoRA (trainable)...")
    m2_base = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    m2_base.config.use_cache = False
    m2_base.gradient_checkpointing_enable()
    
    if args.resume_from:
        print(f"Resuming from checkpoint {args.resume_from}")
        m2 = PeftModel.from_pretrained(m2_base, args.resume_from, adapter_name="stage2", is_trainable=True)
        m2.set_adapter("stage2")
    else:
        lora_cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        m2 = get_peft_model(m2_base, lora_cfg, adapter_name="stage2")
    m2 = m2.to(device).train()
    trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"[M2] trainable={trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in m2.parameters() if p.requires_grad], lr=args.lr, eps=1e-6)

    # ── Gating distribution preview (approximate, first batch of M1 scores) ─
    # Pre-score a small sample to show expected gating stats before training
    print("\n[Gating Preview] Scoring first 100 pairs with M1 to preview v_i distribution...")
    sample_deltas = []
    with torch.no_grad():
        for i in range(min(100, N)):
            p_ids = tok.encode(prompts[i], add_special_tokens=True)
            c_ids = tok.encode(chosen[i],  add_special_tokens=False)
            r_ids = tok.encode(rejected[i], add_special_tokens=False)
            c_full = torch.tensor([(p_ids + c_ids)[:args.max_len]], device=device)
            r_full = torch.tensor([(p_ids + r_ids)[:args.max_len]], device=device)
            pl     = torch.tensor([len(p_ids)], device=device)
            lp_c   = compute_response_logprobs(m1, c_full, pl, pad_id).item()
            lp_r   = compute_response_logprobs(m1, r_full, pl, pad_id).item()
            sample_deltas.append(lp_c - lp_r)
    sample_delta_t = torch.tensor(sample_deltas)
    sample_w       = torch.tensor(weights[:len(sample_deltas)])
    sample_v       = compute_joint_weight(sample_w, sample_delta_t, args.tau)
    print(f"  M1 delta (first 100): mean={sample_delta_t.mean():.4f}  "
          f"frac>0={( sample_delta_t>0).float().mean():.3f}")
    print(f"  v_i (first 100):      mean={sample_v.mean():.4f}  "
          f"frac<0.01={(sample_v<0.01).float().mean():.3f}  "
          f"frac>0.5={(sample_v>0.5).float().mean():.3f}")
    print(f"  Expected: ~44% of pairs gated off (v~0) due to M1 wrong sign")

    # ── Training loop ──────────────────────────────────────────────────────
    global_step = 0
    batches_to_skip = 0
    if args.resume_from:
        import re
        m = re.search(r"step-(\d+)", args.resume_from)
        if m:
            global_step = int(m.group(1))
            batches_to_skip = global_step * args.ga
    
    accum_loss  = 0.0
    max_steps   = args.max_steps if args.max_steps > 0 else None
    ckpt_dirs: List[str] = []

    def save_ckpt(step):
        d = os.path.join(args.out, f"checkpoint-step-{step}")
        m2.save_pretrained(d, adapter_name="stage2")
        ckpt_dirs.append(d)
        while len(ckpt_dirs) > max(1, args.save_total_limit):
            shutil.rmtree(ckpt_dirs.pop(0), ignore_errors=True)
        print(f"[Checkpoint] saved {d}", flush=True)

    for epoch in range(args.epochs):
        for bidx, batch in enumerate(dl):
            if batches_to_skip > 0:
                batches_to_skip -= 1
                continue

            c_ids  = batch["chosen_ids"].to(device)
            r_ids  = batch["rejected_ids"].to(device)
            p_lens = batch["prompt_lens"].to(device)
            w      = batch["w"].to(device)

            with torch.no_grad():
                m1_c   = compute_response_logprobs(m1,       c_ids, p_lens, pad_id)
                m1_r   = compute_response_logprobs(m1,       r_ids, p_lens, pad_id)
                base_c = compute_response_logprobs(base_ref, c_ids, p_lens, pad_id)
                base_r = compute_response_logprobs(base_ref, r_ids, p_lens, pad_id)

            pi_c = compute_response_logprobs(m2, c_ids, p_lens, pad_id)
            pi_r = compute_response_logprobs(m2, r_ids, p_lens, pad_id)

            loss = sb_gated_ref_loss(
                pi_c, pi_r, m1_c, m1_r, base_c, base_r, w,
                beta=args.beta, tau=args.tau,
                step=global_step,
            ) / args.ga
            loss.backward()
            accum_loss += loss.item()

            if (bidx + 1) % args.ga == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in m2.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step(); optimizer.zero_grad()
                global_step += 1
                if global_step <= 3 or global_step % 10 == 0:
                    print(f"  [step={global_step}] loss={accum_loss:.6f}  epoch={epoch}", flush=True)
                accum_loss = 0.0
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_ckpt(global_step)
                if max_steps and global_step >= max_steps:
                    break
        if max_steps and global_step >= max_steps:
            break
        print(f"[Epoch {epoch+1}/{args.epochs}] done")

    if accum_loss != 0.0:
        optimizer.step(); optimizer.zero_grad()

    m2.save_pretrained(args.out, adapter_name="stage2")
    manifest = {
        "base_model": args.model,
        "adapters": [{"name": "stage2", "path": os.path.abspath(args.out)}],
        "method": "oracle_sb_gated_ref_fresh",
        "params": {
            "epsilon": args.epsilon, "beta": args.beta, "tau": args.tau, "lr": args.lr,
            "m1_adapter": args.m1, "oracle_delta": args.oracle_delta,
            "ref_formula": "v_i * M1_log_ratio + (1-v_i) * base_log_ratio",
            "gate_formula": "v_i = w_i * sigmoid(|delta_m1| / tau) * 1[delta_m1 > 0]",
        },
        "note": (
            "Fresh LoRA. Oracle SB weights. Gated M1/base reference per sample. "
            "Gate collapses to base when M1 has wrong sign or low confidence. "
            "Eval uses base as reference (single-adapter mode)."
        ),
    }
    with open(os.path.join(args.out, "M2_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Job 4 complete → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
