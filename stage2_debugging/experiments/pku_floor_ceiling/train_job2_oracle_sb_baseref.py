#!/usr/bin/env python3
"""
Job 2 — Oracle Soft-Bayes, Base Reference (Ceiling).

Fresh LoRA on base model. Oracle weights from was_flipped field.
Base model is reference throughout. No M1 stacking.

Usage:
  CUDA_VISIBLE_DEVICES=1 python train_job2_oracle_sb_baseref.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --data  d2_pku_with_flipped_eps1.0_seed42.jsonl \
    --out   models/pku_oracle_sb_baseref \
    --epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3
"""
from __future__ import annotations

import argparse, gc, json, math, os, shutil, sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

EPSILON = 1.0


# ── Oracle weight computation ────────────────────────────────────────────────

def compute_oracle_weights(
    was_flipped_list: List[bool],
    epsilon: float,
    oracle_delta: float = 10.0,
) -> List[float]:
    """
    Compute SB posterior weights from ground-truth flip indicators.
      was_flipped=False → delta = +oracle_delta → q ≈ 1   → w ≈ p_keep * 1 / (p_keep * 1 + (1-p_keep) * 0) ≈ 1
      was_flipped=True  → delta = -oracle_delta → q ≈ 0   → w ≈ 0

    w_i = p_keep * q_i / (p_keep * q_i + (1 - p_keep) * (1 - q_i))
    """
    gamma_eps = 1.0 / (1.0 + math.exp(epsilon))  # = 0.269 for eps=1.0
    p_keep    = 1.0 - gamma_eps                   # = 0.731

    weights = []
    for flipped in was_flipped_list:
        delta = -oracle_delta if flipped else +oracle_delta
        q_i   = 1.0 / (1.0 + math.exp(-delta))   # sigmoid(±10) ≈ 0 or 1
        num   = p_keep * q_i
        den   = p_keep * q_i + gamma_eps * (1.0 - q_i)
        w_i   = num / den if den > 1e-12 else 0.5
        w_i   = max(0.01, min(0.99, w_i))
        weights.append(w_i)
    return weights


def sanity_check_weights(weights: List[float], was_flipped: List[bool]) -> None:
    w = torch.tensor(weights)
    n = len(w)
    print(f"\n=== Weight Sanity Check ===")
    print(f"  w min={w.min():.3f}  max={w.max():.3f}  mean={w.mean():.3f}  std={w.std():.3f}")
    print(f"  w > 0.9: {(w > 0.9).sum()}/{n}  (expect ~73%, the clean pairs)")
    print(f"  w < 0.1: {(w < 0.1).sum()}/{n}  (expect ~27%, the flipped pairs)")
    n_flip = sum(was_flipped)
    print(f"  Actual flipped: {n_flip}/{n} = {n_flip/n:.3f}")


# ── Loss ─────────────────────────────────────────────────────────────────────

def soft_bayes_loss(pi_c, pi_r, ref_c, ref_r, w, beta):
    """
    r_c = log π(chosen) - log π_ref(chosen)
    r_r = log π(rejected) - log π_ref(rejected)
    m   = r_c - r_r
    L   = -[w * logsigmoid(β m) + (1-w) * logsigmoid(-β m)]
    """
    m = (pi_c - ref_c) - (pi_r - ref_r)
    loss = -(w * F.logsigmoid(beta * m) + (1.0 - w) * F.logsigmoid(-beta * m))
    return loss.mean()


# ── Sequence log-prob ────────────────────────────────────────────────────────

def compute_response_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Sum of log P(token | context) over response tokens. Shape: [B]."""
    out = model(input_ids=input_ids)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs    = F.log_softmax(shift_logits, dim=-1)
    tok_lp       = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    B, L         = tok_lp.shape
    pos  = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    mask = (pos >= (prompt_lens.unsqueeze(1) - 1)) & (shift_labels != pad_id)
    return (tok_lp * mask.float()).sum(dim=1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class PairDataset(TorchDataset):
    def __init__(self, prompts, chosen, rejected, weights, ref_c_lps, ref_r_lps, tok, max_len):
        self.prompts    = prompts
        self.chosen     = chosen
        self.rejected   = rejected
        self.weights    = weights
        self.ref_c_lps  = ref_c_lps
        self.ref_r_lps  = ref_r_lps
        self.tok        = tok
        self.max_len    = max_len

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        p_ids = self.tok.encode(self.prompts[idx],  add_special_tokens=True)
        c_ids = self.tok.encode(self.chosen[idx],   add_special_tokens=False)
        r_ids = self.tok.encode(self.rejected[idx], add_special_tokens=False)
        return {
            "chosen_ids":   (p_ids + c_ids)[: self.max_len],
            "rejected_ids": (p_ids + r_ids)[: self.max_len],
            "prompt_len":   len(p_ids),
            "w":            self.weights[idx],
            "ref_c_lp":     self.ref_c_lps[idx],
            "ref_r_lp":     self.ref_r_lps[idx],
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
        "ref_c_lp":     torch.tensor([b["ref_c_lp"]  for b in batch], dtype=torch.float32),
        "ref_r_lp":     torch.tensor([b["ref_r_lp"]  for b in batch], dtype=torch.float32),
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--data",    required=True, help="D2 JSONL with was_flipped field")
    p.add_argument("--out",     required=True)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--beta",    type=float, default=0.5)
    p.add_argument("--lr",      type=float, default=2.5e-5)
    p.add_argument("--epochs",  type=int,   default=3)
    p.add_argument("--bsz",     type=int,   default=4)
    p.add_argument("--ga",      type=int,   default=4)   # effective batch = 16
    p.add_argument("--max_len", type=int,   default=512)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_steps",    type=int,   default=100)
    p.add_argument("--save_total_limit", type=int, default=2)
    p.add_argument("--max_steps",     type=int,   default=-1)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--bf16",    action="store_true")
    p.add_argument("--oracle_delta", type=float, default=10.0)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    print(f"=== Job 2: Oracle SB — Base Reference ===")
    print(f"model={args.model}  data={args.data}  out={args.out}")
    print(f"epsilon={args.epsilon}  beta={args.beta}  lr={args.lr}  bsz={args.bsz}  ga={args.ga}")

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if "was_flipped" not in rows[0]:
        raise ValueError("Data missing 'was_flipped' field — run gen_oracle_labels.py first")

    prompts     = [r["prompt"]      for r in rows]
    chosen      = [r["chosen"]      for r in rows]
    rejected    = [r["rejected"]    for r in rows]
    was_flipped = [r["was_flipped"] for r in rows]
    N = len(rows)
    print(f"[Data] N={N}")

    # Oracle weights
    weights = compute_oracle_weights(was_flipped, args.epsilon, args.oracle_delta)
    sanity_check_weights(weights, was_flipped)

    # Device / dtype
    cuda_ok = torch.cuda.is_available()
    device  = torch.device("cuda" if cuda_ok else "cpu")
    dtype   = torch.bfloat16 if (cuda_ok and args.bf16) else (torch.float16 if cuda_ok else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    # ── Pre-compute base model reference logprobs ──────────────────────────
    cache_path = os.path.join(args.out, "base_ref_cache.pt")
    if os.path.exists(cache_path):
        print(f"[Cache] Loading base ref logprobs from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        ref_c_lps = cache["ref_c_lps"]
        ref_r_lps = cache["ref_r_lps"]
    else:
        print("[Base] Loading base model to pre-compute reference logprobs...")
        base = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
        base = base.to(device).eval()
        base.config.use_cache = True

        ref_c_lps, ref_r_lps = [], []
        base_c_ids = [(tok.encode(p, add_special_tokens=True) + tok.encode(c, add_special_tokens=False))[-args.max_len:]
                      for p, c in zip(prompts, chosen)]
        base_r_ids = [(tok.encode(p, add_special_tokens=True) + tok.encode(r, add_special_tokens=False))[-args.max_len:]
                      for p, r in zip(prompts, rejected)]
        p_lens = [min(len(tok.encode(p, add_special_tokens=True)), args.max_len) for p in prompts]

        with torch.no_grad():
            for i in range(N):
                for ids_list, lps_list in [(base_c_ids, ref_c_lps), (base_r_ids, ref_r_lps)]:
                    x   = torch.tensor([ids_list[i]], device=device)
                    pl  = torch.tensor([p_lens[i]],   device=device)
                    lp  = compute_response_logprobs(base, x, pl, pad_id).item()
                    lps_list.append(lp)
                if (i + 1) % 500 == 0:
                    print(f"  [ref {i+1}/{N}]", flush=True)

        torch.save({"ref_c_lps": ref_c_lps, "ref_r_lps": ref_r_lps}, cache_path)
        print(f"[Cache] Saved to {cache_path}")
        del base; gc.collect()
        if cuda_ok: torch.cuda.empty_cache()

    # ── Loss sanity (before loading M2) ───────────────────────────────────
    print("\n=== Loss Sanity Check (on first 50 pairs) ===")
    ref_c_t = torch.tensor(ref_c_lps[:50])
    ref_r_t = torch.tensor(ref_r_lps[:50])
    w_t     = torch.tensor(weights[:50])
    # Dummy margin = 0; SB loss should equal log(2) but differ from identical dummy MLE
    dummy_pi_c = ref_c_t + 0.5
    dummy_pi_r = ref_r_t - 0.5
    sb_l  = soft_bayes_loss(dummy_pi_c, dummy_pi_r, ref_c_t, ref_r_t, w_t, args.beta)
    mle_l = -F.logsigmoid(torch.tensor(args.beta) * (dummy_pi_c - ref_c_t - dummy_pi_r + ref_r_t)).mean()
    print(f"  SB loss (oracle weights): {sb_l:.4f}")
    print(f"  MLE loss (uniform w=1):   {mle_l:.4f}")
    if abs(sb_l.item() - mle_l.item()) < 1e-4:
        print("  ⚠️  WARNING: SB and MLE losses are identical — weights not connected!")
    else:
        print("  ✅ SB and MLE losses differ as expected.")

    # ── Build DataLoader ───────────────────────────────────────────────────
    ds = PairDataset(prompts, chosen, rejected, weights, ref_c_lps, ref_r_lps, tok, args.max_len)
    dl = DataLoader(ds, batch_size=args.bsz, shuffle=True,
                    collate_fn=lambda b: collate_fn(b, pad_id))

    # ── Load M2 = fresh LoRA on base ───────────────────────────────────────
    print("\n[M2] Loading fresh base model + LoRA...")
    m2 = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True)
    m2.config.use_cache = False
    m2.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    m2 = get_peft_model(m2, lora_cfg, adapter_name="stage2")
    m2 = m2.to(device).train()
    trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"[M2] trainable={trainable:,}")

    optimizer = torch.optim.AdamW(
        [p for p in m2.parameters() if p.requires_grad],
        lr=args.lr, eps=1e-6,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    global_step = 0
    accum_loss  = 0.0
    max_steps   = args.max_steps if args.max_steps > 0 else None
    ckpt_dirs: List[str] = []

    def save_ckpt(step):
        d = os.path.join(args.out, f"checkpoint-step-{step}")
        m2.save_pretrained(d, adapter_name="stage2")
        ckpt_dirs.append(d)
        while len(ckpt_dirs) > max(1, args.save_total_limit):
            old = ckpt_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
        print(f"[Checkpoint] saved {d}", flush=True)

    for epoch in range(args.epochs):
        for bidx, batch in enumerate(dl):
            c_ids  = batch["chosen_ids"].to(device)
            r_ids  = batch["rejected_ids"].to(device)
            p_lens = batch["prompt_lens"].to(device)
            w      = batch["w"].to(device)
            ref_c  = batch["ref_c_lp"].to(device)
            ref_r  = batch["ref_r_lp"].to(device)

            pi_c = compute_response_logprobs(m2, c_ids, p_lens, pad_id)
            pi_r = compute_response_logprobs(m2, r_ids, p_lens, pad_id)
            loss = soft_bayes_loss(pi_c, pi_r, ref_c, ref_r, w, args.beta) / args.ga
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

    # Flush remainder
    if accum_loss != 0.0:
        optimizer.step(); optimizer.zero_grad()

    m2.save_pretrained(args.out, adapter_name="stage2")
    manifest = {
        "base_model": args.model,
        "adapters": [{"name": "stage2", "path": os.path.abspath(args.out)}],
        "method": "oracle_sb_base_ref_fresh",
        "params": {"epsilon": args.epsilon, "beta": args.beta, "lr": args.lr,
                   "oracle_delta": args.oracle_delta},
        "note": "Fresh LoRA. Oracle SB weights (was_flipped). Base model reference.",
    }
    with open(os.path.join(args.out, "M2_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n✅ Job 2 complete → {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
