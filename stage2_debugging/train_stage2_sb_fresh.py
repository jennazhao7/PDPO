#!/usr/bin/env python3
"""
Stage-2 Soft-Bayes Fresh-Retrain (SB-Fresh).

Combines SB's label-denoising with MAP-Retrain's architectural insight:

  Step 1  Load M1 (base + stage1 adapter), freeze it.
  Step 2  Score D2 pairs: Δ_i = log π_M1(y|x) - log π_base(y|x).
  Step 3  Compute SB posterior weights w_i = P(T=1 | Y=1, M1) from Δ_i and RR noise model.
  Step 4  Pre-compute BASE model logprobs for each pair (reference at training time).
          NOTE: reference = BASE (not Stage1). This is the key architectural change.
  Step 5  Discard M1 from GPU; load a FRESH LoRA on base (no stage1 stacking).
  Step 6  Train with soft-label BCE using SB weights and base reference.
  Step 7  Save stage2 adapter + manifest (single adapter → eval uses base as ref).

Why this beats stacked SB:
  - Strong gradients: base is far from policy (like MAP-Retrain).
  - Soft denoising: SB weights down-weight noisy pairs (unlike MAP-Retrain).
  - Training/eval consistency: both use base as reference throughout.
  - DP guarantee: M1 is used only for post-processing of privatised D2 data.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

DEFAULT_DATA = "lora/preprocessing/d2_rr_flipped.jsonl"


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

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


def _path_slug(s: str) -> str:
    return s.replace("/", "--").replace("\\", "--").strip("-")


# ---------------------------------------------------------------------------
# Sequence scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_sequence(
    model: nn.Module,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    device: torch.device,
    max_len: int,
) -> float:
    """Sum of log P_model(token | context) over response tokens only."""
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    response_ids = tok.encode(response, add_special_tokens=False)
    input_ids = (prompt_ids + response_ids)[:max_len]
    input_ids = torch.tensor([input_ids], device=device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, L, V]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    prompt_len = len(prompt_ids)
    if prompt_len >= input_ids.shape[1]:
        return 0.0
    response_log_probs = token_log_probs[0, prompt_len - 1:]
    if response_log_probs.numel() == 0:
        return 0.0
    return response_log_probs.sum().item()


@torch.no_grad()
def dpo_reward_for_sequence(
    m1,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    device: torch.device,
    max_len: int,
) -> float:
    """DPO implicit reward: log π_M1(y|x) - log π_base(y|x)."""
    m1.set_adapter("stage1")
    s_m1 = score_sequence(m1, tok, prompt, response, device, max_len)
    with m1.disable_adapter():
        s_base = score_sequence(m1, tok, prompt, response, device, max_len)
    return s_m1 - s_base


# ---------------------------------------------------------------------------
# SB posterior weights
# ---------------------------------------------------------------------------

def compute_posteriors(
    deltas: List[float],
    p_keep: float,
    tau: float,
    delta_clamp: float,
    w_lo: float,
    w_hi: float,
) -> List[float]:
    """
    w_i = P(T=1 | Y=1, M1)
    q_i = sigmoid(Δ_i / τ)
    w_i = p_keep * q_i / (p_keep * q_i + (1-p_keep) * (1-q_i))
    """
    weights = []
    for d in deltas:
        scaled = max(-delta_clamp, min(delta_clamp, d / tau))
        q_i = 1.0 / (1.0 + math.exp(-scaled))
        numerator = p_keep * q_i
        denominator = p_keep * q_i + (1.0 - p_keep) * (1.0 - q_i)
        w_i = numerator / denominator if denominator >= 1e-12 else 0.5
        weights.append(max(w_lo, min(w_hi, w_i)))
    return weights


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------

class SBFreshPairDataset(TorchDataset):
    """
    Yields tokenised pairs with SB weight and BASE model reference logprobs.
    Reference = base (not Stage1) — consistent with fresh-retrain training objective.
    """

    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        weights: List[float],
        ref_chosen_lps: List[float],   # log π_base(chosen | prompt)
        ref_rejected_lps: List[float], # log π_base(rejected | prompt)
        tok: PreTrainedTokenizerBase,
        max_len: int,
    ):
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        self.weights = weights
        self.ref_chosen_lps = ref_chosen_lps
        self.ref_rejected_lps = ref_rejected_lps
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict:
        prompt_ids = self.tok.encode(self.prompts[idx], add_special_tokens=True)
        chosen_ids = self.tok.encode(self.chosen[idx], add_special_tokens=False)
        rejected_ids = self.tok.encode(self.rejected[idx], add_special_tokens=False)
        return {
            "chosen_ids": (prompt_ids + chosen_ids)[: self.max_len],
            "rejected_ids": (prompt_ids + rejected_ids)[: self.max_len],
            "prompt_len": len(prompt_ids),
            "w": self.weights[idx],
            "ref_chosen_lp": self.ref_chosen_lps[idx],
            "ref_rejected_lp": self.ref_rejected_lps[idx],
        }


def pad_to_max(sequences: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    return torch.tensor(
        [s + [pad_id] * (max_len - len(s)) for s in sequences], dtype=torch.long
    )


def collate_fn(batch: List[Dict], pad_id: int) -> Dict:
    return {
        "chosen_ids": pad_to_max([b["chosen_ids"] for b in batch], pad_id),
        "rejected_ids": pad_to_max([b["rejected_ids"] for b in batch], pad_id),
        "prompt_lens": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "w": torch.tensor([b["w"] for b in batch], dtype=torch.float32),
        "ref_chosen_lp": torch.tensor([b["ref_chosen_lp"] for b in batch], dtype=torch.float32),
        "ref_rejected_lp": torch.tensor([b["ref_rejected_lp"] for b in batch], dtype=torch.float32),
    }


# ---------------------------------------------------------------------------
# Training objective
# ---------------------------------------------------------------------------

def compute_response_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    B, L_minus_1 = token_log_probs.shape
    positions = torch.arange(L_minus_1, device=input_ids.device).unsqueeze(0).expand(B, -1)
    response_mask = positions >= (prompt_lens.unsqueeze(1) - 1)
    pad_mask = shift_labels != pad_id
    mask = response_mask & pad_mask

    return (token_log_probs * mask.float()).sum(dim=1)


def soft_bayes_loss(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    w: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Soft-label BCE with reference subtraction:
      r_c = log π(chosen) - log π_ref(chosen)
      r_r = log π(rejected) - log π_ref(rejected)
      m   = r_c - r_r
      L   = -[w * log σ(β m) + (1-w) * log(1 - σ(β m))]
    """
    m = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    loss = -(w * F.logsigmoid(beta * m) + (1.0 - w) * F.logsigmoid(-beta * m))
    return loss.mean()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-2 SB-Fresh: SB weights + base reference")

    # Model
    ap.add_argument("--model", required=True, help="Base model name or path")
    ap.add_argument(
        "--stage1_adapter", required=True,
        help="Stage-1 adapter path — used ONLY for M1 scoring (not as training reference).",
    )
    ap.add_argument("--stage1_subfolder", default=None, help="Subfolder in HF repo for Stage-1")
    ap.add_argument("--data", default=DEFAULT_DATA, help="D2 noisy preference JSONL")
    ap.add_argument("--out", default=None, help="Output directory")
    ap.add_argument("--manifest_out", default=None, help="Manifest JSON path")
    ap.add_argument("--output_suffix", default="", help="Suffix appended to auto-derived --out")

    # Column names
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--chosen_key", default="chosen")
    ap.add_argument("--rejected_key", default="rejected")

    # RR noise model
    ap.add_argument("--epsilon", type=float, default=1.0, help="RR epsilon used in Stage 1")
    ap.add_argument("--p_keep", type=float, default=None, help="Override p_keep directly")

    # SB posterior params
    ap.add_argument("--tau", type=float, default=0.0, help="Temperature for sigmoid(Δ/τ). 0=auto.")
    ap.add_argument("--beta", type=float, default=0.5, help="DPO sharpness in soft BCE")
    ap.add_argument("--w_clamp_lo", type=float, default=0.01)
    ap.add_argument("--w_clamp_hi", type=float, default=0.99)
    ap.add_argument("--delta_clamp", type=float, default=500.0)

    # Scoring
    ap.add_argument("--max_len", type=int, default=512)

    # Training
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        default=(
            "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,"
            "query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
        ),
    )
    ap.add_argument("--bf16", action="store_true")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.out is None:
        model_slug = _path_slug(os.path.basename(args.model.rstrip("/")))
        s1_slug = _path_slug(os.path.basename(args.stage1_adapter.rstrip("/")))
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        args.out = f"outputs/stage2_sb_fresh_{model_slug}_{s1_slug}{suffix}"
    if args.manifest_out is None:
        args.manifest_out = os.path.join(args.out, "M2_manifest.json")

    print(f"[Config] method=sb_fresh out={args.out}")
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)

    # p_keep from epsilon
    if args.p_keep is not None:
        p_keep = args.p_keep
    else:
        p_keep = math.exp(args.epsilon) / (math.exp(args.epsilon) + 1.0)
    print(f"[Config] epsilon={args.epsilon} p_keep={p_keep:.6f} tau={args.tau} beta={args.beta}")

    # Load data
    raw = load_dataset("json", data_files=args.data)["train"]
    cols = raw.column_names
    pk = args.prompt_key if args.prompt_key in cols else find_first_key(cols, ["prompt", "question", "instruction"])
    ck = args.chosen_key if args.chosen_key in cols else find_first_key(cols, ["chosen", "answer", "response"])
    rk = args.rejected_key if args.rejected_key in cols else find_first_key(cols, ["rejected"])
    if not pk or not ck or not rk:
        raise ValueError(f"Cannot find prompt/chosen/rejected in columns: {cols}")

    prompts = raw[pk]
    chosen = raw[ck]
    rejected = raw[rk]
    N = len(prompts)
    print(f"[Data] N={N} from {args.data}")

    # Tokenizer
    use_fast = False if "open_llama" in args.model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    # Device / dtype
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    dtype = torch.bfloat16 if (cuda_ok and args.bf16) else (torch.float16 if cuda_ok else torch.float32)

    # ── Step 1-4: Score with M1 and collect SB weights + BASE ref logprobs ──
    cache_path = os.path.join(args.out, "SBFresh_cache.pt")
    if os.path.exists(cache_path):
        print(f"[Cache] Found {cache_path}, loading scoring data...")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        deltas = cache["deltas"]
        ref_chosen_lps = cache["ref_chosen_lps"]
        ref_rejected_lps = cache["ref_rejected_lps"]
        weights = cache["weights"]
        args.tau = cache.get("tau", args.tau)
        p_keep = cache.get("p_keep", p_keep)
        w_mean = sum(weights) / len(weights)
        w_gt_half = sum(1 for w in weights if w > 0.5)
        print(f"[Posterior] Loaded cache. w_mean={w_mean:.4f}  w>0.5: {w_gt_half}/{N}")
    else:
        print("[M1] Loading base model + stage1 adapter for scoring...")
        base_model = load_causal_lm_compat(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None,
        )
        base_model.config.use_cache = True
        if args.stage1_subfolder:
            m1 = PeftModel.from_pretrained(
                base_model, args.stage1_adapter,
                subfolder=args.stage1_subfolder, adapter_name="stage1",
            )
        else:
            m1 = PeftModel.from_pretrained(base_model, args.stage1_adapter, adapter_name="stage1")
        m1.set_adapter("stage1")
        m1 = m1.to(device).eval()

        print("[M1] Scoring D2 pairs: Δ_i = log π_M1 - log π_base, ref = log π_BASE ...")
        deltas: List[float] = []
        ref_chosen_lps: List[float] = []
        ref_rejected_lps: List[float] = []

        for i in range(N):
            # SB delta uses M1 DPO reward (length-normalised)
            dr_c = dpo_reward_for_sequence(m1, tok, prompts[i], chosen[i], device, args.max_len)
            dr_r = dpo_reward_for_sequence(m1, tok, prompts[i], rejected[i], device, args.max_len)
            deltas.append(dr_c - dr_r)

            # Reference logprobs = BASE (disable Stage1 adapter)
            # These are pre-computed once and frozen — the training reference.
            with m1.disable_adapter():
                ref_c = score_sequence(m1, tok, prompts[i], chosen[i], device, args.max_len)
                ref_r = score_sequence(m1, tok, prompts[i], rejected[i], device, args.max_len)
            ref_chosen_lps.append(ref_c)
            ref_rejected_lps.append(ref_r)

            if i < 5 or (i + 1) % 100 == 0:
                print(
                    f"  [{i+1}/{N}] Δ={deltas[-1]:.4f}  "
                    f"ref_c={ref_c:.4f}  ref_r={ref_r:.4f}"
                )

        # Guardrail: M1 must produce real signal
        max_abs_delta = max(abs(x) for x in deltas) if deltas else 0.0
        if max_abs_delta < 1e-8:
            raise RuntimeError(
                "All M1 deltas are ~0. Stage1 adapter produced no signal on D2. "
                "Check --stage1_adapter path before continuing."
            )

        # Auto-tune tau using principled spread approach
        if args.tau <= 0:
            import numpy as np
            std_delta = float(np.std(deltas))
            # Set tau so that pairs 1 std apart in delta get weights ~0.3 apart
            # (sigmoid(std_delta / tau) - sigmoid(-std_delta / tau) ≈ 0.3)
            args.tau = max(0.5, std_delta / 1.2)
            print(f"[Auto-tau] std_delta={std_delta:.4f} → principled tau={args.tau:.4f}")

        # Compute SB posteriors
        weights = compute_posteriors(
            deltas, p_keep=p_keep, tau=args.tau,
            delta_clamp=args.delta_clamp, w_lo=args.w_clamp_lo, w_hi=args.w_clamp_hi,
        )
        w_mean = sum(weights) / len(weights)
        w_gt_half = sum(1 for w in weights if w > 0.5)
        print(f"[Posterior] w_mean={w_mean:.4f}  w>0.5: {w_gt_half}/{N}")

        score_stats = {
            "N": N,
            "delta_mean": sum(deltas) / N,
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "w_mean": w_mean,
            "w_gt_half": w_gt_half,
            "p_keep": p_keep,
            "tau": args.tau,
            "beta": args.beta,
            "ref_model": "base",
            "method": "sb_fresh_retrain",
        }
        with open(os.path.join(args.out, "scoring_stats.json"), "w") as f:
            json.dump(score_stats, f, indent=2)
        print(f"[Stats] saved scoring_stats.json")

        torch.save({
            "deltas": deltas,
            "ref_chosen_lps": ref_chosen_lps,
            "ref_rejected_lps": ref_rejected_lps,
            "weights": weights,
            "tau": args.tau,
            "p_keep": p_keep,
        }, cache_path)
        print(f"[Cache] Saved scoring data to {cache_path}")

        # Free M1 from GPU
        del m1, base_model
        gc.collect()
        if cuda_ok:
            torch.cuda.empty_cache()

    # ── Step 5: Build DataLoader ──
    pair_ds = SBFreshPairDataset(
        prompts, chosen, rejected, weights,
        ref_chosen_lps, ref_rejected_lps,
        tok, args.max_len,
    )
    train_dl = DataLoader(
        pair_ds,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id),
    )

    # ── Step 6: Load M2 = fresh LoRA on BASE (no stage1 stacking) ──
    if args.resume_from_checkpoint:
        print(f"[M2] Resuming from checkpoint: {args.resume_from_checkpoint}...")
        base_model2 = load_causal_lm_compat(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None,
        )
        base_model2.config.use_cache = False
        base_model2.gradient_checkpointing_enable()
        resume_dir = args.resume_from_checkpoint
        if not os.path.exists(os.path.join(resume_dir, "adapter_config.json")) and os.path.exists(os.path.join(resume_dir, "stage2", "adapter_config.json")):
            resume_dir = os.path.join(resume_dir, "stage2")
        m2 = PeftModel.from_pretrained(base_model2, resume_dir, is_trainable=True, adapter_name="stage2")
    else:
        print("[M2] Loading FRESH base model (no stage1)...")
        base_model2 = load_causal_lm_compat(
            args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None,
        )
        base_model2.config.use_cache = False
        base_model2.gradient_checkpointing_enable()

        target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
        stage2_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        m2 = get_peft_model(base_model2, stage2_config, adapter_name="stage2")

    trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m2.parameters())
    print(f"[M2] trainable={trainable:,}  total={total:,}  ({100*trainable/max(1,total):.4f}%)")
    assert trainable > 0, "No trainable params in M2."

    m2 = m2.to(device).train()

    optimizer = torch.optim.AdamW(
        [p for p in m2.parameters() if p.requires_grad],
        lr=args.lr, eps=1e-6,
    )

    # ── Step 7: Train ──
    global_step = 0
    accum_loss = 0.0
    max_steps = args.max_steps if args.max_steps > 0 else None
    ckpt_dirs: List[str] = []

    if args.resume_from_checkpoint:
        try:
            global_step = int(args.resume_from_checkpoint.rstrip("/").split("-")[-1])
            print(f"[Resume] Fast-forwarding to global_step={global_step}")
        except:
            print(f"[Resume] Warning: Could not parse step from {args.resume_from_checkpoint}")

    batches_to_skip = global_step * args.ga
    total_batches_processed = 0

    def save_checkpoint(step: int) -> None:
        ckpt_dir = os.path.join(args.out, f"checkpoint-step-{step}")
        m2.save_pretrained(ckpt_dir, adapter_name="stage2")
        ckpt_dirs.append(ckpt_dir)
        print(f"[Checkpoint] saved {ckpt_dir}")
        while len(ckpt_dirs) > max(1, args.save_total_limit):
            old = ckpt_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"[Checkpoint] pruned {old}")

    if batches_to_skip > 0:
        print(f"[Resume] Skipping {batches_to_skip} batches to resume from step {global_step}...")

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_dl):
            total_batches_processed += 1
            if total_batches_processed <= batches_to_skip:
                continue

            c_ids = batch["chosen_ids"].to(device)
            r_ids = batch["rejected_ids"].to(device)
            p_lens = batch["prompt_lens"].to(device)
            w = batch["w"].to(device)
            ref_c = batch["ref_chosen_lp"].to(device)
            ref_r = batch["ref_rejected_lp"].to(device)

            pi_c = compute_response_logprobs(m2, c_ids, p_lens, pad_id)
            pi_r = compute_response_logprobs(m2, r_ids, p_lens, pad_id)

            loss = soft_bayes_loss(pi_c, pi_r, ref_c, ref_r, w, beta=args.beta) / args.ga
            loss.backward()
            accum_loss += loss.item()

            if total_batches_processed % args.ga == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in m2.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0 or global_step <= 3:
                    print(f"  [step={global_step}] loss={accum_loss:.6f}  epoch={epoch}", flush=True)
                accum_loss = 0.0

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(global_step)
                if max_steps and global_step >= max_steps:
                    break

        # Flush leftover gradient accumulation batch
        if len(train_dl) % args.ga != 0 and accum_loss != 0.0 and (not max_steps or global_step < max_steps):
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in m2.parameters() if p.requires_grad],
                    args.max_grad_norm,
                )
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print(f"  [step={global_step}] loss={accum_loss:.6f}  epoch={epoch} (flush)", flush=True)
            accum_loss = 0.0
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(global_step)

        if max_steps and global_step >= max_steps:
            break
        print(f"[Epoch {epoch+1}/{args.epochs}] done")

    # ── Step 8: Save ──
    m2.save_pretrained(args.out, adapter_name="stage2")

    # Single-adapter manifest: eval_preference_accuracy.py will detect has_stage1=False
    # and evaluate as:  policy = base + stage2,  ref = base
    # This is consistent with how this model was trained.
    manifest = {
        "base_model": args.model,
        "adapters": [
            {"name": "stage2", "path": os.path.abspath(args.out)},
        ],
        "method": "sb_fresh_retrain",
        "params": {
            "epsilon": args.epsilon,
            "p_keep": p_keep,
            "tau": args.tau,
            "beta": args.beta,
            "ref_at_training": "base",
            "stage1_used_for": "scoring_only",
            "stage1_adapter": args.stage1_adapter,
        },
        "note": (
            "Single fresh LoRA on base. Eval uses base as reference (consistent with training). "
            "Stage1 adapter was used ONLY to score D2 pairs for SB weight computation."
        ),
    }
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] Saved SB-Fresh adapter -> {args.out}")
    print(f"[Done] Manifest -> {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
