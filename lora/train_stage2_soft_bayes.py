#!/usr/bin/env python3
"""
Stage-2 Soft-Bayes LoRA training.

1. Load M1 (base + stage1 adapter), freeze it.
2. Score D2 pairs: Δ_i = log P_M1(chosen|prompt) - log P_M1(rejected|prompt).
3. Compute posterior w_i = P(T=1 | Y=1, M1) using Bayes + RR noise model.
4. Add stage2 adapter, train with soft-label BCE on pairwise logistic objective.
5. Save stage2 adapter + manifest.

No DP flipping is applied here. The RR noise parameters (epsilon / p_keep) are
used only to compute the posterior correction.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

# Default: D2 output from rr_stream_flip (partition 1, RR-flipped). Run rr_stream_flip first.
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


def find_first_key(cols: List[str], candidates: List[str]) -> str:
    for key in candidates:
        if key in cols:
            return key
    return ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-2 Soft-Bayes LoRA training")
    # Model
    ap.add_argument("--model", required=True, help="Base model name or path")
    ap.add_argument("--stage1_adapter", required=True, help="Path to Stage-1 LoRA adapter or HF repo")
    ap.add_argument("--stage1_subfolder", default=None, help="Subfolder in HF repo (e.g. gpt2-large/truthydpo/stage1/eps_1.0)")
    ap.add_argument("--data", default=DEFAULT_DATA, help="Full dataset (JSONL); D2 selected via partition args")
    ap.add_argument(
        "--out",
        default=None,
        help="Output dir (default: outputs/stage2_soft_bayes_lora_{model}_{stage1}).",
    )
    ap.add_argument(
        "--manifest_out",
        default=None,
        help="Manifest JSON path (default: {out}/M2_manifest.json).",
    )
    ap.add_argument(
        "--output_suffix",
        default="",
        help="Appended to auto-derived --out when --out is not set (e.g. eps1_seed42).",
    )

    # Columns
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--chosen_key", default="chosen")
    ap.add_argument("--rejected_key", default="rejected")

    # RR noise model
    ap.add_argument("--epsilon", type=float, default=1.0, help="RR epsilon used in stage1")
    ap.add_argument("--p_keep", type=float, default=None, help="Override p_keep directly")

    # Soft Bayes params
    ap.add_argument("--tau", type=float, default=2.0, help="Temperature for sigmoid(Δ/τ)")
    ap.add_argument("--beta", type=float, default=0.1, help="DPO-like sharpness in soft BCE")
    ap.add_argument("--w_clamp_lo", type=float, default=0.01)
    ap.add_argument("--w_clamp_hi", type=float, default=0.99)
    ap.add_argument("--delta_clamp", type=float, default=10.0, help="Clamp Δ/τ to [-v,v]")

    # Scoring
    ap.add_argument("--score_batch_size", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=512)

    # Training
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)

    # LoRA
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="c_attn,c_proj,c_fc")

    return ap.parse_args()


def _path_slug(s: str) -> str:
    """Turn model/adapter path into a safe dir name."""
    return s.replace("/", "--").replace("\\", "--").strip("-")


@torch.no_grad()
def score_sequence(
    model: nn.Module,
    tok: PreTrainedTokenizerBase,
    prompt: str,
    response: str,
    device: torch.device,
    max_len: int,
) -> float:
    """Compute log P_model(response | prompt) as mean log-prob over response tokens."""
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    response_ids = tok.encode(response, add_special_tokens=False)
    input_ids = (prompt_ids + response_ids)[:max_len]
    input_ids = torch.tensor([input_ids], device=device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [1, L, V]

    # Shift: predict token t from position t-1
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    # Only sum over response tokens
    prompt_len = len(prompt_ids)
    if prompt_len >= input_ids.shape[1]:
        return 0.0
    response_log_probs = token_log_probs[0, prompt_len - 1:]  # offset by 1 for shift
    if response_log_probs.numel() == 0:
        return 0.0
    return response_log_probs.mean().item()


def compute_posteriors(
    deltas: List[float],
    p_keep: float,
    tau: float,
    delta_clamp: float,
    w_lo: float,
    w_hi: float,
) -> List[float]:
    """
    Compute posterior w_i = P(T=1 | Y=1, M1).

    q_i = sigmoid(Δ_i / τ)   (M1 prior)
    w_i = p_keep * q_i / (p_keep * q_i + (1 - p_keep) * (1 - q_i))
    """
    weights = []
    for d in deltas:
        scaled = max(-delta_clamp, min(delta_clamp, d / tau))
        q_i = 1.0 / (1.0 + math.exp(-scaled))
        numerator = p_keep * q_i
        denominator = p_keep * q_i + (1.0 - p_keep) * (1.0 - q_i)
        if denominator < 1e-12:
            w_i = 0.5
        else:
            w_i = numerator / denominator
        w_i = max(w_lo, min(w_hi, w_i))
        weights.append(w_i)
    return weights


class SoftBayesPairDataset(TorchDataset):
    """Dataset yielding (prompt_chosen_ids, prompt_rejected_ids, w_i) for soft DPO."""

    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        weights: List[float],
        tok: PreTrainedTokenizerBase,
        max_len: int,
    ):
        self.prompts = prompts
        self.chosen = chosen
        self.rejected = rejected
        self.weights = weights
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict:
        prompt = self.prompts[idx]
        prompt_ids = self.tok.encode(prompt, add_special_tokens=True)

        chosen_ids = self.tok.encode(self.chosen[idx], add_special_tokens=False)
        rejected_ids = self.tok.encode(self.rejected[idx], add_special_tokens=False)

        c_ids = (prompt_ids + chosen_ids)[: self.max_len]
        r_ids = (prompt_ids + rejected_ids)[: self.max_len]

        return {
            "chosen_ids": c_ids,
            "rejected_ids": r_ids,
            "prompt_len": len(prompt_ids),
            "w": self.weights[idx],
        }


def pad_to_max(sequences: List[List[int]], pad_id: int) -> torch.Tensor:
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_id] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)


def collate_soft_bayes(batch: List[Dict], pad_id: int) -> Dict:
    chosen_ids = [b["chosen_ids"] for b in batch]
    rejected_ids = [b["rejected_ids"] for b in batch]
    prompt_lens = [b["prompt_len"] for b in batch]
    ws = [b["w"] for b in batch]

    return {
        "chosen_ids": pad_to_max(chosen_ids, pad_id),
        "rejected_ids": pad_to_max(rejected_ids, pad_id),
        "prompt_lens": torch.tensor(prompt_lens, dtype=torch.long),
        "w": torch.tensor(ws, dtype=torch.float32),
    }


def compute_response_logprobs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    Compute mean log P(response | prompt) for each example in a batch.
    Returns shape [B].
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # [B, L, V]

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

    # Mask: only response tokens (after prompt), and not padding
    B, L_minus_1 = token_log_probs.shape
    positions = torch.arange(L_minus_1, device=input_ids.device).unsqueeze(0).expand(B, -1)
    # Response starts at prompt_len (in shifted space, prompt_len - 1)
    response_mask = positions >= (prompt_lens.unsqueeze(1) - 1)
    # Also mask padding
    pad_mask = shift_labels != pad_id
    mask = response_mask & pad_mask

    # Mean log-prob over response tokens per example
    masked_lp = token_log_probs * mask.float()
    counts = mask.float().sum(dim=1).clamp(min=1.0)
    return masked_lp.sum(dim=1) / counts


def soft_bayes_loss(
    m_chosen: torch.Tensor,
    m_rejected: torch.Tensor,
    w: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """
    Soft-label BCE on pairwise logistic:
    m_i = log P_M2(chosen|prompt) - log P_M2(rejected|prompt)
    L_i = -[w_i * log σ(β m_i) + (1 - w_i) * log(1 - σ(β m_i))]
    """
    m = m_chosen - m_rejected  # [B]
    log_sigmoid_pos = F.logsigmoid(beta * m)
    log_sigmoid_neg = F.logsigmoid(-beta * m)
    loss = -(w * log_sigmoid_pos + (1.0 - w) * log_sigmoid_neg)
    return loss.mean()


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if args.out is None:
        model_slug = _path_slug(os.path.basename(args.model.rstrip("/")))
        stage1_slug = _path_slug(os.path.basename(args.stage1_adapter.rstrip("/")))
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        args.out = f"outputs/stage2_soft_bayes_{model_slug}_{stage1_slug}{suffix}"
    if args.manifest_out is None:
        args.manifest_out = os.path.join(args.out, "M2_manifest.json")

    print(f"[Config] out={args.out} manifest={args.manifest_out}")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)

    torch.manual_seed(args.seed)

    # Compute p_keep from epsilon
    if args.p_keep is not None:
        p_keep = args.p_keep
    else:
        p_keep = math.exp(args.epsilon) / (math.exp(args.epsilon) + 1.0)
    print(f"[Config] epsilon={args.epsilon} p_keep={p_keep:.6f} tau={args.tau} beta={args.beta}")

    # Load data: D2 output from rr_stream_flip (already partitioned + RR-flipped)
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
    print(f"[Data] N={N} from {args.data} (D2 from rr_stream_flip)")

    # Tokenizer
    # OpenLLaMA model card recommends slow tokenizer to avoid tokenization mismatch.
    use_fast = False if "open_llama" in args.model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    # Device
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    dtype = torch.float16 if cuda_ok else torch.float32

    # Load M1 (base + stage1)
    print("[M1] Loading base model...")
    base_model = load_causal_lm_compat(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None,
    )
    base_model.config.use_cache = True
    if args.stage1_subfolder:
        m1 = PeftModel.from_pretrained(base_model, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1")
    else:
        m1 = PeftModel.from_pretrained(base_model, args.stage1_adapter, adapter_name="stage1")
    m1.set_adapter("stage1")
    m1 = m1.to(device)
    m1.eval()

    # Score D2 with M1
    print("[M1] Scoring D2 pairs...")
    deltas = []
    for i in range(N):
        s_c = score_sequence(m1, tok, prompts[i], chosen[i], device, args.max_len)
        s_r = score_sequence(m1, tok, prompts[i], rejected[i], device, args.max_len)
        d = s_c - s_r
        deltas.append(d)
        if i < 5 or (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N}] s_c={s_c:.4f} s_r={s_r:.4f} Δ={d:.4f}")

    # Compute posteriors
    weights = compute_posteriors(
        deltas, p_keep=p_keep, tau=args.tau,
        delta_clamp=args.delta_clamp, w_lo=args.w_clamp_lo, w_hi=args.w_clamp_hi,
    )
    w_mean = sum(weights) / len(weights)
    w_gt_half = sum(1 for w in weights if w > 0.5)
    print(f"[Posterior] w_mean={w_mean:.4f} w>0.5: {w_gt_half}/{N}")

    # Save scoring stats
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
    }
    with open(os.path.join(args.out, "scoring_stats.json"), "w") as f:
        json.dump(score_stats, f, indent=2)

    # Free M1/base refs from GPU before loading M2 (important for 7B on 24GB cards).
    del m1
    del base_model
    gc.collect()
    torch.cuda.empty_cache() if cuda_ok else None

    # Build dataset
    pair_ds = SoftBayesPairDataset(prompts, chosen, rejected, weights, tok, args.max_len)
    train_dl = DataLoader(
        pair_ds,
        batch_size=args.bsz,
        shuffle=True,
        collate_fn=lambda batch: collate_soft_bayes(batch, pad_id),
    )

    # Load M2 = base + stage1 (frozen) + stage2 (trainable)
    print("[M2] Loading base + stage1 + stage2...")
    # Keep M2 dtype aligned with device capability to avoid RTX 6000 OOM.
    base_model2 = load_causal_lm_compat(
        args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None,
    )
    base_model2.config.use_cache = False
    if args.stage1_subfolder:
        m2 = PeftModel.from_pretrained(base_model2, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1")
    else:
        m2 = PeftModel.from_pretrained(base_model2, args.stage1_adapter, adapter_name="stage1")

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    stage2_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    m2.add_adapter("stage2", stage2_config)
    try:
        m2.set_adapter(["stage1", "stage2"])
    except Exception:
        m2.set_adapter("stage2")

    # Freeze everything except stage2
    for name, param in m2.named_parameters():
        param.requires_grad = ".stage2." in name

    trainable_count = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in m2.parameters())
    print(f"[M2] trainable={trainable_count} total={total_count} ({100*trainable_count/max(1,total_count):.4f}%)")
    assert trainable_count > 0, "No trainable params."

    m2 = m2.to(device)
    m2.train()

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in m2.parameters() if p.requires_grad],
        lr=args.lr, eps=1e-6,
    )

    # Training loop
    global_step = 0
    accum_loss = 0.0
    max_steps = args.max_steps if args.max_steps > 0 else None

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_dl):
            c_ids = batch["chosen_ids"].to(device)
            r_ids = batch["rejected_ids"].to(device)
            p_lens = batch["prompt_lens"].to(device)
            w = batch["w"].to(device)

            m_c = compute_response_logprobs(m2, c_ids, p_lens, pad_id)
            m_r = compute_response_logprobs(m2, r_ids, p_lens, pad_id)

            loss = soft_bayes_loss(m_c, m_r, w, beta=args.beta)
            loss = loss / args.ga
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.ga == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in m2.parameters() if p.requires_grad],
                        args.max_grad_norm,
                    )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % 10 == 0 or global_step <= 3:
                    print(f"  [step={global_step}] loss={accum_loss:.6f} epoch={epoch}")
                accum_loss = 0.0

                if max_steps and global_step >= max_steps:
                    break

        if max_steps and global_step >= max_steps:
            break
        print(f"[Epoch {epoch+1}/{args.epochs}] completed")

    # Save stage2 adapter
    m2.save_pretrained(args.out, adapter_name="stage2")

    manifest = {
        "base_model": args.model,
        "adapters": [
            {"name": "stage1", "path": os.path.abspath(args.stage1_adapter)},
            {"name": "stage2", "path": os.path.abspath(args.out)},
        ],
        "adapter_order": ["stage1", "stage2"],
        "method": "soft_bayes",
        "params": {
            "epsilon": args.epsilon,
            "p_keep": p_keep,
            "tau": args.tau,
            "beta": args.beta,
        },
        "notes": "M2 = base + stage1(frozen) + stage2(soft-bayes trained)",
    }
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✅ Saved Stage-2 (Soft-Bayes) adapter to {args.out}")
    print(f"✅ Wrote manifest to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
