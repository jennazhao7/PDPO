#!/usr/bin/env python3
"""
DP-LoRA fine-tuning on a preference (DPO) or instruction (SFT) dataset.

Mirrors the plain-LoRA pipeline with differential privacy via manual DP-SGD.
Uses Opacus privacy accountants for epsilon tracking but implements the
per-sample clip-and-noise loop directly so it works with PEFT LoRA modules
and custom DPO loss.

Outputs written to --output_dir (appends /seed_{N} when --auto_seed_dir):
    adapter_model.safetensors   LoRA weights
    adapter_config.json         PEFT config
    run_manifest.json           Full run config, git hash, data stats, DP/LoRA params
    split_indices.json          Train / val index lists
    trainable_params.json       Trainable parameter names (DP/non-DP comparison)
    metrics.jsonl               Per-step metrics (loss, grad_norm, lr, epsilon_spent, ...)
    summary.json                Final metrics + DP fields
    convergence_summary.json    best/final metric, steps_to_95pct, AUC
    eval_metric_vs_steps.png    Convergence plot
    generations.jsonl           Fixed-prompt generation samples

Privacy model:
    DP-SGD with per-example gradient clipping (norm C) and Gaussian noise
    (std = sigma * C). Accounting uses RDP (Renyi DP) via Opacus, assuming
    Poisson sub-sampling at rate q = batch_size / N. Fixed-size shuffled
    batches are treated under the Poisson assumption (standard practice;
    provides an upper bound on true epsilon).

Usage:
    # Single seed
    python train_dp_lora.py --model_id gpt2-medium --dataset_path d1.jsonl \\
        --output_dir outputs/dp_run --dp --epsilon 8.0 --target_modules c_attn,c_proj,c_fc

    # Five seeds (via run_multiseed.py)
    python run_multiseed.py --seeds 0 1 2 3 4 --base_output_dir outputs/dp_runs \\
        -- --model_id gpt2-medium --dataset_path d1.jsonl \\
        --dp --epsilon 8.0 --target_modules c_attn,c_proj,c_fc
"""
from __future__ import annotations

import argparse
import datetime
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
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_scheduler,
)

# ---------------------------------------------------------------------------
# Constants (identical to plain-lora)
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
# Shared utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


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
    has_chosen = bool(find_key(columns, CHOSEN_CANDIDATES))
    has_rejected = bool(find_key(columns, REJECTED_CANDIDATES))
    return "dpo" if (has_chosen and has_rejected) else "sft"


def append_jsonl(path: str, record: Dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_split_indices(
    n: int, val_frac: float, seed: int
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = max(1, int(n * val_frac))
    return indices[n_val:], indices[:n_val]


# ---------------------------------------------------------------------------
# Log-prob helpers (differentiable — used in training loop)
# ---------------------------------------------------------------------------

def response_logprob(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
    pad_id: int,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    L = token_lp.shape[1]
    positions = torch.arange(L, device=input_ids.device)
    response_mask = positions >= (prompt_len - 1)
    pad_mask = shift_labels[0] != pad_id
    mask = response_mask & pad_mask

    if mask.sum() == 0:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
    return (token_lp[0] * mask.float()).sum() / mask.float().sum()


# ---------------------------------------------------------------------------
# Per-example loss functions
# ---------------------------------------------------------------------------

def dpo_loss_single(
    model: torch.nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    prompt_len: int,
    ref_chosen_lp: float,
    ref_rejected_lp: float,
    beta: float,
    pad_id: int,
) -> torch.Tensor:
    pi_c = response_logprob(model, chosen_ids, prompt_len, pad_id)
    pi_r = response_logprob(model, rejected_ids, prompt_len, pad_id)
    pi_logratio = pi_c - pi_r
    ref_logratio = ref_chosen_lp - ref_rejected_lp
    return -F.logsigmoid(beta * (pi_logratio - ref_logratio))


def sft_loss_single(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    outputs = model(input_ids=input_ids.unsqueeze(0))
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ---------------------------------------------------------------------------
# Reference log-prob pre-computation (DPO only)
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_ref_logprobs(
    model: torch.nn.Module,
    train_data: List[Dict],
    device: torch.device,
    pad_id: int,
) -> None:
    model.eval()
    for i, ex in enumerate(train_data):
        with model.disable_adapter():
            c_lp = response_logprob(
                model, ex["chosen_ids"].to(device), ex["prompt_len"], pad_id
            ).item()
            r_lp = response_logprob(
                model, ex["rejected_ids"].to(device), ex["prompt_len"], pad_id
            ).item()
        ex["ref_chosen_lp"] = c_lp
        ex["ref_rejected_lp"] = r_lp
        if (i + 1) % 100 == 0 or i < 3:
            print(f"  [ref] {i+1}/{len(train_data)}  c={c_lp:.4f} r={r_lp:.4f}")
    model.train()


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_dpo_data(
    ds: Dataset, tok: PreTrainedTokenizerBase, max_len: int,
    prompt_key: str, chosen_key: str, rejected_key: str,
) -> List[Dict]:
    data = []
    for row in ds:
        prompt_ids = tok.encode(row[prompt_key], add_special_tokens=True)
        chosen_ids = tok.encode(row[chosen_key], add_special_tokens=False)
        rejected_ids = tok.encode(row[rejected_key], add_special_tokens=False)
        c_ids = torch.tensor([(prompt_ids + chosen_ids)[:max_len]])
        r_ids = torch.tensor([(prompt_ids + rejected_ids)[:max_len]])
        data.append({"chosen_ids": c_ids, "rejected_ids": r_ids, "prompt_len": len(prompt_ids)})
    return data


def prepare_sft_data(
    ds: Dataset, tok: PreTrainedTokenizerBase, max_len: int,
    prompt_key: str, response_key: str,
) -> List[Dict]:
    data = []
    for row in ds:
        prompt_ids = tok.encode(row[prompt_key], add_special_tokens=True)
        response_ids = tok.encode(row[response_key], add_special_tokens=False)
        input_ids = (prompt_ids + response_ids)[:max_len]
        labels = [-100] * len(prompt_ids) + response_ids
        labels = labels[:max_len]
        data.append({"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)})
    return data


# ---------------------------------------------------------------------------
# Privacy accounting (Opacus or minimal fallback for older PyTorch)
# ---------------------------------------------------------------------------

def _make_minimal_rdp_accountant():
    """Minimal RDP accountant when Opacus fails (e.g. PyTorch < 2.1 lacks RMSNorm)."""
    import math
    from scipy import special

    def _log_add(logx: float, logy: float) -> float:
        a, b = min(logx, logy), max(logx, logy)
        if a == -np.inf:
            return b
        return math.log1p(math.exp(a - b)) + b

    def _compute_rdp(q: float, sigma: float, alpha: float) -> float:
        if q == 0:
            return 0.0
        if sigma == 0:
            return np.inf
        if q >= 1.0:
            return alpha / (2 * sigma**2)
        if np.isinf(alpha):
            return np.inf
        # Integer alpha: exact formula from Wang et al. (Opacus/Google TF Privacy)
        if float(alpha).is_integer():
            log_a = -np.inf
            for i in range(int(alpha) + 1):
                coef = special.binom(int(alpha), i)
                log_coef = math.log(abs(coef))
                j = int(alpha) - i
                log_t = log_coef + i * math.log(q) + j * math.log(1 - q)
                s = log_t + (i * i - i) / (2 * (sigma**2))
                log_a = _log_add(log_a, s)
            return float(log_a) / (alpha - 1)
        # Fractional alpha: conservative upper bound (Gaussian w/o subsampling)
        return alpha / (2 * sigma**2)

    class MinimalRDPAccountant:
        def __init__(self):
            self.history: List[Tuple[float, float, int]] = []

        def step(self, *, noise_multiplier: float, sample_rate: float) -> None:
            if self.history and self.history[-1][:2] == (noise_multiplier, sample_rate):
                nm, sr, n = self.history.pop()
                self.history.append((nm, sr, n + 1))
            else:
                self.history.append((noise_multiplier, sample_rate, 1))

        def get_epsilon(self, delta: float, **kwargs: Any) -> float:
            if not self.history:
                return 0.0
            alphas = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
            rdp = np.zeros(len(alphas))
            for nm, sr, n in self.history:
                rdp += np.array([_compute_rdp(sr, nm, a) * n for a in alphas])
            eps = rdp - (np.log(delta) + np.log(alphas)) / (np.array(alphas) - 1) + np.log((np.array(alphas) - 1) / np.array(alphas))
            return float(np.nanmin(eps))

    return MinimalRDPAccountant()


def make_accountant(accountant_type: str = "rdp"):
    if accountant_type != "rdp":
        try:
            from opacus.accountants import PRVAccountant
            return PRVAccountant()
        except (ImportError, AttributeError) as e:
            print(f"[DP] Opacus PRV failed ({e}), falling back to minimal RDP.")
            return _make_minimal_rdp_accountant()
    try:
        from opacus.accountants import RDPAccountant
        return RDPAccountant()
    except (ImportError, AttributeError) as e:
        print(f"[DP] Opacus import failed ({e}), using minimal RDP accountant.")
        return _make_minimal_rdp_accountant()


def resolve_noise_multiplier(
    target_epsilon: float, target_delta: float, sample_rate: float, steps: int,
) -> float:
    try:
        from opacus.accountants.utils import get_noise_multiplier
        return get_noise_multiplier(
            target_epsilon=target_epsilon, target_delta=target_delta,
            sample_rate=sample_rate, steps=steps,
        )
    except (ImportError, AttributeError, TypeError):
        pass
    # Fallback: binary search with minimal accountant
    acc_factory = _make_minimal_rdp_accountant
    lo, hi = 0.001, 500.0
    for _ in range(64):
        mid = (lo + hi) / 2.0
        acc = acc_factory()
        for _ in range(steps):
            acc.step(noise_multiplier=mid, sample_rate=sample_rate)
        eps = acc.get_epsilon(delta=target_delta)
        if eps > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# Evaluation (identical to plain-lora)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequence_logprob(
    model: torch.nn.Module, tok: PreTrainedTokenizerBase,
    prompt: str, response: str, device: torch.device, max_len: int,
) -> float:
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    response_ids = tok.encode(response, add_special_tokens=False)
    input_ids = (prompt_ids + response_ids)[:max_len]
    input_ids_t = torch.tensor([input_ids], device=device)
    logits = model(input_ids=input_ids_t).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids_t[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    prompt_len = len(prompt_ids)
    if prompt_len >= input_ids_t.shape[1]:
        return 0.0
    response_lp = token_lp[0, prompt_len - 1:]
    if response_lp.numel() == 0:
        return 0.0
    return response_lp.mean().item()


def eval_pairwise_accuracy(
    model: torch.nn.Module, tok: PreTrainedTokenizerBase,
    ds: Dataset, prompt_key: str, chosen_key: str, rejected_key: str,
    device: torch.device, max_len: int, max_eval: int = 200,
) -> Dict[str, float]:
    model.eval()
    correct = total = 0
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
    model: torch.nn.Module, tok: PreTrainedTokenizerBase,
    prompts: List[str], device: torch.device,
    max_new_tokens: int = 128, temperature: float = 0.7, top_p: float = 0.9,
) -> List[Dict[str, str]]:
    model.eval()
    results = []
    for prompt in prompts:
        inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, do_sample=True, pad_token_id=tok.pad_token_id,
        )
        generated = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({"prompt": prompt, "generation": generated})
    return results


# ---------------------------------------------------------------------------
# Convergence summary + plot (post-training)
# ---------------------------------------------------------------------------

def compute_convergence_summary(metrics: List[Dict], metric_key: str = "eval_pairwise_accuracy") -> Dict[str, Any]:
    """Compute convergence statistics from metrics.jsonl records."""
    eval_pts = [(m["step"], m[metric_key]) for m in metrics if metric_key in m]
    loss_pts = [(m["step"], m["eval_loss"]) for m in metrics if "eval_loss" in m]
    train_pts = [(m["step"], m["loss"]) for m in metrics if "loss" in m]

    summary: Dict[str, Any] = {}

    if eval_pts:
        steps_e, vals_e = zip(*eval_pts)
        best_val = max(vals_e)
        best_idx = vals_e.index(best_val)
        best_step = steps_e[best_idx]
        final_val = vals_e[-1]
        final_step = steps_e[-1]

        # Steps to 95% of best
        threshold = 0.95 * best_val
        steps_to_95 = None
        for s, v in eval_pts:
            if v >= threshold:
                steps_to_95 = s
                break

        # AUC (trapezoidal)
        auc = 0.0
        for i in range(1, len(eval_pts)):
            dx = steps_e[i] - steps_e[i - 1]
            auc += 0.5 * (vals_e[i] + vals_e[i - 1]) * dx

        summary["best_eval_metric"] = best_val
        summary["best_step"] = best_step
        summary["final_eval_metric"] = final_val
        summary["final_step"] = final_step
        summary["steps_to_95pct_best"] = steps_to_95
        summary["auc_eval_metric_vs_steps"] = auc
        summary["metric_key"] = metric_key
    else:
        summary["best_eval_metric"] = None
        summary["note"] = f"no eval points found for key={metric_key}"

    if loss_pts:
        steps_l, vals_l = zip(*loss_pts)
        auc_loss = 0.0
        for i in range(1, len(loss_pts)):
            dx = steps_l[i] - steps_l[i - 1]
            auc_loss += 0.5 * (vals_l[i] + vals_l[i - 1]) * dx
        summary["auc_eval_loss_vs_steps"] = auc_loss

    if train_pts:
        summary["train_loss_first"] = train_pts[0][1]
        summary["train_loss_final"] = train_pts[-1][1]

    return summary


def plot_convergence(metrics: List[Dict], output_path: str, metric_key: str = "eval_pairwise_accuracy") -> None:
    """Save eval_metric_vs_steps.png using matplotlib only."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not installed; skipping convergence plot.")
        return

    eval_pts = [(m["step"], m[metric_key]) for m in metrics if metric_key in m]
    train_pts = [(m["step"], m["loss"]) for m in metrics if "loss" in m]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    if eval_pts:
        steps_e, vals_e = zip(*eval_pts)
        ax1.plot(steps_e, vals_e, "b-o", markersize=4, label=f"eval {metric_key}")
        ax1.set_xlabel("global_step")
        ax1.set_ylabel(metric_key, color="b")
        ax1.tick_params(axis="y", labelcolor="b")

    if train_pts:
        ax2 = ax1.twinx()
        steps_t, vals_t = zip(*train_pts)
        ax2.plot(steps_t, vals_t, "r-", alpha=0.5, linewidth=0.8, label="train loss")
        ax2.set_ylabel("train loss", color="r")
        ax2.tick_params(axis="y", labelcolor="r")

    ax1.set_title("Convergence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved {output_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DP-LoRA fine-tuning (DPO or SFT) with differential privacy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Same as plain-lora ---
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto_seed_dir", action="store_true",
                    help="Append /seed_{N} to output_dir automatically.")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--split_indices_json", type=str, default=None,
                    help="Load split from a plain-LoRA run for fair comparison.")

    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=5e-5)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=16)
    ap.add_argument("--max_seq_length", type=int, default=512)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj")

    ap.add_argument("--dpo_beta", type=float, default=0.1)

    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--load_in_8bit", action="store_true")

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_every_steps", type=int, default=0,
                    help="Run eval every N optimizer steps. 0 = eval only at end.")
    ap.add_argument("--eval_prompts_json", type=str, default=None)
    ap.add_argument("--max_eval_samples", type=int, default=200)
    ap.add_argument("--gen_max_new_tokens", type=int, default=128)

    # --- DP-specific ---
    ap.add_argument("--dp", action="store_true", help="Enable differential privacy.")
    ap.add_argument("--epsilon", type=float, default=None)
    ap.add_argument("--noise_multiplier", type=float, default=None)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--accountant", type=str, default="rdp", choices=["rdp", "prv"])
    ap.add_argument("--secure_rng", action="store_true")

    ap.add_argument("--config", type=str, default=None)

    args = ap.parse_args(argv)

    if args.config:
        with open(args.config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if not hasattr(args, k):
                ap.error(f"Unknown config key: {k}")
            default = ap.get_default(k)
            if getattr(args, k) == default:
                setattr(args, k, v)

    if args.dp:
        if args.epsilon is None and args.noise_multiplier is None:
            ap.error("DP enabled: provide --epsilon or --noise_multiplier.")
        if args.epsilon is not None and args.noise_multiplier is not None:
            ap.error("Provide only one of --epsilon or --noise_multiplier.")
        if args.fp16 or args.bf16:
            print("[DP] WARNING: fp16/bf16 auto-disabled for DP-SGD correctness.")
            args.fp16 = False
            args.bf16 = False

    if args.auto_seed_dir:
        args.output_dir = os.path.join(args.output_dir, f"seed_{args.seed}")

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Dataset
    # -----------------------------------------------------------------------
    if os.path.isfile(args.dataset_path):
        raw = load_dataset("json", data_files=args.dataset_path)["train"]
    else:
        raw = load_dataset(args.dataset_path, split="train")

    cols = raw.column_names
    ds_type = detect_dataset_type(cols)
    print(f"[Data] type={ds_type} columns={cols} N={len(raw)}")

    prompt_key = find_key(cols, PROMPT_CANDIDATES)
    if not prompt_key:
        raise ValueError(f"No prompt column found in {cols}")

    if ds_type == "dpo":
        chosen_key = find_key(cols, CHOSEN_CANDIDATES)
        rejected_key = find_key(cols, REJECTED_CANDIDATES)
        if not chosen_key or not rejected_key:
            raise ValueError(f"DPO missing chosen/rejected in {cols}")
    else:
        response_key = find_key(cols, RESPONSE_CANDIDATES + CHOSEN_CANDIDATES)
        if not response_key:
            raise ValueError(f"SFT missing response column in {cols}")

    # -----------------------------------------------------------------------
    # 2. Split
    # -----------------------------------------------------------------------
    if args.split_indices_json:
        with open(args.split_indices_json, "r") as f:
            split = json.load(f)
        train_idx, val_idx = split["train"], split["val"]
        print(f"[Data] loaded split from {args.split_indices_json}")
    else:
        train_idx, val_idx = stable_split_indices(len(raw), args.val_fraction, args.seed)

    write_json(
        os.path.join(args.output_dir, "split_indices.json"),
        {"train": train_idx, "val": val_idx, "seed": args.seed, "val_fraction": args.val_fraction},
    )
    train_ds = raw.select(train_idx)
    val_ds = raw.select(val_idx)
    print(f"[Data] train={len(train_ds)} val={len(val_ds)}")

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

    # -----------------------------------------------------------------------
    # 3. Tokenizer + Model + LoRA
    # -----------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    pad_id = tok.pad_token_id

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")

    model_kwargs: Dict[str, Any] = {
        "low_cpu_mem_usage": True, "trust_remote_code": False,
        "device_map": None, "torch_dtype": torch.float32,
    }
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True

    print("[Model] loading...")
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
    model.config.use_cache = False

    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--target_modules must include at least one module.")

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, target_modules=target_modules,
        lora_dropout=args.lora_dropout if not args.dp else 0.0,
        bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    trainable_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for _, p in trainable_params)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"[Params] trainable={trainable_count} total={total_count} "
          f"({100*trainable_count/max(1,total_count):.4f}%)")

    write_json(os.path.join(args.output_dir, "trainable_params.json"),
               {"names": [n for n, _ in trainable_params], "count": trainable_count, "total": total_count})

    for n, _ in trainable_params:
        if "lora_" not in n.lower():
            raise RuntimeError(f"Non-LoRA param is trainable: {n}")

    # -----------------------------------------------------------------------
    # 4. Prepare data
    # -----------------------------------------------------------------------
    if ds_type == "dpo":
        train_data = prepare_dpo_data(train_ds, tok, args.max_seq_length, "prompt", "chosen", "rejected")
    else:
        train_data = prepare_sft_data(train_ds, tok, args.max_seq_length, prompt_key, response_key)
    N = len(train_data)

    if ds_type == "dpo":
        print("[Ref] pre-computing reference log-probs...")
        precompute_ref_logprobs(model, train_data, device, pad_id)

    # -----------------------------------------------------------------------
    # 5. DP config
    # -----------------------------------------------------------------------
    logical_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    effective_batch_size = logical_batch_size  # single GPU
    sample_rate = logical_batch_size / N

    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        total_steps = math.ceil(N * args.num_train_epochs / logical_batch_size)

    if args.dp:
        delta = args.delta if args.delta is not None else 1.0 / N
        if args.epsilon is not None:
            print(f"[DP] Resolving sigma for eps={args.epsilon} delta={delta} q={sample_rate:.6f} T={total_steps}...")
            sigma = resolve_noise_multiplier(args.epsilon, delta, sample_rate, total_steps)
        else:
            sigma = args.noise_multiplier
            delta = args.delta if args.delta is not None else 1.0 / N
        accountant = make_accountant(args.accountant)
        C = args.max_grad_norm
        print(f"[DP] sigma={sigma:.6f} C={C} q={sample_rate:.6f} delta={delta} T={total_steps}")
    else:
        sigma = 0.0
        delta = None
        C = args.max_grad_norm
        accountant = None

    # -----------------------------------------------------------------------
    # 6. run_manifest.json
    # -----------------------------------------------------------------------
    manifest: Dict[str, Any] = {
        "git_hash": get_git_hash(),
        "timestamp": datetime.datetime.now().isoformat(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_ok,
        "cli_args": vars(args),
        "model_id": args.model_id,
        "dataset_path": args.dataset_path,
        "dataset_type": ds_type,
        "dataset_columns": cols,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "dp_enabled": args.dp,
        "noise_multiplier": sigma if args.dp else None,
        "max_grad_norm": C,
        "target_epsilon": args.epsilon,
        "delta": delta,
        "accountant": args.accountant if args.dp else None,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": target_modules,
        "trainable_params": trainable_count,
        "total_params": total_count,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "total_steps": total_steps,
        "eval_every_steps": args.eval_every_steps,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
    }
    write_json(os.path.join(args.output_dir, "run_manifest.json"), manifest)

    title = "DP-LoRA Training" if args.dp else "LoRA Training (DP disabled)"
    print("=" * 60)
    print(title)
    print("=" * 60)
    for k, v in sorted(manifest.items()):
        if k != "cli_args":
            print(f"  {k}: {v}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 7. Optimizer + scheduler
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW([p for _, p in trainable_params], lr=args.learning_rate, eps=1e-8)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        args.lr_scheduler_type, optimizer=optimizer,
        num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    noise_generator = None
    if args.dp and args.secure_rng:
        noise_generator = torch.Generator(device=device)
        noise_generator.manual_seed(args.seed + 9999)

    # -----------------------------------------------------------------------
    # 8. Eval helper (inline, called during training)
    # -----------------------------------------------------------------------
    def run_eval_and_log(step: int, epoch_f: float) -> Dict[str, Any]:
        """Run eval on val set, log to metrics.jsonl, return metrics dict."""
        record: Dict[str, Any] = {"step": step, "epoch": round(epoch_f, 4), "is_eval": True}
        if ds_type == "dpo" and len(val_ds) > 0:
            pair_m = eval_pairwise_accuracy(
                model=model, tok=tok, ds=val_ds,
                prompt_key="prompt", chosen_key="chosen", rejected_key="rejected",
                device=device, max_len=args.max_seq_length, max_eval=args.max_eval_samples,
            )
            record["eval_pairwise_accuracy"] = round(pair_m["pairwise_accuracy"], 6)
            record["eval_chosen_logp_mean"] = round(pair_m["chosen_logp_mean"], 6)
            record["eval_rejected_logp_mean"] = round(pair_m["rejected_logp_mean"], 6)
            record["eval_n"] = pair_m["eval_n"]
            print(f"  [eval step={step}] acc={pair_m['pairwise_accuracy']:.4f}")
        # Compute eval loss on first N val examples
        eval_loss_sum = 0.0
        eval_n = min(len(val_ds), args.max_eval_samples)
        model.eval()
        with torch.no_grad():
            if ds_type == "dpo":
                val_data = prepare_dpo_data(val_ds, tok, args.max_seq_length, "prompt", "chosen", "rejected")
                for j in range(eval_n):
                    ex = val_data[j]
                    c_lp = response_logprob(model, ex["chosen_ids"].to(device), ex["prompt_len"], pad_id)
                    r_lp = response_logprob(model, ex["rejected_ids"].to(device), ex["prompt_len"], pad_id)
                    eval_loss_sum += (-F.logsigmoid(args.dpo_beta * (c_lp - r_lp))).item()
            else:
                val_data_sft = prepare_sft_data(val_ds, tok, args.max_seq_length, prompt_key, response_key)
                for j in range(eval_n):
                    ex = val_data_sft[j]
                    eval_loss_sum += sft_loss_single(model, ex["input_ids"].to(device), ex["labels"].to(device)).item()
        record["eval_loss"] = round(eval_loss_sum / max(1, eval_n), 6)
        model.train()
        append_jsonl(metrics_path, record)
        return record

    # -----------------------------------------------------------------------
    # 9. Training loop
    # -----------------------------------------------------------------------
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")
    start_time = time.time()
    global_step = 0
    total_examples_seen = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    running_count = 0
    running_clip_count = 0  # examples where grad was clipped

    model.train()
    print(f"[Train] type={ds_type} dp={args.dp} starting... "
          f"(steps={total_steps} batch={logical_batch_size})")

    done = False
    epoch = 0
    while not done:
        rng = random.Random(args.seed + epoch)
        order = list(range(N))
        rng.shuffle(order)

        for batch_start in range(0, N, logical_batch_size):
            if done:
                break

            batch_indices = order[batch_start:batch_start + logical_batch_size]
            actual_bs = len(batch_indices)

            accumulated = {n: torch.zeros_like(p, device=device) for n, p in trainable_params}
            batch_loss = 0.0
            batch_clip_count = 0

            for idx in batch_indices:
                ex = train_data[idx]
                optimizer.zero_grad(set_to_none=True)

                if ds_type == "dpo":
                    loss = dpo_loss_single(
                        model, ex["chosen_ids"].to(device), ex["rejected_ids"].to(device),
                        ex["prompt_len"], ex["ref_chosen_lp"], ex["ref_rejected_lp"],
                        args.dpo_beta, pad_id,
                    )
                else:
                    loss = sft_loss_single(model, ex["input_ids"].to(device), ex["labels"].to(device))

                loss.backward()
                batch_loss += loss.item()

                if args.dp:
                    grad_norm_sq = sum(
                        p.grad.detach().norm(2).item() ** 2
                        for _, p in trainable_params if p.grad is not None
                    )
                    grad_norm = math.sqrt(grad_norm_sq)
                    clip_coeff = min(1.0, C / (grad_norm + 1e-8))
                    if clip_coeff < 1.0:
                        batch_clip_count += 1
                    running_grad_norm += grad_norm

                    for n, p in trainable_params:
                        if p.grad is not None:
                            accumulated[n].add_(p.grad.detach() * clip_coeff)
                else:
                    for n, p in trainable_params:
                        if p.grad is not None:
                            accumulated[n].add_(p.grad.detach())
                    grad_norm = math.sqrt(sum(
                        p.grad.detach().norm(2).item() ** 2
                        for _, p in trainable_params if p.grad is not None
                    ))
                    running_grad_norm += grad_norm

            # Noise + average
            if args.dp:
                for n in accumulated:
                    if noise_generator is not None:
                        noise = torch.randn(accumulated[n].shape, generator=noise_generator,
                                            device=device, dtype=accumulated[n].dtype)
                    else:
                        noise = torch.randn_like(accumulated[n])
                    accumulated[n].add_(noise * sigma * C)
                    accumulated[n].div_(actual_bs)
                accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
            else:
                for n in accumulated:
                    accumulated[n].div_(actual_bs)

            optimizer.zero_grad(set_to_none=True)
            for n, p in trainable_params:
                p.grad = accumulated[n]

            if not args.dp:
                torch.nn.utils.clip_grad_norm_([p for _, p in trainable_params], args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            global_step += 1
            total_examples_seen += actual_bs
            running_loss += batch_loss
            running_count += actual_bs
            running_clip_count += batch_clip_count

            # --- Per-step logging ---
            if global_step % args.logging_steps == 0:
                avg_loss = running_loss / max(1, running_count)
                avg_grad = running_grad_norm / max(1, running_count)
                clip_frac = running_clip_count / max(1, running_count) if args.dp else None
                lr_now = scheduler.get_last_lr()[0]
                epoch_f = epoch + batch_start / N

                record: Dict[str, Any] = {
                    "step": global_step,
                    "epoch": round(epoch_f, 4),
                    "loss": round(avg_loss, 6),
                    "grad_norm": round(avg_grad, 6),
                    "lr": lr_now,
                    "examples_seen": total_examples_seen,
                    "effective_batch_size": effective_batch_size,
                }
                if args.dp and accountant is not None:
                    eps_spent = accountant.get_epsilon(delta=delta)
                    record["epsilon_spent"] = round(eps_spent, 6)
                    record["noise_multiplier"] = sigma
                    record["clipping_fraction"] = round(clip_frac, 6)
                    print(f"  [step={global_step}] loss={avg_loss:.4f} eps={eps_spent:.4f} "
                          f"clip={clip_frac:.3f} grad={avg_grad:.4f} lr={lr_now:.2e}")
                else:
                    print(f"  [step={global_step}] loss={avg_loss:.4f} "
                          f"grad={avg_grad:.4f} lr={lr_now:.2e}")

                append_jsonl(metrics_path, record)
                running_loss = 0.0
                running_grad_norm = 0.0
                running_count = 0
                running_clip_count = 0

            # --- Periodic eval ---
            if args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0:
                epoch_f = epoch + batch_start / N
                run_eval_and_log(global_step, epoch_f)
                model.train()

            if args.max_steps > 0 and global_step >= args.max_steps:
                done = True
                break

        epoch += 1
        if args.max_steps <= 0 and epoch >= args.num_train_epochs:
            done = True

    train_time = time.time() - start_time
    print(f"[Train] done in {train_time:.1f}s  steps={global_step}")

    final_eps = None
    if args.dp and accountant is not None:
        final_eps = accountant.get_epsilon(delta=delta)
        print(f"[DP] Final epsilon: {final_eps:.6f} (delta={delta})")

    # -----------------------------------------------------------------------
    # 10. Save adapter
    # -----------------------------------------------------------------------
    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"[Save] adapter -> {args.output_dir}")

    # -----------------------------------------------------------------------
    # 11. Final evaluation
    # -----------------------------------------------------------------------
    final_eval = run_eval_and_log(global_step, float(epoch))

    summary: Dict[str, Any] = {
        "train_time_seconds": train_time,
        "dataset_type": ds_type,
        "train_n": len(train_ds),
        "val_n": len(val_ds),
        "dp_enabled": args.dp,
        "epsilon_spent": final_eps,
        "noise_multiplier": sigma if args.dp else None,
        "max_grad_norm": C,
        "delta": delta,
        "total_steps": global_step,
        "seed": args.seed,
        "model_id": args.model_id,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "num_train_epochs": args.num_train_epochs,
        "trainable_params": trainable_count,
    }
    # Merge final eval metrics
    for k, v in final_eval.items():
        if k not in ("step", "epoch", "is_eval"):
            summary[k] = v

    # Generation eval
    if args.eval_prompts_json:
        with open(args.eval_prompts_json, "r") as f:
            eval_prompts = json.load(f)
    else:
        eval_prompts = DEFAULT_EVAL_PROMPTS

    print(f"[Eval] generating on {len(eval_prompts)} prompts...")
    generations = generate_samples(model=model, tok=tok, prompts=eval_prompts,
                                   device=device, max_new_tokens=args.gen_max_new_tokens)
    gen_path = os.path.join(args.output_dir, "generations.jsonl")
    with open(gen_path, "w", encoding="utf-8") as f:
        for g in generations:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    print(f"[Eval] generations -> {gen_path}")

    write_json(os.path.join(args.output_dir, "summary.json"), summary)

    # -----------------------------------------------------------------------
    # 12. Convergence summary + plot
    # -----------------------------------------------------------------------
    all_metrics = read_jsonl(metrics_path)
    metric_key = "eval_pairwise_accuracy" if ds_type == "dpo" else "eval_loss"
    conv_summary = compute_convergence_summary(all_metrics, metric_key)
    conv_summary["seed"] = args.seed
    write_json(os.path.join(args.output_dir, "convergence_summary.json"), conv_summary)
    print(f"[Conv] convergence_summary.json written")

    plot_convergence(all_metrics, os.path.join(args.output_dir, "eval_metric_vs_steps.png"), metric_key)

    print("=" * 60)
    print("Summary")
    print("=" * 60)
    for k, v in sorted(summary.items()):
        print(f"  {k}: {v}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
