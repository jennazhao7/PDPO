#!/usr/bin/env python3
"""
Stage-2 SB Fresh-Retrain with Less-is-More selection.

Key additions over SB-Fresh:
  1) Selection: keep pairs where |q_i - 0.5| >= tau_drop
  2) Optional ablations:
     - weighting-only: tau_drop=0
     - selection-only: --disable_weighting
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_DATA = "stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps1.0_seed42.jsonl"


def load_causal_lm_compat(model_id: str, **kwargs):
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, use_safetensors=True, **kwargs)
    except Exception as safe_exc:
        original_import_check = None
        original_modeling_check = None
        try:
            import transformers.modeling_utils as modeling_utils
            import transformers.utils.import_utils as import_utils

            original_import_check = import_utils.check_torch_load_is_safe
            original_modeling_check = modeling_utils.check_torch_load_is_safe
            import_utils.check_torch_load_is_safe = lambda: None
            modeling_utils.check_torch_load_is_safe = lambda: None
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            print(f"[Load] safetensors unavailable ({safe_exc}); loaded via compatibility fallback.")
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


def path_slug(s: str) -> str:
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
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    response_ids = tok.encode(response, add_special_tokens=False)
    input_ids = (prompt_ids + response_ids)[:max_len]
    input_ids = torch.tensor([input_ids], device=device)

    outputs = model(input_ids=input_ids)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    prompt_len = len(prompt_ids)
    if prompt_len >= input_ids.shape[1]:
        return 0.0
    response_log_probs = token_log_probs[0, prompt_len - 1 :]
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
    m1.set_adapter("stage1")
    s_m1 = score_sequence(m1, tok, prompt, response, device, max_len)
    with m1.disable_adapter():
        s_base = score_sequence(m1, tok, prompt, response, device, max_len)
    return s_m1 - s_base


def compute_q_and_w(
    deltas: List[float],
    p_keep: float,
    tau: float,
    delta_clamp: float,
    w_lo: float,
    w_hi: float,
) -> tuple[List[float], List[float]]:
    q_list: List[float] = []
    w_list: List[float] = []
    for d in deltas:
        scaled = max(-delta_clamp, min(delta_clamp, d / tau))
        q_i = 1.0 / (1.0 + math.exp(-scaled))
        num = p_keep * q_i
        den = p_keep * q_i + (1.0 - p_keep) * (1.0 - q_i)
        w_i = num / den if den >= 1e-12 else 0.5
        w_i = max(w_lo, min(w_hi, w_i))
        q_list.append(q_i)
        w_list.append(w_i)
    return q_list, w_list


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class PairDataset(TorchDataset):
    def __init__(
        self,
        prompts: List[str],
        chosen: List[str],
        rejected: List[str],
        weights: List[float],
        ref_chosen_lps: List[float],
        ref_rejected_lps: List[float],
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
    return torch.tensor([s + [pad_id] * (max_len - len(s)) for s in sequences], dtype=torch.long)


def collate_fn(batch: List[Dict], pad_id: int) -> Dict:
    return {
        "chosen_ids": pad_to_max([b["chosen_ids"] for b in batch], pad_id),
        "rejected_ids": pad_to_max([b["rejected_ids"] for b in batch], pad_id),
        "prompt_lens": torch.tensor([b["prompt_len"] for b in batch], dtype=torch.long),
        "w": torch.tensor([b["w"] for b in batch], dtype=torch.float32),
        "ref_chosen_lp": torch.tensor([b["ref_chosen_lp"] for b in batch], dtype=torch.float32),
        "ref_rejected_lp": torch.tensor([b["ref_rejected_lp"] for b in batch], dtype=torch.float32),
    }


def compute_response_logprobs(model: nn.Module, input_ids: torch.Tensor, prompt_lens: torch.Tensor, pad_id: int) -> torch.Tensor:
    outputs = model(input_ids=input_ids)
    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    bsz, seq_m1 = token_log_probs.shape
    positions = torch.arange(seq_m1, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
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
    m = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
    return -(w * F.logsigmoid(beta * m) + (1.0 - w) * F.logsigmoid(-beta * m)).mean()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SB-Fresh with preference-data selection")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage1_adapter", required=True)
    ap.add_argument("--stage1_subfolder", default=None)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest_out", default=None)
    ap.add_argument("--output_suffix", default="")
    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--chosen_key", default="chosen")
    ap.add_argument("--rejected_key", default="rejected")
    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--p_keep", type=float, default=None)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--tau_drop", type=float, default=0.10, help="selection threshold on |q_i - 0.5|")
    ap.add_argument(
        "--keep_fraction",
        type=float,
        default=None,
        help="If set, keep the top fraction of pairs by |q_i - 0.5|; overrides tau_drop selection.",
    )
    ap.add_argument("--min_keep", type=int, default=64, help="minimum kept pairs after selection")
    ap.add_argument(
        "--selection_mode",
        choices=["both", "selection_only", "weighting_only"],
        default="both",
        help="Ablation mode: both=selection+SB weights, selection_only=selected pairs with unit weights, weighting_only=all pairs with SB weights.",
    )
    ap.add_argument("--disable_weighting", action="store_true", help="Deprecated alias for --selection_mode selection_only")
    ap.add_argument("--score_cache_in", default=None, help="Optional torch cache with deltas/ref logprobs from prior M1 scoring.")
    ap.add_argument("--score_cache_out", default=None, help="Optional path to save deltas/ref logprobs after M1 scoring.")
    ap.add_argument("--score_only", action="store_true", help="Only score D2 and save --score_cache_out; do not train M2.")
    ap.add_argument("--save_selection_jsonl", default=None, help="Optional per-entry audit JSONL with q/w/keep decisions.")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--w_clamp_lo", type=float, default=0.01)
    ap.add_argument("--w_clamp_hi", type=float, default=0.99)
    ap.add_argument("--delta_clamp", type=float, default=500.0)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--ga", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--save_steps", type=int, default=100)
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
    )
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.disable_weighting:
        args.selection_mode = "selection_only"
    if args.keep_fraction is not None and not (0.0 < args.keep_fraction <= 1.0):
        raise ValueError(f"--keep_fraction must be in (0, 1], got {args.keep_fraction}")
    if args.score_only and not args.score_cache_out:
        raise ValueError("--score_only requires --score_cache_out")
    if args.out is None:
        model_slug = path_slug(os.path.basename(args.model.rstrip("/")))
        s1_slug = path_slug(os.path.basename(args.stage1_adapter.rstrip("/")))
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        args.out = f"stage2_debugging/newplans/less_is_more_dp/results/sb_select_weight_{model_slug}_{s1_slug}{suffix}"
    if args.manifest_out is None:
        args.manifest_out = os.path.join(args.out, "M2_manifest.json")
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    p_keep = args.p_keep if args.p_keep is not None else math.exp(args.epsilon) / (math.exp(args.epsilon) + 1.0)
    print(
        f"[Config] eps={args.epsilon} p_keep={p_keep:.6f} tau={args.tau} "
        f"tau_drop={args.tau_drop} keep_fraction={args.keep_fraction} mode={args.selection_mode}"
    )

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
    n_all = len(prompts)
    print(f"[Data] loaded {n_all} pairs from {args.data}")

    use_fast = False if "open_llama" in args.model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    dtype = torch.bfloat16 if (cuda_ok and args.bf16) else (torch.float16 if cuda_ok else torch.float32)

    if args.score_cache_in:
        print(f"[Cache] loading score cache: {args.score_cache_in}")
        cache = torch.load(args.score_cache_in, map_location="cpu")
        deltas = [float(x) for x in cache["deltas"]]
        ref_c = [float(x) for x in cache["ref_chosen_lps"]]
        ref_r = [float(x) for x in cache["ref_rejected_lps"]]
        if len(deltas) != n_all or len(ref_c) != n_all or len(ref_r) != n_all:
            raise ValueError(
                "Score cache length mismatch: "
                f"cache={len(deltas)}, data={n_all}. cache={args.score_cache_in}, data={args.data}"
            )
    else:
        print("[M1] loading base+stage1 for scoring...")
        base_model = load_causal_lm_compat(args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None)
        if args.stage1_subfolder:
            m1 = PeftModel.from_pretrained(
                base_model, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1"
            )
        else:
            m1 = PeftModel.from_pretrained(base_model, args.stage1_adapter, adapter_name="stage1")
        m1.set_adapter("stage1")
        m1 = m1.to(device).eval()

        deltas: List[float] = []
        ref_c: List[float] = []
        ref_r: List[float] = []
        for i in range(n_all):
            dr_c = dpo_reward_for_sequence(m1, tok, prompts[i], chosen[i], device, args.max_len)
            dr_r = dpo_reward_for_sequence(m1, tok, prompts[i], rejected[i], device, args.max_len)
            deltas.append(dr_c - dr_r)
            with m1.disable_adapter():
                ref_c.append(score_sequence(m1, tok, prompts[i], chosen[i], device, args.max_len))
                ref_r.append(score_sequence(m1, tok, prompts[i], rejected[i], device, args.max_len))
            if i < 3 or (i + 1) % 200 == 0:
                print(f"  [score {i+1}/{n_all}] delta={deltas[-1]:.4f}")

        if args.score_cache_out:
            os.makedirs(os.path.dirname(args.score_cache_out) or ".", exist_ok=True)
            torch.save(
                {
                    "deltas": deltas,
                    "ref_chosen_lps": ref_c,
                    "ref_rejected_lps": ref_r,
                    "data": os.path.abspath(args.data),
                    "stage1_adapter": os.path.abspath(args.stage1_adapter),
                    "base_model": args.model,
                    "epsilon": args.epsilon,
                    "max_len": args.max_len,
                    "n": n_all,
                },
                args.score_cache_out,
            )
            print(f"[Cache] saved score cache: {args.score_cache_out}")

        del m1, base_model
        gc.collect()
        if cuda_ok:
            torch.cuda.empty_cache()

    if max(abs(x) for x in deltas) < 1e-8:
        raise RuntimeError("All deltas are ~0. Stage1 scoring appears broken.")

    if args.tau <= 0:
        import numpy as np

        std_delta = float(np.std(deltas))
        args.tau = max(0.5, std_delta / 1.2)
        print(f"[Auto-tau] std_delta={std_delta:.4f} -> tau={args.tau:.4f}")

    q_list, w_list = compute_q_and_w(deltas, p_keep, args.tau, args.delta_clamp, args.w_clamp_lo, args.w_clamp_hi)

    if args.score_only:
        print("[Done] score_only requested; exiting before selection/training.")
        return 0

    # Selection by confidence distance from 0.5
    conf = [abs(q - 0.5) for q in q_list]
    selection_rule = "tau_drop"
    if args.selection_mode == "weighting_only":
        keep_mask = [True for _ in conf]
        selection_rule = "weighting_only_all_pairs"
    elif args.keep_fraction is not None:
        target_keep = max(1, min(n_all, int(round(n_all * args.keep_fraction))))
        order = sorted(range(n_all), key=lambda i: (-conf[i], i))
        keep_set = set(order[:target_keep])
        keep_mask = [i in keep_set for i in range(n_all)]
        selection_rule = "top_keep_fraction"
    else:
        keep_mask = [c >= args.tau_drop for c in conf]
    n_keep = sum(1 for k in keep_mask if k)

    # Safety fallback: keep top min_keep most confident pairs
    if args.selection_mode != "weighting_only" and n_keep < min(args.min_keep, n_all):
        order = sorted(range(n_all), key=lambda i: (-conf[i], i))
        keep_set = set(order[: min(args.min_keep, n_all)])
        keep_mask = [i in keep_set for i in range(n_all)]
        n_keep = sum(1 for k in keep_mask if k)
        selection_rule = f"{selection_rule}_min_keep_fallback"
        print(f"[Select] fallback activated, keeping top-{n_keep} by |q-0.5|")

    if n_keep <= 0:
        raise RuntimeError("Selection kept zero pairs.")

    sel_prompts = [prompts[i] for i in range(n_all) if keep_mask[i]]
    sel_chosen = [chosen[i] for i in range(n_all) if keep_mask[i]]
    sel_rejected = [rejected[i] for i in range(n_all) if keep_mask[i]]
    sel_ref_c = [ref_c[i] for i in range(n_all) if keep_mask[i]]
    sel_ref_r = [ref_r[i] for i in range(n_all) if keep_mask[i]]
    sel_w = [w_list[i] for i in range(n_all) if keep_mask[i]]
    sel_q = [q_list[i] for i in range(n_all) if keep_mask[i]]

    if args.selection_mode == "selection_only":
        sel_w = [1.0 for _ in sel_w]
        mode = "selection_only"
    elif args.selection_mode == "weighting_only":
        mode = "weighting_only"
    else:
        mode = "selection_plus_weighting"

    audit_rows = [
        {
            "idx": i,
            "delta": deltas[i],
            "q": q_list[i],
            "w": 1.0 if args.selection_mode == "selection_only" else w_list[i],
            "confidence": conf[i],
            "kept": bool(keep_mask[i]),
        }
        for i in range(n_all)
    ]
    if args.save_selection_jsonl:
        save_jsonl(args.save_selection_jsonl, audit_rows)
        print(f"[Audit] saved selection audit: {args.save_selection_jsonl}")

    sel_stats = {
        "mode": mode,
        "N_total": n_all,
        "N_kept": n_keep,
        "keep_fraction": n_keep / max(1, n_all),
        "requested_keep_fraction": args.keep_fraction,
        "selection_rule": selection_rule,
        "tau_drop": args.tau_drop,
        "tau": args.tau,
        "epsilon": args.epsilon,
        "p_keep": p_keep,
        "q_mean_all": sum(q_list) / len(q_list),
        "w_mean_all": sum(w_list) / len(w_list),
        "q_mean_kept": sum(sel_q) / len(sel_q),
        "w_mean_kept": sum(sel_w) / len(sel_w),
        "score_cache_in": args.score_cache_in,
        "score_cache_out": args.score_cache_out,
        "selection_jsonl": args.save_selection_jsonl,
    }
    with open(os.path.join(args.out, "selection_stats.json"), "w", encoding="utf-8") as f:
        json.dump(sel_stats, f, indent=2)
    print(f"[Select] kept {n_keep}/{n_all} ({100.0*n_keep/max(1,n_all):.2f}%)")

    pair_ds = PairDataset(sel_prompts, sel_chosen, sel_rejected, sel_w, sel_ref_c, sel_ref_r, tok, args.max_len)
    train_dl = DataLoader(pair_ds, batch_size=args.bsz, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))

    print("[M2] loading fresh base + stage2 LoRA...")
    base_model2 = load_causal_lm_compat(args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None)
    base_model2.config.use_cache = False
    base_model2.gradient_checkpointing_enable()
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    m2 = get_peft_model(base_model2, lora_cfg, adapter_name="stage2")
    m2 = m2.to(device).train()

    optim = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=args.lr, eps=1e-6)
    global_step = 0
    accum_loss = 0.0
    max_steps = args.max_steps if args.max_steps > 0 else None
    ckpt_dirs: List[str] = []

    def save_ckpt(step: int) -> None:
        ckpt = os.path.join(args.out, f"checkpoint-step-{step}")
        m2.save_pretrained(ckpt, adapter_name="stage2")
        ckpt_dirs.append(ckpt)
        while len(ckpt_dirs) > max(1, args.save_total_limit):
            old = ckpt_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_dl):
            c_ids = batch["chosen_ids"].to(device)
            r_ids = batch["rejected_ids"].to(device)
            p_lens = batch["prompt_lens"].to(device)
            w = batch["w"].to(device)
            ref_ch = batch["ref_chosen_lp"].to(device)
            ref_rj = batch["ref_rejected_lp"].to(device)

            pi_c = compute_response_logprobs(m2, c_ids, p_lens, pad_id)
            pi_r = compute_response_logprobs(m2, r_ids, p_lens, pad_id)
            loss = soft_bayes_loss(pi_c, pi_r, ref_ch, ref_rj, w, beta=args.beta) / args.ga
            loss.backward()
            accum_loss += loss.item()

            if (batch_idx + 1) % args.ga == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in m2.parameters() if p.requires_grad], args.max_grad_norm)
                optim.step()
                optim.zero_grad()
                global_step += 1
                if global_step <= 3 or global_step % 10 == 0:
                    print(f"[Train] step={global_step} loss={accum_loss:.6f} epoch={epoch}")
                accum_loss = 0.0
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_ckpt(global_step)
                if max_steps and global_step >= max_steps:
                    break
        if max_steps and global_step >= max_steps:
            break

    m2.save_pretrained(args.out, adapter_name="stage2")
    manifest = {
        "base_model": args.model,
        "adapters": [{"name": "stage2", "path": os.path.abspath(args.out)}],
        "method": "sb_select_weight_fresh",
        "params": {
            "epsilon": args.epsilon,
            "p_keep": p_keep,
            "tau": args.tau,
            "tau_drop": args.tau_drop,
            "keep_fraction": args.keep_fraction,
            "selection_rule": selection_rule,
            "beta": args.beta,
            "selection_mode": args.selection_mode,
            "disable_weighting": bool(args.selection_mode == "selection_only"),
            "ref_at_training": "base",
            "stage1_used_for": "scoring_only",
            "stage1_adapter": args.stage1_adapter,
            "score_cache_in": args.score_cache_in,
            "selection_jsonl": args.save_selection_jsonl,
        },
    }
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Done] saved adapter to {args.out}")
    print(f"[Done] wrote manifest to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

