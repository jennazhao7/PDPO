#!/usr/bin/env python3
"""
Step-1 Oracle Soft-Bayes (stacked architecture sanity test).

Purpose:
- Keep the original stacked Stage-2 architecture from train_stage2_soft_bayes.py.
- Replace M1-derived deltas with oracle deltas from known RR flip indicators.
- Verify that SB weighting machinery can exploit near-perfect weights.

This script does NOT overwrite any existing training logic scripts.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sys
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

try:
    from stage2_debugging.train_stage2_soft_bayes import (
        SoftBayesPairDataset,
        collate_soft_bayes,
        compute_posteriors,
        compute_response_logprobs,
        find_first_key,
        load_causal_lm_compat,
        score_sequence,
        soft_bayes_loss,
    )
except ModuleNotFoundError:
    # Fallback for direct execution from within stage2_debugging/
    from train_stage2_soft_bayes import (  # type: ignore
        SoftBayesPairDataset,
        collate_soft_bayes,
        compute_posteriors,
        compute_response_logprobs,
        find_first_key,
        load_causal_lm_compat,
        score_sequence,
        soft_bayes_loss,
    )


DEFAULT_DATA = "lora/preprocessing/d2_rr_flipped.jsonl"


def _path_slug(s: str) -> str:
    return s.replace("/", "--").replace("\\", "--").strip("-")


def _row_key(prompt: str, chosen: str, rejected: str) -> str:
    raw = f"{prompt}\n{chosen}\n{rejected}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _row_key_unordered(prompt: str, chosen: str, rejected: str) -> str:
    a, b = sorted([chosen, rejected])
    raw = f"{prompt}\n{a}\n{b}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Step-1 Oracle Soft-Bayes (stacked)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage1_adapter", required=True)
    ap.add_argument("--stage1_subfolder", default=None)
    ap.add_argument("--data", default=DEFAULT_DATA, help="Noisy D2 JSONL used for training")
    ap.add_argument(
        "--oracle_labels",
        required=True,
        help="JSONL with per-example oracle flip info (expects 'flipped' bool).",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--manifest_out", default=None)
    ap.add_argument("--output_suffix", default="")

    ap.add_argument("--prompt_key", default="prompt")
    ap.add_argument("--chosen_key", default="chosen")
    ap.add_argument("--rejected_key", default="rejected")

    ap.add_argument("--epsilon", type=float, default=1.0)
    ap.add_argument("--p_keep", type=float, default=None)

    # Oracle controls
    ap.add_argument("--oracle_delta", type=float, default=10.0, help="Use +/- this value for oracle delta.")
    ap.add_argument("--tau", type=float, default=1.0, help="Temperature for sigmoid(delta/tau).")
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
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,query_key_value,dense,dense_h_to_4h,dense_4h_to_h",
    )
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()


def load_oracle_flip_map(path: str) -> tuple[Dict[str, bool], Dict[str, bool]]:
    rows = [json.loads(x) for x in open(path, "r", encoding="utf-8") if x.strip()]
    flip_map_ordered: Dict[str, bool] = {}
    flip_map_unordered: Dict[str, bool] = {}
    for i, r in enumerate(rows):
        if "flipped" not in r:
            raise ValueError(f"oracle_labels row {i} missing 'flipped' field")
        k_ord = _row_key(r["prompt"], r["chosen"], r["rejected"])
        k_unord = _row_key_unordered(r["prompt"], r["chosen"], r["rejected"])
        flip_map_ordered[k_ord] = bool(r["flipped"])
        flip_map_unordered[k_unord] = bool(r["flipped"])
    if not flip_map_ordered:
        raise ValueError(f"oracle_labels empty: {path}")
    return flip_map_ordered, flip_map_unordered


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    if args.tau <= 0:
        raise ValueError("For oracle Step-1, set --tau > 0 (recommended 1.0).")

    if args.out is None:
        model_slug = _path_slug(os.path.basename(args.model.rstrip("/")))
        stage1_slug = _path_slug(os.path.basename(args.stage1_adapter.rstrip("/")))
        suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        args.out = f"outputs/stage2_oracle_soft_bayes_{model_slug}_{stage1_slug}{suffix}"
    if args.manifest_out is None:
        args.manifest_out = os.path.join(args.out, "M2_manifest.json")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest_out) or ".", exist_ok=True)
    torch.manual_seed(args.seed)

    p_keep = args.p_keep if args.p_keep is not None else math.exp(args.epsilon) / (math.exp(args.epsilon) + 1.0)
    print(f"[Config] out={args.out}")
    print(f"[Config] epsilon={args.epsilon} p_keep={p_keep:.6f} tau={args.tau} beta={args.beta} oracle_delta={args.oracle_delta}")

    raw = load_dataset("json", data_files=args.data)["train"]
    cols = raw.column_names
    pk = args.prompt_key if args.prompt_key in cols else find_first_key(cols, ["prompt", "question", "instruction"])
    ck = args.chosen_key if args.chosen_key in cols else find_first_key(cols, ["chosen", "answer", "response"])
    rk = args.rejected_key if args.rejected_key in cols else find_first_key(cols, ["rejected"])
    if not pk or not ck or not rk:
        raise ValueError(f"Cannot find prompt/chosen/rejected in columns: {cols}")
    prompts, chosen, rejected = raw[pk], raw[ck], raw[rk]
    N = len(prompts)
    print(f"[Data] N={N} from {args.data}")

    flip_map_ordered, flip_map_unordered = load_oracle_flip_map(args.oracle_labels)
    print(f"[Oracle] loaded {len(flip_map_ordered)} labeled rows from {args.oracle_labels}")

    use_fast = False if "open_llama" in args.model.lower() else True
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=use_fast)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_ok else "cpu")
    dtype = torch.float32 if not cuda_ok else (torch.bfloat16 if args.bf16 else torch.float16)

    print("[M1] Loading base + stage1 for reference logprobs...")
    base_model = load_causal_lm_compat(args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None)
    base_model.config.use_cache = True
    if args.stage1_subfolder:
        m1 = PeftModel.from_pretrained(base_model, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1")
    else:
        m1 = PeftModel.from_pretrained(base_model, args.stage1_adapter, adapter_name="stage1")
    m1.set_adapter("stage1")
    m1 = m1.to(device).eval()

    # Oracle deltas and stage1 reference logprobs.
    deltas: List[float] = []
    ref_chosen_lps: List[float] = []
    ref_rejected_lps: List[float] = []
    misses = 0
    unordered_hits = 0
    flipped_ct = 0
    for i in range(N):
        key_ord = _row_key(prompts[i], chosen[i], rejected[i])
        if key_ord in flip_map_ordered:
            is_flipped = flip_map_ordered[key_ord]
        else:
            # RR datasets may swap chosen/rejected relative to oracle file.
            # Use unordered pair key as fallback; flip indicator remains valid.
            key_unord = _row_key_unordered(prompts[i], chosen[i], rejected[i])
            if key_unord in flip_map_unordered:
                is_flipped = flip_map_unordered[key_unord]
                unordered_hits += 1
            else:
                misses += 1
                is_flipped = False
        if is_flipped:
            flipped_ct += 1
        deltas.append(-args.oracle_delta if is_flipped else args.oracle_delta)

        m1.set_adapter("stage1")
        s_c = score_sequence(m1, tok, prompts[i], chosen[i], device, args.max_len)
        s_r = score_sequence(m1, tok, prompts[i], rejected[i], device, args.max_len)
        ref_chosen_lps.append(s_c)
        ref_rejected_lps.append(s_r)
        if i < 5 or (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N}] flipped={is_flipped} delta={deltas[-1]:.2f} ref_c={s_c:.4f} ref_r={s_r:.4f}")

    print(f"[Oracle] key match stats: ordered={N-unordered_hits-misses}, unordered={unordered_hits}, misses={misses}")
    if misses > 0:
        raise RuntimeError(
            f"Oracle mapping coverage failed: {misses}/{N} rows in --data not found in --oracle_labels. "
            "Use matching D2+labels files generated from the same run."
        )

    weights = compute_posteriors(
        deltas,
        p_keep=p_keep,
        tau=args.tau,
        delta_clamp=args.delta_clamp,
        w_lo=args.w_clamp_lo,
        w_hi=args.w_clamp_hi,
    )
    w_mean = sum(weights) / len(weights)
    w_gt_half = sum(1 for w in weights if w > 0.5)
    w_lo_ct = sum(1 for w in weights if w < 0.1)
    w_hi_ct = sum(1 for w in weights if w > 0.9)
    print(f"[Oracle] flipped={flipped_ct}/{N} ({flipped_ct/max(1,N):.3f})")
    print(f"[Oracle] weights: mean={w_mean:.4f} >0.5={w_gt_half}/{N} <0.1={w_lo_ct}/{N} >0.9={w_hi_ct}/{N}")

    with open(os.path.join(args.out, "scoring_stats.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "N": N,
                "method": "oracle_soft_bayes_stacked",
                "oracle_labels": os.path.abspath(args.oracle_labels),
                "oracle_delta": args.oracle_delta,
                "tau": args.tau,
                "epsilon": args.epsilon,
                "p_keep": p_keep,
                "flipped_count": flipped_ct,
                "weights_mean": w_mean,
                "weights_lt_0p1": w_lo_ct,
                "weights_gt_0p9": w_hi_ct,
            },
            f,
            indent=2,
        )

    del m1
    del base_model
    gc.collect()
    if cuda_ok:
        torch.cuda.empty_cache()

    pair_ds = SoftBayesPairDataset(prompts, chosen, rejected, weights, ref_chosen_lps, ref_rejected_lps, tok, args.max_len)
    train_dl = DataLoader(pair_ds, batch_size=args.bsz, shuffle=True, collate_fn=lambda b: collate_soft_bayes(b, pad_id))

    # Original stacked architecture: base + merged stage1 + trainable stage2.
    print("[M2] Loading stacked setup: base + stage1(merged) + stage2(trainable)")
    base_model2 = load_causal_lm_compat(args.model, torch_dtype=dtype, low_cpu_mem_usage=True, device_map=None)
    base_model2.config.use_cache = False
    base_model2.gradient_checkpointing_enable()
    if args.stage1_subfolder:
        m2 = PeftModel.from_pretrained(base_model2, args.stage1_adapter, subfolder=args.stage1_subfolder, adapter_name="stage1")
    else:
        m2 = PeftModel.from_pretrained(base_model2, args.stage1_adapter, adapter_name="stage1")
    m2 = m2.merge_and_unload()
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    stage2_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    m2 = get_peft_model(m2, stage2_cfg, adapter_name="stage2")
    m2 = m2.to(device).train()

    trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    total = sum(p.numel() for p in m2.parameters())
    print(f"[M2] trainable={trainable} total={total} ({100*trainable/max(1,total):.4f}%)")

    optimizer = torch.optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=args.lr, eps=1e-6)
    global_step = 0
    accum_loss = 0.0
    max_steps = args.max_steps if args.max_steps > 0 else None
    ckpt_dirs: List[str] = []

    def save_ckpt(step: int) -> None:
        d = os.path.join(args.out, f"checkpoint-step-{step}")
        m2.save_pretrained(d, adapter_name="stage2")
        ckpt_dirs.append(d)
        print(f"[Checkpoint] saved {d}")
        while len(ckpt_dirs) > max(1, args.save_total_limit):
            old = ckpt_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"[Checkpoint] pruned {old}")

    for epoch in range(args.epochs):
        for bidx, batch in enumerate(train_dl):
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

            if (bidx + 1) % args.ga == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in m2.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 10 == 0 or global_step <= 3:
                    print(f"  [step={global_step}] loss={accum_loss:.6f} epoch={epoch}")
                accum_loss = 0.0
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_ckpt(global_step)
                if max_steps and global_step >= max_steps:
                    break

        if len(train_dl) % args.ga != 0 and accum_loss != 0.0 and (not max_steps or global_step < max_steps):
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for p in m2.parameters() if p.requires_grad], args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            print(f"  [step={global_step}] loss={accum_loss:.6f} epoch={epoch} (flush)")
            accum_loss = 0.0
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_ckpt(global_step)
        if max_steps and global_step >= max_steps:
            break
        print(f"[Epoch {epoch+1}/{args.epochs}] completed")

    m2.save_pretrained(args.out, adapter_name="stage2")
    manifest = {
        "base_model": args.model,
        "adapters": [
            {"name": "stage1", "path": os.path.abspath(args.stage1_adapter)},
            {"name": "stage2", "path": os.path.abspath(args.out)},
        ],
        "adapter_order": ["stage1", "stage2"],
        "method": "oracle_soft_bayes_stacked",
        "params": {
            "epsilon": args.epsilon,
            "p_keep": p_keep,
            "tau": args.tau,
            "beta": args.beta,
            "oracle_delta": args.oracle_delta,
            "oracle_labels": os.path.abspath(args.oracle_labels),
        },
        "notes": "Step-1 oracle SB sanity test: stacked architecture unchanged; only delta source replaced by oracle flips.",
    }
    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Done] Saved oracle SB adapter to {args.out}")
    print(f"[Done] Manifest -> {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

