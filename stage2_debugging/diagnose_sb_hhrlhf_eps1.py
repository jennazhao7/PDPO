#!/usr/bin/env python3
"""
Soft-Bayes Diagnostic: HH-RLHF, eps=1.0, seed=42
==================================================
Runs Steps A-D in order on a single GPU, writing all outputs to
  stage2_debugging/diag_hhrlhf_eps1/

Step A  — M1 accuracy on clean held-out test set
Step B  — Raw delta (DPO reward margin) distribution on D2
Step C  — SB weight distribution under auto-tau AND fixed tau sweep
Step D  — Sign convention check: do flipped pairs get LOW w_i?

Runtime: ~30-45 min on 1 GPU for N_D2=10k pairs (use --n_d2 500 for a
         quick 5-min smoke test).

Usage:
    python diagnose_sb_hhrlhf_eps1.py [--n_d2 N] [--n_test N] [--quick]
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Paths (all relative to repo root; run from repo root or stage2_debugging/) ──
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

BASE_MODEL      = "Qwen/Qwen2.5-3B"
M1_ADAPTER      = os.path.join(REPO_ROOT, "stage1/results_eps1.0_seed42/hhrlhf_eps1.0_s42")
D2_JSONL        = os.path.join(REPO_ROOT, "hhrlhf_d2_eps1.0.jsonl")       # RR-flipped D2
ORIG_JSONL      = os.path.join(REPO_ROOT, "hhrlhf_train_eps1.0.jsonl")    # pre-flip D1 (same partition)
TEST_JSONL      = os.path.join(REPO_ROOT, "testsets/hhrlhf_test_pref_clean.jsonl")
OUT_DIR         = os.path.join(REPO_ROOT, "diag_hhrlhf_eps1")

EPSILON         = 1.0
P_KEEP          = math.exp(EPSILON) / (math.exp(EPSILON) + 1.0)   # ≈ 0.7311
MAX_LEN         = 512
TAU_SWEEP       = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]       # fixed taus to test in Step C


# ──────────────────────────────────────────────────────────────────────────────
# Utility: token log-prob sum over response tokens
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_sequence(model, tok, prompt, response, device, max_len=MAX_LEN):
    """Sum of log P(response_token | prompt, preceding) — same as SB training."""
    prompt_ids   = tok.encode(prompt,   add_special_tokens=True)
    response_ids = tok.encode(response, add_special_tokens=False)
    input_ids    = (prompt_ids + response_ids)[:max_len]
    x = torch.tensor([input_ids], device=device)

    out = model(input_ids=x)
    log_probs = F.log_softmax(out.logits[:, :-1, :], dim=-1)
    targets   = x[:, 1:]
    tok_lp    = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)[0]

    p_len = len(prompt_ids)
    if p_len >= x.shape[1]:
        return 0.0
    return float(tok_lp[p_len - 1:].sum().item())


@torch.no_grad()
def dpo_reward(m1, tok, prompt, response, device):
    """log π_M1(y|x) - log π_base(y|x) — cancels length bias."""
    m1.set_adapter("stage1")
    s_m1 = score_sequence(m1, tok, prompt, response, device)
    with m1.disable_adapter():
        s_base = score_sequence(m1, tok, prompt, response, device)
    return s_m1 - s_base


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x); return 1.0 / (1.0 + z)
    z = math.exp(x);  return z / (1.0 + z)


def compute_weights(deltas, tau, p_keep, w_lo=0.01, w_hi=0.99):
    weights = []
    for d in deltas:
        q_i = sigmoid(d / tau)
        num = p_keep * q_i
        den = num + (1.0 - p_keep) * (1.0 - q_i)
        w_i = num / den if den > 1e-12 else 0.5
        weights.append(max(w_lo, min(w_hi, w_i)))
    return weights


def weight_stats(weights, label):
    w = np.array(weights)
    lo  = (w < 0.3).sum()
    hi  = (w > 0.7).sum()
    mid = ((w >= 0.4) & (w <= 0.6)).sum()
    print(f"  [{label}] mean={w.mean():.4f} std={w.std():.4f} "
          f"min={w.min():.4f} max={w.max():.4f}")
    print(f"           < 0.3 (suppress): {lo:5d}  |  0.4-0.6 (useless): {mid:5d}  |  > 0.7 (trust): {hi:5d}")


# ──────────────────────────────────────────────────────────────────────────────
# Load data helpers
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path, n=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if n and i >= n:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_d2",   type=int, default=None,
                    help="Cap D2 pairs for scoring (default: all ~10k). Use 500 for quick test.")
    ap.add_argument("--n_test", type=int, default=None,
                    help="Cap test pairs for Step A (default: all).")
    ap.add_argument("--quick",  action="store_true",
                    help="Shorthand for --n_d2 500 --n_test 200")
    args = ap.parse_args()

    if args.quick:
        if args.n_d2   is None: args.n_d2   = 500
        if args.n_test is None: args.n_test = 200

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Config] device={device}  p_keep={P_KEEP:.4f}  eps={EPSILON}")
    print(f"[Config] out_dir={OUT_DIR}")

    # ── Load M1 ──────────────────────────────────────────────────────────────
    print("\n[Load] Loading base model + M1 adapter...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    m1 = PeftModel.from_pretrained(base, M1_ADAPTER, adapter_name="stage1")
    m1 = m1.to(device).eval()
    print("[Load] Done.")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP A — M1 accuracy on clean test set
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP A  — M1 preference accuracy on clean HH-RLHF test set")
    print("="*70)

    test_rows = load_jsonl(TEST_JSONL, n=args.n_test)
    print(f"  Test set size: {len(test_rows)}")

    correct_a = 0
    margins_a = []
    for i, r in enumerate(test_rows):
        # M1 DPO reward (log π_M1 - log π_base)
        rc = dpo_reward(m1, tok, r["prompt"], r["chosen"],   device)
        rr = dpo_reward(m1, tok, r["prompt"], r["rejected"], device)
        margin = rc - rr
        margins_a.append(margin)
        if margin > 0:
            correct_a += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_rows)}] running_acc={correct_a/(i+1):.3f}", flush=True)

    acc_a = correct_a / len(test_rows)
    m_arr = np.array(margins_a)
    print(f"\n  ✅ Step A result:")
    print(f"     M1 accuracy  = {acc_a:.4f}  ({correct_a}/{len(test_rows)})")
    print(f"     margin mean  = {m_arr.mean():.4f}  std={m_arr.std():.4f}")
    print(f"     margin range = [{m_arr.min():.2f}, {m_arr.max():.2f}]")

    step_a = {"accuracy": acc_a, "n": len(test_rows),
               "margin_mean": float(m_arr.mean()), "margin_std": float(m_arr.std()),
               "margin_min": float(m_arr.min()),   "margin_max": float(m_arr.max())}
    with open(os.path.join(OUT_DIR, "step_a_m1_clean_accuracy.json"), "w") as f:
        json.dump(step_a, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP B — Raw delta distribution on D2
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP B  — M1 DPO-reward delta distribution on D2 (RR-flipped)")
    print("="*70)

    d2_rows = load_jsonl(D2_JSONL, n=args.n_d2)
    N = len(d2_rows)
    print(f"  D2 size: {N}")

    deltas = []
    for i, r in enumerate(d2_rows):
        rc = dpo_reward(m1, tok, r["prompt"], r["chosen"],   device)
        rr = dpo_reward(m1, tok, r["prompt"], r["rejected"], device)
        deltas.append(rc - rr)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{N}] delta={deltas[-1]:.3f}  running_mean={np.mean(deltas):.3f}", flush=True)

    d = np.array(deltas)
    delta_std  = float(d.std())
    delta_mad  = float(np.median(np.abs(d - np.median(d))))
    robust_tau = max(1.0, delta_mad * 1.4826 * 2.0)   # robust alternative to auto-tau
    auto_tau   = max(1.0, delta_std * 2.0)

    print(f"\n  ✅ Step B result:")
    print(f"     mean   = {d.mean():.4f}")
    print(f"     std    = {delta_std:.4f}")
    print(f"     median = {np.median(d):.4f}")
    print(f"     MAD    = {delta_mad:.4f}  (robust std proxy)")
    print(f"     min    = {d.min():.4f}")
    print(f"     max    = {d.max():.4f}")
    print(f"     pct10  = {np.percentile(d, 10):.4f}  pct90={np.percentile(d,90):.4f}")
    print(f"     auto-tau (2*std)        = {auto_tau:.4f}  ← likely too large?")
    print(f"     robust-tau (2*1.4826*MAD) = {robust_tau:.4f}  ← proposed fix")
    print(f"     frac(delta>0) = {(d>0).mean():.4f}  ← M1 agrees with D2 label")
    print(f"     frac(delta<0) = {(d<0).mean():.4f}  ← M1 disagrees (flipped?)")

    step_b = {
        "N": N, "mean": float(d.mean()), "std": delta_std,
        "median": float(np.median(d)), "mad": delta_mad,
        "min": float(d.min()), "max": float(d.max()),
        "p10": float(np.percentile(d, 10)), "p90": float(np.percentile(d, 90)),
        "auto_tau": auto_tau, "robust_tau": robust_tau,
        "frac_delta_pos": float((d > 0).mean()),
        "frac_delta_neg": float((d < 0).mean()),
    }
    with open(os.path.join(OUT_DIR, "step_b_delta_distribution.json"), "w") as f:
        json.dump(step_b, f, indent=2)

    # Save raw deltas for histogram
    np.save(os.path.join(OUT_DIR, "deltas_d2.npy"), d)
    print(f"  Raw deltas saved to {OUT_DIR}/deltas_d2.npy")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP C — SB weight distribution under auto-tau and tau sweep
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP C  — SB weight distribution: auto-tau vs fixed tau sweep")
    print("="*70)

    taus_to_test = [("auto",   auto_tau),
                    ("robust", robust_tau)] + [(f"fixed_{t}", t) for t in TAU_SWEEP]

    step_c = {}
    for label, tau in taus_to_test:
        weights = compute_weights(deltas, tau, P_KEEP)
        w = np.array(weights)
        lo  = int((w < 0.3).sum())
        hi  = int((w > 0.7).sum())
        mid = int(((w >= 0.4) & (w <= 0.6)).sum())
        print(f"\n  tau={tau:8.2f}  [{label}]")
        weight_stats(weights, label)
        step_c[label] = {
            "tau": tau, "w_mean": float(w.mean()), "w_std": float(w.std()),
            "w_min": float(w.min()), "w_max": float(w.max()),
            "n_suppress": lo, "n_useless": mid, "n_trust": hi,
            "frac_informative": float((lo + hi) / len(weights)),
        }

    with open(os.path.join(OUT_DIR, "step_c_weight_distribution.json"), "w") as f:
        json.dump(step_c, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP D — Sign convention: flipped pairs should get LOW w_i
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("STEP D  — Sign convention check on inferred-flipped pairs")
    print("="*70)

    # Infer flipped pairs: same id, but chosen/rejected are swapped vs original
    print("  Loading original (pre-flip) D1 to infer flipped pairs by id...")
    orig_rows = load_jsonl(ORIG_JSONL, n=args.n_d2)
    orig_by_id = {r["id"]: r for r in orig_rows}

    d2_ids  = [r["id"] for r in d2_rows]
    flipped = []   # index into d2_rows / deltas
    kept    = []

    for i, r in enumerate(d2_rows):
        orig = orig_by_id.get(r["id"])
        if orig is None:
            continue
        # Flipped = chosen_in_D2 == rejected_in_orig  (RR swapped the label)
        if r["chosen"].strip() == orig["rejected"].strip():
            flipped.append(i)
        else:
            kept.append(i)

    print(f"  Identified: {len(flipped)} flipped  |  {len(kept)} kept  "
          f"(of {len(d2_rows)} D2 pairs with matching orig id)")

    if not flipped:
        print("  ⚠️  No flipped pairs found by id-match — "
              "orig file may be D2 not D1. Sign check skipped.")
    else:
        # Use robust tau for this check
        weights_robust = compute_weights(deltas, robust_tau, P_KEEP)

        d_flip = np.array([deltas[i]          for i in flipped])
        w_flip = np.array([weights_robust[i]  for i in flipped])
        d_kept = np.array([deltas[i]          for i in kept])
        w_kept = np.array([weights_robust[i]  for i in kept])

        print(f"\n  ✅ Flipped pairs (should: delta<0, w<0.5):")
        print(f"     delta  mean={d_flip.mean():.4f}  std={d_flip.std():.4f}  "
              f"frac<0={(d_flip<0).mean():.3f}")
        print(f"     weight mean={w_flip.mean():.4f}  std={w_flip.std():.4f}  "
              f"frac<0.5={(w_flip<0.5).mean():.3f}")

        print(f"\n  ✅ Kept pairs (should: delta>0, w>0.5):")
        print(f"     delta  mean={d_kept.mean():.4f}  std={d_kept.std():.4f}  "
              f"frac>0={(d_kept>0).mean():.3f}")
        print(f"     weight mean={w_kept.mean():.4f}  std={w_kept.std():.4f}  "
              f"frac>0.5={(w_kept>0.5).mean():.3f}")

        # Sign correctness: flipped→low w, kept→high w
        sign_ok_flip = float((w_flip < 0.5).mean())
        sign_ok_kept = float((w_kept > 0.5).mean())
        sign_correct = sign_ok_flip > 0.55 and sign_ok_kept > 0.55

        if sign_correct:
            print(f"\n  ✅ Sign convention is CORRECT: "
                  f"flipped→w<0.5 ({sign_ok_flip:.2%})  kept→w>0.5 ({sign_ok_kept:.2%})")
        else:
            print(f"\n  ❌ Sign convention may be WRONG: "
                  f"flipped→w<0.5 ({sign_ok_flip:.2%})  kept→w>0.5 ({sign_ok_kept:.2%})")
            print(f"     If both are ~0.5 → tau is too large (compression, not sign bug)")

        step_d = {
            "n_flipped": len(flipped), "n_kept": len(kept),
            "robust_tau_used": robust_tau,
            "flipped": {
                "delta_mean": float(d_flip.mean()), "delta_frac_neg": float((d_flip<0).mean()),
                "w_mean": float(w_flip.mean()),     "w_frac_lt_half": float((w_flip<0.5).mean()),
            },
            "kept": {
                "delta_mean": float(d_kept.mean()), "delta_frac_pos": float((d_kept>0).mean()),
                "w_mean": float(w_kept.mean()),     "w_frac_gt_half": float((w_kept>0.5).mean()),
            },
            "sign_convention_correct": sign_correct,
        }
        with open(os.path.join(OUT_DIR, "step_d_sign_check.json"), "w") as f:
            json.dump(step_d, f, indent=2)

    # ══════════════════════════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  Step A  M1 clean accuracy : {acc_a:.4f}")
    print(f"  Step B  delta mean        : {d.mean():.4f}  (neg → M1 disagrees with D2)")
    print(f"          auto-tau          : {auto_tau:.2f}  |  robust-tau: {robust_tau:.2f}")
    print(f"  Step C  see {OUT_DIR}/step_c_weight_distribution.json")
    print(f"  Step D  see {OUT_DIR}/step_d_sign_check.json")
    print(f"\n  All outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
