#!/usr/bin/env python3
"""
Diagnostic Checks for Job 3 (Oracle SB Interpolated Reference).

Check 1:
  On the D2 training data, compute M1's log-ratio (delta_m1 = log π_M1(chosen) - log π_M1(rejected))
  for every pair. Split by sign:
    - correct sign: delta_m1 > 0  (M1 agrees with oracle label — chosen > rejected)
    - wrong sign:   delta_m1 < 0  (M1 disagrees)
  Report fraction of each and the mean DPO margin from Job 3 for each group.

Check 2:
  Among the N=221 examples in bin 0 of Job 3's eval (model predicted: rejected > chosen with
  near-zero confidence), what fraction have M1's log-ratio with the wrong sign?
  If >60%, the mechanism is confirmed.

Usage (CPU-only feasible, but GPU strongly recommended):
  CUDA_VISIBLE_DEVICES=0 python check_m1_sign.py

All paths are hardcoded to match the floor/ceiling experiment layout.
"""

from __future__ import annotations
import json, math, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = "/users/jzhao7/PDPO/stage2_debugging"
EXP  = f"{ROOT}/experiments/pku_floor_ceiling"

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
M1_PATH    = f"{ROOT}/stage1/results_instruct_eps1.0_seed42/pku_eps1.0_s42"
JOB3_PATH  = f"{EXP}/models/pku_oracle_sb_corrref"
DATA_PATH  = f"{EXP}/d2_pku_with_flipped_eps1.0_seed42.jsonl"
RESULT_J3  = f"{EXP}/results/job3.json"
OUT_PATH   = f"{EXP}/results/check_m1_sign.json"

MAX_LEN = 512
EPSILON = 1.0
ORACLE_DELTA = 10.0
BETA = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_oracle_weights(was_flipped_list, epsilon, oracle_delta=10.0):
    gamma_eps = 1.0 / (1.0 + math.exp(epsilon))
    p_keep = 1.0 - gamma_eps
    weights = []
    for flipped in was_flipped_list:
        delta = -oracle_delta if flipped else +oracle_delta
        q_i = 1.0 / (1.0 + math.exp(-delta))
        num = p_keep * q_i
        den = p_keep * q_i + gamma_eps * (1.0 - q_i)
        w_i = max(0.01, min(0.99, num / den if den > 1e-12 else 0.5))
        weights.append(w_i)
    return weights


def compute_response_logprobs(model, input_ids, prompt_lens, pad_id):
    """Sum log P(response tokens | context). Returns tensor of shape [B]."""
    out = model(input_ids=input_ids)
    shift_logits = out.logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    tok_lp = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    B, L = tok_lp.shape
    pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)
    mask = (pos >= (prompt_lens.unsqueeze(1) - 1)) & (shift_labels != pad_id)
    return (tok_lp * mask.float()).sum(dim=1)


@torch.no_grad()
def score_single(model, ids, p_len, pad_id, device):
    x  = torch.tensor([ids], device=device)
    pl = torch.tensor([p_len], device=device)
    return compute_response_logprobs(model, x, pl, pad_id).item()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"[device={device}  dtype={dtype}]")

    # Load data
    rows = [json.loads(l) for l in open(DATA_PATH) if l.strip()]
    N = len(rows)
    print(f"[Data] N={N}")
    prompts     = [r["prompt"]      for r in rows]
    chosen      = [r["chosen"]      for r in rows]
    rejected    = [r["rejected"]    for r in rows]
    was_flipped = [r["was_flipped"] for r in rows]
    weights     = compute_oracle_weights(was_flipped, EPSILON, ORACLE_DELTA)

    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    pad_id = tok.pad_token_id

    # Pre-tokenise
    def enc(prompt, resp):
        p_ids = tok.encode(prompt, add_special_tokens=True)
        r_ids = tok.encode(resp,   add_special_tokens=False)
        full  = (p_ids + r_ids)[:MAX_LEN]
        return full, len(p_ids)

    c_tok = [enc(p, c) for p, c in zip(prompts, chosen)]
    r_tok = [enc(p, r) for p, r in zip(prompts, rejected)]

    # ── Score M1 ─────────────────────────────────────────────────────────────
    print(f"\n[M1] Loading {BASE_MODEL} + adapter {M1_PATH} (frozen)...")
    base_for_m1 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    m1 = PeftModel.from_pretrained(base_for_m1, M1_PATH, adapter_name="stage1")
    m1.set_adapter("stage1")
    m1 = m1.to(device).eval()
    for p in m1.parameters():
        p.requires_grad = False

    m1_c_lps, m1_r_lps = [], []
    print("[M1] Scoring training pairs...")
    for i in range(N):
        c_ids, c_plen = c_tok[i]
        r_ids, r_plen = r_tok[i]
        m1_c_lps.append(score_single(m1, c_ids, c_plen, pad_id, device))
        m1_r_lps.append(score_single(m1, r_ids, r_plen, pad_id, device))
        if (i + 1) % 500 == 0:
            print(f"  [M1 scored {i+1}/{N}]", flush=True)

    del m1, base_for_m1
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    m1_delta = torch.tensor(m1_c_lps) - torch.tensor(m1_r_lps)  # shape [N]
    print(f"\n[M1] delta stats: min={m1_delta.min():.3f}  max={m1_delta.max():.3f}  "
          f"mean={m1_delta.mean():.3f}  std={m1_delta.std():.3f}")

    # ── Score Job 3 (M2) ──────────────────────────────────────────────────────
    print(f"\n[M2/Job3] Loading {BASE_MODEL} + adapter {JOB3_PATH} (frozen, for margin check)...")
    base_for_m2 = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    m2 = PeftModel.from_pretrained(base_for_m2, JOB3_PATH, adapter_name="stage2")
    m2.set_adapter("stage2")
    m2 = m2.to(device).eval()
    for p in m2.parameters():
        p.requires_grad = False

    # Also load base (frozen) for the interpolated reference
    print(f"[Base] Loading {BASE_MODEL} (frozen)...")
    base_ref = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    base_ref = base_ref.to(device).eval()
    for p in base_ref.parameters():
        p.requires_grad = False

    # We also need base logprobs for the interpolated reference
    # (can reuse the cache from job2 if it exists)
    cache_path = f"{EXP}/models/pku_oracle_sb_baseref/base_ref_cache.pt"
    if os.path.exists(cache_path):
        print(f"[Cache] Loading base ref logprobs from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        base_c_lps = cache["ref_c_lps"]
        base_r_lps = cache["ref_r_lps"]
    else:
        print("[Base] No cache found — scoring with base model on the fly...")
        base_c_lps, base_r_lps = [], []
        for i in range(N):
            c_ids, c_plen = c_tok[i]
            r_ids, r_plen = r_tok[i]
            base_c_lps.append(score_single(base_ref, c_ids, c_plen, pad_id, device))
            base_r_lps.append(score_single(base_ref, r_ids, r_plen, pad_id, device))
            if (i + 1) % 500 == 0:
                print(f"  [base scored {i+1}/{N}]", flush=True)

    del base_ref
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    base_delta = torch.tensor(base_c_lps) - torch.tensor(base_r_lps)
    w_t = torch.tensor(weights)

    # Interpolated reference = w_i * m1_delta + (1-w_i) * base_delta
    interp_ref = w_t * m1_delta + (1.0 - w_t) * base_delta

    print("[M2] Scoring training pairs (Job 3 / M2 model)...")
    m2_c_lps, m2_r_lps = [], []
    for i in range(N):
        c_ids, c_plen = c_tok[i]
        r_ids, r_plen = r_tok[i]
        m2_c_lps.append(score_single(m2, c_ids, c_plen, pad_id, device))
        m2_r_lps.append(score_single(m2, r_ids, r_plen, pad_id, device))
        if (i + 1) % 500 == 0:
            print(f"  [M2 scored {i+1}/{N}]", flush=True)

    del m2, base_for_m2
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    m2_delta  = torch.tensor(m2_c_lps) - torch.tensor(m2_r_lps)
    # DPO margin = beta * (pi_log_ratio - ref_log_ratio)
    margin    = BETA * (m2_delta - interp_ref)

    # ── Check 1: margin split by M1 sign ─────────────────────────────────────
    correct_sign = m1_delta > 0   # M1 agrees with oracle label direction
    wrong_sign   = m1_delta < 0

    n_correct = correct_sign.sum().item()
    n_wrong   = wrong_sign.sum().item()
    frac_correct = n_correct / N
    frac_wrong   = n_wrong   / N

    margin_correct = margin[correct_sign].mean().item()
    margin_wrong   = margin[wrong_sign].mean().item()

    print("\n" + "="*60)
    print("  CHECK 1: DPO Margin Split by M1 Log-Ratio Sign")
    print("="*60)
    print(f"  Pairs where M1 correct sign (delta_m1 > 0): {n_correct}/{N} = {frac_correct:.3f}")
    print(f"  Pairs where M1 wrong sign   (delta_m1 < 0): {n_wrong}/{N}  = {frac_wrong:.3f}")
    print(f"  Mean margin (M1 correct sign): {margin_correct:.4f}")
    print(f"  Mean margin (M1 wrong sign):   {margin_wrong:.4f}")
    if margin_wrong < 0 and abs(margin_wrong) > abs(margin_correct):
        print("  ✅ DIAGNOSIS CONFIRMED: Large negative margin when M1 has wrong sign.")
    elif margin_wrong < 0:
        print("  ⚠️  margin is negative for wrong sign (correct direction), but magnitude modest.")
    else:
        print("  ❌ margin_wrong is positive — M1 sign does not appear to be the issue.")

    # ── Check 2: Bin 0 analysis ───────────────────────────────────────────────
    # Bin 0 = model's implicit reward puts rejected > chosen with <10% confidence
    # i.e. margin < 0 and |sigma(margin) - 0.5| < 0.05  →  margin in roughly (-inf,0)
    # From the eval script, bin 0 is confidence [0, 0.1), i.e. sigma(margin) < 0.1
    # That means margin < logit(0.1) = log(0.1/0.9) ≈ -2.2
    # But we're on the TRAINING data, not test data.  The 221 examples are from the
    # TEST set eval.  We report the training-data statistics as a proxy.
    # For a proper check of test-set bin 0 we'd need the test set scored too.
    # We report training statistics here and flag the caveat.

    # Bin 0 analog on training: margin < 0 (model predicts rejected > chosen)
    bin0_train = margin < 0
    n_bin0_train = bin0_train.sum().item()
    frac_wrong_in_bin0 = (wrong_sign & bin0_train).sum().item() / max(n_bin0_train, 1)

    print("\n" + "="*60)
    print("  CHECK 2: 'Bin 0 Analog' on Training Data")
    print("="*60)
    print(f"  Training pairs with margin < 0 (wrong prediction): {n_bin0_train}/{N}")
    print(f"  Of those, fraction with M1 wrong sign: {frac_wrong_in_bin0:.3f}")
    if frac_wrong_in_bin0 >= 0.60:
        print("  ✅ MECHANISM CONFIRMED: ≥60% of wrong-prediction pairs have M1 wrong sign.")
    else:
        print(f"  (threshold 0.60 — observed {frac_wrong_in_bin0:.3f})")

    # ── Summary stats for further reference ───────────────────────────────────
    print("\n" + "="*60)
    print("  ADDITIONAL STATS")
    print("="*60)
    print(f"  Overall M1 gamma (fraction correct sign): {frac_correct:.3f}")
    print(f"  mean interp_ref: {interp_ref.mean():.4f}  "
          f"(base: {base_delta.mean():.4f}  m1: {m1_delta.mean():.4f})")
    print(f"  Mean margin overall: {margin.mean():.4f}")
    print(f"  Fraction margin > 0 (correct): {(margin > 0).float().mean():.3f}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "N": N,
        "m1_correct_sign_frac": frac_correct,
        "m1_wrong_sign_frac": frac_wrong,
        "margin_mean_m1_correct_sign": margin_correct,
        "margin_mean_m1_wrong_sign": margin_wrong,
        "training_bin0_count": n_bin0_train,
        "training_bin0_frac_wrong_m1_sign": frac_wrong_in_bin0,
        "margin_mean_overall": margin.mean().item(),
        "margin_frac_positive": (margin > 0).float().mean().item(),
        "interp_ref_mean": interp_ref.mean().item(),
        "base_delta_mean": base_delta.mean().item(),
        "m1_delta_mean": m1_delta.mean().item(),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved] {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
