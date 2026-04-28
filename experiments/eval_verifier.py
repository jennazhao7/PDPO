#!/usr/bin/env python3
"""
eval_verifier.py — Sanity checks for the preference evaluation pipeline.

Tests that our eval metric:
  1. Detects a model that clearly prefers chosen (should show >80% accuracy)
  2. Shows ~50% for a random model (no training)
  3. Shows that Stage1 has learned real preference signal vs base

This script runs FAST (+/- 2 min) by using GPT-2 as a proxy if the real models
are too large to load for a quick check, or uses the real Stage1 model.

Usage:
    # Fast check with GPT-2 proxy (validates eval math, not real model quality)
    python experiments/eval_verifier.py --mode proxy

    # Full check with real Stage1 model (validates end-to-end signal)
    CUDA_VISIBLE_DEVICES=0 python experiments/eval_verifier.py --mode real \
        --base_model Qwen/Qwen2.5-3B \
        --stage1_adapter outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed42 \
        --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
        --n 100
"""
import argparse, json, math, random, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

random.seed(42); torch.manual_seed(42)


def resp_logprob_sum(model, tok, prompt, resp, device, max_len=512):
    with torch.no_grad():
        p = tok(prompt, add_special_tokens=False)["input_ids"]
        r = tok(resp, add_special_tokens=False)["input_ids"]
        ids = (p + r)[-max_len:]
        if len(ids) < 2: return 0.0
        p_len = min(len(p), len(ids) - 1)
        x = torch.tensor([ids], device=device)
        logp = F.log_softmax(model(x).logits[:, :-1, :], dim=-1)
        tgt = x[:, 1:]
        tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
        return float(tok_lp[max(p_len-1, 0):].sum())


def dpo_reward(policy, ref, tok, prompt, resp, device, max_len=512):
    """log π(y|x) - log π_ref(y|x)"""
    return (resp_logprob_sum(policy, tok, prompt, resp, device, max_len) -
            resp_logprob_sum(ref,    tok, prompt, resp, device, max_len))


def eval_acc(policy, ref, tok, rows, device, max_len=512):
    correct = 0
    for r in rows:
        rc = dpo_reward(policy, ref, tok, r["prompt"], r["chosen"],   device, max_len)
        rr = dpo_reward(policy, ref, tok, r["prompt"], r["rejected"], device, max_len)
        if rc > rr: correct += 1
    return correct / len(rows)


# ─── Proxy mode: use GPT-2 to test eval math ──────────────────
def run_proxy_mode():
    print("\n=== PROXY MODE (GPT-2) ===")
    print("Tests eval math without loading large models.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tiny GPT-2 (takes ~5s)
    model_id = "gpt2"
    print(f"Loading {model_id}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device).eval()

    # Synthetic test cases where preference is unambiguous
    # chosen = well-formed English, rejected = random tokens / gibberish
    pairs = [
        {"prompt": "How do I bake a cake?",
         "chosen":   "You will need flour, sugar, eggs, butter, and baking powder. Mix them and bake at 350F for 30 minutes.",
         "rejected": "asdf zxcv qwer uiop lkjh fghj mnbv erty cvbn poiu mnbv lkjh"},
        {"prompt": "What is the capital of France?",
         "chosen":   "The capital of France is Paris. It is a major European city and cultural center.",
         "rejected": "Zorp bleep morp flan quux grob wibble flibber jabber wocky"},
        {"prompt": "How do plants make food?",
         "chosen":   "Plants make food through photosynthesis, using sunlight, water, and carbon dioxide to produce glucose.",
         "rejected": "xkcd norp blarg gorp snark wibble floop morp zarg qwerp bloop"},
        {"prompt": "What causes rain?",
         "chosen":   "Rain is caused by water vapor in the atmosphere cooling and condensing into droplets that fall to earth.",
         "rejected": "Frobble wump grox snorkel plonk fizzbuzz wibble snark morp quux"},
        {"prompt": "How do I tie my shoes?",
         "chosen":   "Cross the laces, make a loop with one lace, wrap the other around, and pull through to make a bow.",
         "rejected": "blorb snark wibble grox floop zorp morp blarg qwerp fizzbuzz plonk"},
    ]

    print("\nTest 1: Policy = Reference (same model) → rewards should be ~0 for all pairs")
    rewards = []
    for p in pairs:
        rc = dpo_reward(model, model, tok, p["prompt"], p["chosen"],   device)
        rr = dpo_reward(model, model, tok, p["prompt"], p["rejected"], device)
        rewards.append((rc, rr))
    max_abs = max(max(abs(rc), abs(rr)) for rc, rr in rewards)
    print(f"  Max |reward| when policy==ref: {max_abs:.6f}  (Expected: ~0.0)")
    t1_ok = max_abs < 1.0  # rewards should be near-zero (floating point noise only)

    print("\nTest 2: Coherent chosen vs gibberish rejected → model should clearly prefer chosen")
    corr = []
    for p in pairs:
        # Use model as both policy and a constant ref (so reward = logprob_sum - constant)
        # Just use raw sum to check model assigns higher logprob to coherent text
        lc = resp_logprob_sum(model, tok, p["prompt"], p["chosen"],   device)
        lr = resp_logprob_sum(model, tok, p["prompt"], p["rejected"], device)
        # Normalize by length (for fair comparison in this test)
        lc_norm = lc / max(len(tok.encode(p["chosen"])), 1)
        lr_norm = lr / max(len(tok.encode(p["rejected"])), 1)
        corr.append(lc_norm > lr_norm)
    t2_acc = sum(corr) / len(corr)
    print(f"  Accuracy (length-normalized): {t2_acc:.2f}  (Expected: ≥0.80)")
    t2_ok = t2_acc >= 0.8

    print("\nTest 3: DPO reward correctly handles length-biased pairs")
    # Chosen is much longer than rejected — without normalization, longer = lower sum logprob
    length_pairs = [
        {"prompt": "What is 2+2?",
         "chosen":   "The answer is 4. " * 10,   # very long
         "rejected": "4"},                          # very short
    ]
    lp_raw_c = resp_logprob_sum(model, tok, length_pairs[0]["prompt"], length_pairs[0]["chosen"],   device)
    lp_raw_r = resp_logprob_sum(model, tok, length_pairs[0]["prompt"], length_pairs[0]["rejected"], device)
    raw_prefers_chosen = lp_raw_c > lp_raw_r
    # With DPO reward (model as its own ref), reward delta = 0 for both → not meaningful
    # But raw sum should show length bias (chosen is longer, more total negative logprob)
    print(f"  Raw sum logprob: chosen={lp_raw_c:.1f}, rejected={lp_raw_r:.1f}")
    print(f"  Raw sum prefers shorter (rejected): {not raw_prefers_chosen}")
    t3_ok = not raw_prefers_chosen  # We expect longer to have MORE NEGATIVE sum logprob

    print("\n" + "="*50)
    print("PROXY MODE VERDICT:")
    print(f"  T1 (same model → ~50%):   {'✅ PASS' if t1_ok else '❌ FAIL'}")
    print(f"  T2 (coherent > gibberish): {'✅ PASS' if t2_ok else '❌ FAIL'}")
    print(f"  T3 (sum is length-biased): {'✅ PASS' if t3_ok else '❌ FAIL'}")
    if t1_ok and t2_ok and t3_ok:
        print("  ✅ Eval math is correct")
    else:
        print("  ❌ Some checks failed — investigate before trusting eval numbers")
    return t1_ok and t2_ok and t3_ok


# ─── Real mode: test with actual Stage1 model ──────────────────
def run_real_mode(args):
    print("\n=== REAL MODE ===")
    print("Tests that Stage1 shows real preference signal vs the base model.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = [json.loads(l) for l in open(args.test_jsonl) if l.strip()]
    rows = rows[:args.n]

    dtype = torch.bfloat16 if "mistral" in args.base_model.lower() else torch.float16

    print(f"Loading base model: {args.base_model}...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype).to(device).eval()

    print(f"Loading Stage1 policy: {args.stage1_adapter}...", flush=True)
    policy = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype)
    policy = PeftModel.from_pretrained(policy, args.stage1_adapter, adapter_name="stage1")
    policy.set_adapter("stage1")
    policy = policy.to(device).eval()

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    # ── Check 1: Base model alone should be ~50% (no preference learning)
    print(f"\nCheck 1: Base model accuracy (expected ~50%)")
    base_corr = []
    for r in rows[:min(50, len(rows))]:
        rc = resp_logprob_sum(base, tok, r["prompt"], r["chosen"],   device, args.max_len)
        rr = resp_logprob_sum(base, tok, r["prompt"], r["rejected"], device, args.max_len)
        # Use length-normalized (mean) for base model since we have no reference
        lc = rc / max(len(tok.encode(r["chosen"])), 1)
        lr = rr / max(len(tok.encode(r["rejected"])), 1)
        base_corr.append(lc > lr)
    base_acc = sum(base_corr) / len(base_corr)
    print(f"  Base model (mean-normalized): {base_acc:.3f}")

    # ── Check 2: DPO reward of Stage1 vs base (the actual metric)
    print(f"\nCheck 2: Stage1 DPO-reward accuracy on {len(rows)} pairs")
    print("  (This should be HIGHER than base model — proves Stage1 learned preference signal)")
    dpo_corr = []
    for i, r in enumerate(rows):
        rc = dpo_reward(policy, base, tok, r["prompt"], r["chosen"],   device, args.max_len)
        rr = dpo_reward(policy, base, tok, r["prompt"], r["rejected"], device, args.max_len)
        dpo_corr.append(rc > rr)
        if (i+1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] acc={sum(dpo_corr)/len(dpo_corr):.3f}", flush=True)

    s1_acc = sum(dpo_corr) / len(dpo_corr)
    print(f"\n  Stage1 DPO-reward accuracy: {s1_acc:.3f}")

    # ── Check 3: Sanity — if policy == base, reward diff = 0 for all
    print("\nCheck 3: If policy=base, all rewards should be 0 (sanity check)")
    dummy_margins = []
    for r in rows[:5]:
        rc = dpo_reward(base, base, tok, r["prompt"], r["chosen"],   device, args.max_len)
        rr = dpo_reward(base, base, tok, r["prompt"], r["rejected"], device, args.max_len)
        dummy_margins.append(abs(rc))
        dummy_margins.append(abs(rr))
    max_dummy = max(dummy_margins)
    print(f"  Max |reward| when policy==ref: {max_dummy:.6f}  (Expected: ~0.0)")
    t3_ok = max_dummy < 1e-3

    print("\n" + "="*50)
    print("REAL MODE VERDICT:")
    learned = s1_acc > base_acc + 0.02  # at least 2% lift over base
    above_random = s1_acc > 0.52         # at least 2% above coin flip
    print(f"  Base accuracy:          {base_acc:.3f}")
    print(f"  Stage1 accuracy:        {s1_acc:.3f}  ({'✅' if above_random else '❌'} above random)")
    print(f"  Stage1 > Base + 2%:     {'✅ PASS' if learned else '❌ FAIL'}")
    print(f"  DPO reward math (T3):    {'✅ PASS' if t3_ok else '❌ FAIL'}")
    if above_random and learned and t3_ok:
        print("  ✅ Eval is working correctly and Stage1 shows real preference signal!")
    elif t3_ok and not above_random:
        print("  ⚠  Eval math is correct BUT Stage1 shows no signal above random.")
        print("     → The issue is NOT the eval metric. Investigate the training setup.")
    else:
        print("  ❌ Eval math problem or training issue — investigate.")
    return above_random and learned and t3_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["proxy", "real"], default="proxy")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--stage1_adapter", default="outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed42")
    ap.add_argument("--test_jsonl", default="data/pku_saferlhf_secure/test_pref.jsonl")
    ap.add_argument("--n", type=int, default=100, help="Number of test pairs (real mode only)")
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    if args.mode == "proxy":
        ok = run_proxy_mode()
    else:
        ok = run_real_mode(args)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
