#!/usr/bin/env python3
"""
Minimal Soft Bayes Fix Verifier.

Runs everything in one script, printing every step to stdout.
Uses first N_PAIRS rows of D2 for speed.
All output goes to stdout (no interactive prompts).

Usage (background job):
  CUDA_VISIBLE_DEVICES=0 nohup python -u experiments/verify_sb_fix.py \
      > outputs_quick_test/logs/verify_sb.txt 2>&1 &

Expected: SB accuracy > Stage1 accuracy > base accuracy
"""
import json, math, gc, os, sys
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType
from datasets import load_dataset

# ─────────────── CONFIGURATION ───────────────
MODEL        = "Qwen/Qwen2.5-3B"
S1_ADAPTER   = "outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed42"
D2_DATA      = "outputs_new_models/preprocessing/d2_rr_flipped_eps1.0_seed42.jsonl"
TEST_JSONL   = "data/pku_saferlhf_secure/test_pref.jsonl"
N_PAIRS      = 200     # how many D2 pairs to score + train on (full=5082, quick=200)
N_TEST       = 200     # how many test pairs to eval on (full=1000, quick=200)
TRAIN_STEPS  = 30      # optimizer steps (full=300, quick=30, ~1 per 16 samples)
EPSILON      = 1.0
BETA         = 0.1
LR           = 1e-5
MAX_LEN      = 256
SEED         = 42
# ─────────────────────────────────────────────

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
p_keep = math.exp(EPSILON) / (math.exp(EPSILON) + 1)
print(f"\nDevice: {device} | p_keep={p_keep:.4f} | N_PAIRS={N_PAIRS} | steps={TRAIN_STEPS}")


# ── helpers ──────────────────────────────────

def load_base_with_s1(trainable=False):
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.config.use_cache = False
    m = PeftModel.from_pretrained(model, S1_ADAPTER, adapter_name="stage1")
    m.set_adapter("stage1")
    if not trainable:
        for p in m.parameters(): p.requires_grad = False
    return m.to(device)


def get_tok():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    return tok


@torch.no_grad()
def seq_logprob_sum(model, tok, prompt, resp):
    """Sum of token log-probs over response only (DPO-correct)."""
    p_ids = tok.encode(prompt, add_special_tokens=True)
    r_ids = tok.encode(resp, add_special_tokens=False)
    ids = (p_ids + r_ids)[-MAX_LEN:]
    x = torch.tensor([ids], device=device)
    logp = F.log_softmax(model(x).logits[:, :-1, :], dim=-1)
    tgt  = x[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    plen = min(len(p_ids), len(ids) - 1)
    return float(tok_lp[max(plen-1, 0):].sum())


@torch.no_grad()
def seq_logprob_mean(model, tok, prompt, resp):
    """Mean of token log-probs (eval-correct, no length bias)."""
    p_ids = tok.encode(prompt, add_special_tokens=True)
    r_ids = tok.encode(resp, add_special_tokens=False)
    ids = (p_ids + r_ids)[-MAX_LEN:]
    x = torch.tensor([ids], device=device)
    logp = F.log_softmax(model(x).logits[:, :-1, :], dim=-1)
    tgt  = x[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)[0]
    plen = min(len(p_ids), len(ids) - 1)
    resp_lp = tok_lp[max(plen-1, 0):]
    denom = max(len(resp_lp), 1)
    return float(resp_lp.sum()) / denom


def load_base_only():
    """Load bare base model (no adapter) — used as reference for S1 eval."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    for p in model.parameters(): p.requires_grad = False
    return model.to(device).eval()


@torch.no_grad()
def eval_s1_accuracy(policy, base, tok, rows):
    """DPO implicit reward: log π_S1(y|x) - log π_base(y|x)"""
    correct = 0
    for r in rows:
        rc = seq_logprob_sum(policy, tok, r["prompt"], r["chosen"])  - seq_logprob_sum(base, tok, r["prompt"], r["chosen"])
        rr = seq_logprob_sum(policy, tok, r["prompt"], r["rejected"]) - seq_logprob_sum(base, tok, r["prompt"], r["rejected"])
        if rc > rr: correct += 1
    return correct / len(rows)


@torch.no_grad()
def eval_s2_accuracy(m2, s1_ref_c, s1_ref_r, tok, rows):
    """DPO implicit reward: log π_S1+S2(y|x) - log π_S1(y|x)
    Uses precomputed S1 logprobs from the scoring phase as the reference."""
    correct = 0
    for i, r in enumerate(rows):
        pi_c = seq_logprob_sum(m2, tok, r["prompt"], r["chosen"])
        pi_r = seq_logprob_sum(m2, tok, r["prompt"], r["rejected"])
        # Reference logprobs come from S1 scoring — but those were D2 pairs.
        # For test rows we need to compute S1 logprobs on the fly.
        # m2 has stage1 adapter — switch to stage1-only for reference.
        ref_c = _s1_logprob(m2, tok, r["prompt"], r["chosen"])
        ref_r = _s1_logprob(m2, tok, r["prompt"], r["rejected"])
        rc = pi_c - ref_c
        rr = pi_r - ref_r
        if rc > rr: correct += 1
    return correct / len(rows)


@torch.no_grad()
def _s1_logprob(m2, tok, prompt, resp):
    """Compute logprob under stage1-only (reference) via adapter switching."""
    try:    m2.set_adapter("stage1")
    except: pass
    val = seq_logprob_sum(m2, tok, prompt, resp)
    try:    m2.set_adapter(["stage1", "stage2"])
    except: m2.set_adapter("stage2")
    return val


def add_stage2(m, name="stage2"):
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                     task_type=TaskType.CAUSAL_LM,
                     target_modules=["q_proj","k_proj","v_proj","o_proj",
                                     "gate_proj","up_proj","down_proj"])
    m.add_adapter(name, cfg)
    try:    m.set_adapter(["stage1", name])
    except: m.set_adapter(name)
    for n, p in m.named_parameters():
        p.requires_grad = f".{name}." in n
    return m


def dpo_loss_soft(pi_c, pi_r, ref_c, ref_r, w):
    m = (pi_c - ref_c) - (pi_r - ref_r)
    return (-(w * F.logsigmoid(BETA * m) + (1-w) * F.logsigmoid(-BETA * m))).mean()


def dpo_loss_hard(pi_c, pi_r, ref_c, ref_r):
    m = (pi_c - ref_c) - (pi_r - ref_r)
    return (-F.logsigmoid(BETA * m)).mean()


def compute_batch_lp(model, ids_list, p_len_list, pad_id):
    """Batched sum log-prob for a list of id lists."""
    max_l = max(len(x) for x in ids_list)
    padded = torch.tensor([[*x, *[pad_id]*(max_l-len(x))] for x in ids_list], device=device)
    p_lens = torch.tensor(p_len_list, device=device)
    logits = model(padded).logits[:, :-1, :]
    logp = F.log_softmax(logits, dim=-1)
    tgt = padded[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    B, L = tok_lp.shape
    pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    mask = (pos >= (p_lens.unsqueeze(1) - 1)) & (tgt != pad_id)
    return (tok_lp * mask.float()).sum(dim=1)


# ═══════════════ MAIN ════════════════════════

tok = get_tok()
pad_id = tok.pad_token_id

# Load test rows
test_rows = [json.loads(l) for l in open(TEST_JSONL) if l.strip()][:N_TEST]

# Load D2 training rows
d2_rows = [json.loads(l) for l in open(D2_DATA) if l.strip()][:N_PAIRS]
d2_prompts  = [r["prompt"]   for r in d2_rows]
d2_chosen   = [r["chosen"]   for r in d2_rows]
d2_rejected = [r["rejected"] for r in d2_rows]

# ─── Step 1: Stage 1 baseline accuracy ───────
print("\n[1/5] Evaluating Stage 1 baseline (DPO implicit reward)...")
base_model = load_base_only()
m1 = load_base_with_s1(trainable=False)
m1.eval()
s1_acc = eval_s1_accuracy(m1, base_model, tok, test_rows)
print(f"      Stage 1 accuracy: {s1_acc:.4f}")
del base_model; gc.collect(); torch.cuda.empty_cache()

# ─── Step 2: Score D2 with M1 + get ref lps ──
print(f"\n[2/5] Scoring {N_PAIRS} D2 pairs with M1 (reference logprobs)...")
deltas, ref_c_lps, ref_r_lps = [], [], []
for i, (p, c, r) in enumerate(zip(d2_prompts, d2_chosen, d2_rejected)):
    lc = seq_logprob_sum(m1, tok, p, c)
    lr = seq_logprob_sum(m1, tok, p, r)
    with m1.disable_adapter():
        lc_base = seq_logprob_sum(m1, tok, p, c)
        lr_base = seq_logprob_sum(m1, tok, p, r)
    delta_dpo = (lc - lc_base) - (lr - lr_base)
    deltas.append(delta_dpo)
    ref_c_lps.append(lc)
    ref_r_lps.append(lr)
    if (i+1) % 20 == 0 or i < 3:
        print(f"      [{i+1}/{N_PAIRS}] delta={delta_dpo:.2f}", flush=True)

delta_std = float(torch.tensor(deltas).std())
tau = max(1.0, delta_std * 2)
print(f"      delta_std={delta_std:.2f}  tau={tau:.2f}")

# Compute SB weights
weights = []
for d in deltas:
    q = 1 / (1 + math.exp(-max(-500, min(500, d / tau))))
    w = p_keep * q / (p_keep * q + (1 - p_keep) * (1 - q))
    weights.append(max(0.01, min(0.99, w)))

w_mean = sum(weights) / len(weights)
w_spread = max(weights) - min(weights)
print(f"      w_mean={w_mean:.4f}  w_spread={w_spread:.4f}  p_keep={p_keep:.4f}")
if w_spread < 0.05:
    print("      ⚠ WARNING: weights barely moved from p_keep — prior is flat!")
else:
    print("      ✅ Weights show spread — SB prior is informative")

# Free M1
del m1; gc.collect(); torch.cuda.empty_cache()

# Precompute token id lists for training
d2_c_ids = [tok.encode(p, add_special_tokens=True) + tok.encode(c, add_special_tokens=False)
            for p, c in zip(d2_prompts, d2_chosen)]
d2_r_ids = [tok.encode(p, add_special_tokens=True) + tok.encode(r, add_special_tokens=False)
            for p, r in zip(d2_prompts, d2_rejected)]
d2_c_ids = [x[-MAX_LEN:] for x in d2_c_ids]
d2_r_ids = [x[-MAX_LEN:] for x in d2_r_ids]
p_lens   = [min(len(tok.encode(p, add_special_tokens=True)), MAX_LEN) for p in d2_prompts]

# ─── Step 3: Train SB Stage 2 ─────────────────
print(f"\n[3/5] Training Soft Bayes Stage 2 ({TRAIN_STEPS} optimizer steps)...")
m_sb = load_base_with_s1(trainable=False)
m_sb = add_stage2(m_sb, "stage2")
m_sb.train()

opt_sb = torch.optim.AdamW([p for p in m_sb.parameters() if p.requires_grad], lr=LR, eps=1e-6)
GA = 4  # gradient accumulation
batch_size = 4
opt_sb.zero_grad()
step = 0; accum = 0

ref_c_t = torch.tensor(ref_c_lps, dtype=torch.float32, device=device)
ref_r_t = torch.tensor(ref_r_lps, dtype=torch.float32, device=device)
w_t     = torch.tensor(weights,   dtype=torch.float32, device=device)

indices = list(range(N_PAIRS))
import random; random.seed(SEED)
random.shuffle(indices)

for epoch in range(100):  # will hit max_steps early
    for bi in range(0, N_PAIRS, batch_size):
        bidx = indices[bi:bi+batch_size]
        if not bidx: continue
        c_ids_b = [d2_c_ids[i] for i in bidx]
        r_ids_b = [d2_r_ids[i] for i in bidx]
        p_b     = [p_lens[i]   for i in bidx]
        w_b     = w_t[bidx]
        rc_b    = ref_c_t[bidx]
        rr_b    = ref_r_t[bidx]

        pi_c = compute_batch_lp(m_sb, c_ids_b, p_b, pad_id)
        pi_r = compute_batch_lp(m_sb, r_ids_b, p_b, pad_id)
        loss = dpo_loss_soft(pi_c, pi_r, rc_b, rr_b, w_b) / GA
        loss.backward()
        accum += 1

        if accum >= GA:
            torch.nn.utils.clip_grad_norm_([p for p in m_sb.parameters() if p.requires_grad], 1.0)
            opt_sb.step(); opt_sb.zero_grad()
            accum = 0; step += 1
            if step % 5 == 0 or step <= 3:
                print(f"      [SB step {step}/{TRAIN_STEPS}] loss={loss.item()*GA:.4f}", flush=True)
            if step >= TRAIN_STEPS: break
    if step >= TRAIN_STEPS: break

m_sb.eval()
print(f"\n[4/5] Evaluating Soft Bayes Stage 2 (DPO implicit reward)...")
sb_acc = eval_s2_accuracy(m_sb, ref_c_lps, ref_r_lps, tok, test_rows)
print(f"      SB Stage 2 accuracy: {sb_acc:.4f}")
del m_sb; gc.collect(); torch.cuda.empty_cache()

# ─── Step 4: Train MLE-DPO Stage 2 ────────────
print(f"\n[4b/5] Training MLE-DPO Stage 2 ({TRAIN_STEPS} optimizer steps)...")
m_mle = load_base_with_s1(trainable=False)
m_mle = add_stage2(m_mle, "stage2")
m_mle.train()

opt_mle = torch.optim.AdamW([p for p in m_mle.parameters() if p.requires_grad], lr=LR, eps=1e-6)
opt_mle.zero_grad()
step = 0; accum = 0; random.shuffle(indices)

for epoch in range(100):
    for bi in range(0, N_PAIRS, batch_size):
        bidx = indices[bi:bi+batch_size]
        if not bidx: continue
        c_ids_b = [d2_c_ids[i] for i in bidx]
        r_ids_b = [d2_r_ids[i] for i in bidx]
        p_b     = [p_lens[i]   for i in bidx]
        rc_b    = ref_c_t[bidx]
        rr_b    = ref_r_t[bidx]

        pi_c = compute_batch_lp(m_mle, c_ids_b, p_b, pad_id)
        pi_r = compute_batch_lp(m_mle, r_ids_b, p_b, pad_id)
        loss = dpo_loss_hard(pi_c, pi_r, rc_b, rr_b) / GA
        loss.backward()
        accum += 1

        if accum >= GA:
            torch.nn.utils.clip_grad_norm_([p for p in m_mle.parameters() if p.requires_grad], 1.0)
            opt_mle.step(); opt_mle.zero_grad()
            accum = 0; step += 1
            if step % 5 == 0 or step <= 3:
                print(f"      [MLE step {step}/{TRAIN_STEPS}] loss={loss.item()*GA:.4f}", flush=True)
            if step >= TRAIN_STEPS: break
    if step >= TRAIN_STEPS: break

m_mle.eval()
print(f"\n[5/5] Evaluating MLE Stage 2 (DPO implicit reward)...")
mle_acc = eval_s2_accuracy(m_mle, ref_c_lps, ref_r_lps, tok, test_rows)
print(f"      MLE Stage 2 accuracy: {mle_acc:.4f}")

# ─── Results ──────────────────────────────────
print("\n" + "="*50)
print("          VERIFICATION RESULTS")
print("="*50)
print(f"  Stage 1   (M1, noisy D1):  {s1_acc:.4f}")
print(f"  MLE-DPO   (M2, noisy D2):  {mle_acc:.4f}  (delta={mle_acc-s1_acc:+.4f})")
print(f"  Soft Bayes (M2, noisy D2): {sb_acc:.4f}  (delta={sb_acc-s1_acc:+.4f})")
print("="*50)
if sb_acc > s1_acc and sb_acc > mle_acc:
    print("✅ SUCCESS: SB > Stage1 AND SB > MLE — fix is working!")
elif sb_acc > mle_acc:
    print("⚠ PARTIAL: SB > MLE but SB <= Stage1 — signal recovered vs MLE")
elif sb_acc > s1_acc:
    print("⚠ PARTIAL: SB > Stage1 but SB <= MLE — check beta/tau")
else:
    print("❌ FAIL: SB did not beat Stage1 or MLE — still broken")
