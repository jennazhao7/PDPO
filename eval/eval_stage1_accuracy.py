#!/usr/bin/env python3
"""
Compute DPO implicit reward accuracy on the Stage 1 (M1) model alone.

Metric: reward(y) = log π_S1(y|x) - log π_base(y|x)
  where π_S1  = base + stage1   (the policy: Stage1-trained model)
  and   π_base = base model only (the reference: pre-DPO model)

This is naturally length-normalized and matches what Stage1 DPO training optimized.
"""
import argparse
import json
import math
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None


def load_models(base_model_id, stage1_adapter_path, device):
    dtype = torch.bfloat16 if "mistral" in base_model_id.lower() else torch.float16
    tok = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Load base model (reference)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
    base = base.to(device).eval()

    # Load policy = base + stage1
    policy = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=dtype)
    policy = PeftModel.from_pretrained(policy, stage1_adapter_path, adapter_name="stage1")
    policy.set_adapter("stage1")
    policy = policy.to(device).eval()

    return base, policy, tok


@torch.no_grad()
def resp_logprob_sum(model, tok, prompt, resp, device, max_len=512):
    """Sum of token log-probs over response tokens only."""
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"]
    ids = p + r
    if len(ids) < 2:
        return 0.0
    ids = ids[-max_len:]
    p_len = min(len(p), len(ids) - 1)
    x = torch.tensor([ids], device=device)
    out = model(input_ids=x)
    logp = torch.log_softmax(out.logits[:, :-1, :], dim=-1)
    target = x[:, 1:]
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    start = max(p_len - 1, 0)
    return float(tok_lp[start:].sum().item())


def dpo_reward(policy, base, tok, prompt, resp, device, max_len):
    """DPO implicit reward: log π_S1(y|x) - log π_base(y|x)."""
    return (resp_logprob_sum(policy, tok, prompt, resp, device, max_len) -
            resp_logprob_sum(base, tok, prompt, resp, device, max_len))


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def compute_ece(conf, correct, n_bins=10):
    ece = 0.0
    bins = []
    n = len(conf)
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        idx = [i for i, c in enumerate(conf) if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            bins.append({"bin": b, "count": 0, "acc": None, "conf": None})
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        cavg = sum(conf[i] for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(acc - cavg)
        bins.append({"bin": b, "count": len(idx), "acc": acc, "conf": cavg})
    return ece, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True, help="HuggingFace model ID for the base model")
    ap.add_argument("--stage1_adapter", required=True, help="Path to stage1 LoRA adapter folder")
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.test_jsonl, "r", encoding="utf-8") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, policy, tok = load_models(args.base_model, args.stage1_adapter, device)

    margins, confs, corr = [], [], []
    for i, r in enumerate(rows):
        reward_c = dpo_reward(policy, base, tok, r["prompt"], r["chosen"], device, args.max_len)
        reward_r = dpo_reward(policy, base, tok, r["prompt"], r["rejected"], device, args.max_len)
        m = reward_c - reward_r
        margins.append(m)
        confs.append(sigmoid(m))
        corr.append(1 if m > 0 else 0)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] running_acc={sum(corr)/len(corr):.3f}", flush=True)

    acc = sum(corr) / len(corr)
    ece, bins = compute_ece(confs, corr, args.n_bins)
    mean_margin = sum(margins) / len(margins)

    out = {
        "model": args.base_model,
        "stage1_adapter": args.stage1_adapter,
        "metric": "dpo_implicit_reward",
        "stage": "stage1",
        "n": len(rows),
        "accuracy": acc,
        "ece": ece,
        "mean_margin": mean_margin,
        "bins": bins,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"accuracy": acc, "ece": ece, "mean_margin": mean_margin, "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
