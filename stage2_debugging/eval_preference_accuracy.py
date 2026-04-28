#!/usr/bin/env python3
"""
Compute DPO implicit reward accuracy and ECE on test_pref.jsonl.

Metric: reward(y) = log π(y|x) - log π_ref(y|x)
  where π     = base + stage1 + stage2  (the trained policy)
  and   π_ref = base + stage1           (stage1 = frozen reference for stage2)

This is naturally length-normalized (reference sees same length pressure)
and matches exactly what the DPO training loss optimizes.

accuracy = Σ [reward(chosen) > reward(rejected)] / N
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


def _resolve_adapter_path(path, manifest_dir):
    import os
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    sub = os.path.join(path, "stage2")
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return sub
    return path


def load_model(manifest_path, device):
    import os
    m = json.load(open(manifest_path, "r", encoding="utf-8"))
    base = pick(m, ["base_model", "base_model_id", "base_model_name_or_path"])
    adapters = m.get("adapters", [])
    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    if len(adapters) == 1:
        s1 = None
        s2 = _resolve_adapter_path(adapters[0]["path"], manifest_dir)
    else:
        s1 = _resolve_adapter_path(adapters[0]["path"], manifest_dir)
        s2 = _resolve_adapter_path(adapters[1]["path"], manifest_dir)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if "mistral" in base.lower() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype)
    
    if s1:
        model = PeftModel.from_pretrained(model, s1, adapter_name="stage1")
        model.load_adapter(s2, adapter_name="stage2")
    else:
        # If there's only one adapter (e.g. map_retrain), just call it stage2
        model = PeftModel.from_pretrained(model, s2, adapter_name="stage2")
        
    model = model.to(device).eval()
    return model, tok, (s1 is not None)


@torch.no_grad()
def resp_logprob_sum(model, active_adapters, tok, prompt, resp, device, max_len=512):
    """Sum of token log-probs over response tokens only."""
    if active_adapters:
        try:
            model.set_adapter(active_adapters)
        except Exception:
            model.set_adapter(active_adapters[-1])
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


def dpo_reward(model, tok, prompt, resp, device, max_len, has_stage1: bool):
    """DPO implicit reward: log π(y|x) - log π_ref(y|x)."""
    if has_stage1:
        # policy = stage1 + stage2
        log_policy = resp_logprob_sum(model, ["stage1", "stage2"], tok, prompt, resp, device, max_len)
        # reference = stage1 only
        log_ref = resp_logprob_sum(model, "stage1", tok, prompt, resp, device, max_len)
    else:
        # policy = single adapter (we named it "stage2" above)
        log_policy = resp_logprob_sum(model, ["stage2"], tok, prompt, resp, device, max_len)
        # reference = base model (disable adapters)
        with model.disable_adapter():
            log_ref = resp_logprob_sum(model, [], tok, prompt, resp, device, max_len)
            
    return log_policy - log_ref


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
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--n_bins", type=int, default=10)
    args = ap.parse_args()

    rows = [json.loads(x) for x in open(args.test_jsonl, "r", encoding="utf-8") if x.strip()]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, has_stage1 = load_model(args.manifest, device)

    margins, confs, corr = [], [], []
    for i, r in enumerate(rows):
        reward_c = dpo_reward(model, tok, r["prompt"], r["chosen"], device, args.max_len, has_stage1)
        reward_r = dpo_reward(model, tok, r["prompt"], r["rejected"], device, args.max_len, has_stage1)
        m = reward_c - reward_r
        margins.append(m)
        confs.append(sigmoid(m))
        corr.append(1 if m > 0 else 0)
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(rows)}] running_acc={sum(corr)/len(corr):.3f}", flush=True)

    acc = sum(corr) / len(corr)
    ece, bins = compute_ece(confs, corr, args.n_bins)

    out = {
        "manifest": args.manifest,
        "metric": "dpo_implicit_reward",
        "n": len(rows),
        "accuracy": acc,
        "ece": ece,
        "mean_margin": sum(margins) / len(margins),
        "bins": bins,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"accuracy": acc, "ece": ece, "n": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
