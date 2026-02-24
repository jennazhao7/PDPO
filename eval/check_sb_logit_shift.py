#!/usr/bin/env python3
"""Check SB logit shift: compare margin between MLE and SB models. Expected mean delta ~1.0 for eps=1.0."""
import argparse
import json
import math
import os
import random
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None

def _resolve(path):
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    sub = os.path.join(path, "stage2")
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return sub
    return path

def load_stage2(manifest_path, device):
    m = json.load(open(manifest_path, "r", encoding="utf-8"))
    base = pick(m, ["base_model", "base_model_id", "base_model_name_or_path"])
    adapters = m.get("adapters", [])
    if not adapters or len(adapters) < 2:
        raise ValueError("Manifest needs 2 adapters")
    s1, s2 = _resolve(adapters[0]["path"]), _resolve(adapters[1]["path"])
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(model, s1)
    model = PeftModel.from_pretrained(model, s2)
    return model.to(device).eval(), tok

@torch.no_grad()
def resp_logprob(model, tok, prompt, resp, device, max_len=512):
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"]
    ids = (p + r)[-max_len:]
    if len(ids) < 2:
        return 0.0
    p_len = min(len(p), len(ids) - 1)
    x = torch.tensor([ids], device=device)
    logp = torch.log_softmax(model(input_ids=x).logits[:, :-1, :], dim=-1)
    target = x[:, 1:]
    tok_lp = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)[0]
    return float(tok_lp[max(p_len - 1, 0):].sum().item())

def margin(model, tok, row, device, max_len):
    return resp_logprob(model, tok, row["prompt"], row["chosen"], device, max_len) - resp_logprob(model, tok, row["prompt"], row["rejected"], device, max_len)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mle_manifest", required=True)
    ap.add_argument("--sb_manifest", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()
    rows = [json.loads(x) for x in open(args.test_jsonl, "r", encoding="utf-8") if x.strip()]
    if args.n > 0 and args.n < len(rows):
        rows = random.Random(args.seed).sample(rows, args.n)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mle, tok_m = load_stage2(args.mle_manifest, device)
    sb, tok_s = load_stage2(args.sb_manifest, device)
    deltas = [margin(sb, tok_s, r, device, args.max_len) - margin(mle, tok_m, r, device, args.max_len) for r in rows]
    mean_delta = sum(deltas) / len(deltas)
    expected = math.log(0.731 / 0.269)
    print(f"n={len(deltas)}")
    print(f"mean_delta_margin_SB_minus_MLE={mean_delta:.6f}")
    print(f"expected_approx={expected:.6f}")

if __name__ == "__main__":
    main()
