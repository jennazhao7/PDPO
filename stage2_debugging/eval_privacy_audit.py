#!/usr/bin/env python3
import argparse
import json
import random
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
def roc_auc_score(y_true, y_score):
    n1 = sum(y_true)
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0: return 0.5
    
    data = sorted(zip(y_true, y_score), key=lambda x: x[1])
    rank_sum = 0.0
    
    i = 0
    while i < len(data):
        j = i
        while j < len(data) and data[j][1] == data[i][1]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            if data[k][0] == 1:
                rank_sum += rank
        i = j
        
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def pick(d, keys):
    for k in keys:
        if k in d and d[k]:
            return d[k]
    return None

def _resolve_adapter_path(path, manifest_dir):
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return path
    sub = os.path.join(path, "stage2")
    if os.path.exists(os.path.join(sub, "adapter_config.json")):
        return sub
    return path

def load_model(manifest_path, device):
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
        model = PeftModel.from_pretrained(model, s2, adapter_name="stage2")
        
    model = model.to(device).eval()
    return model, tok, (s1 is not None)

@torch.no_grad()
def resp_logprob_sum(model, active_adapters, tok, prompt, resp, device, max_len=512):
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
    if has_stage1:
        log_policy = resp_logprob_sum(model, ["stage1", "stage2"], tok, prompt, resp, device, max_len)
        log_ref = resp_logprob_sum(model, "stage1", tok, prompt, resp, device, max_len)
    else:
        log_policy = resp_logprob_sum(model, ["stage2"], tok, prompt, resp, device, max_len)
        with model.disable_adapter():
            log_ref = resp_logprob_sum(model, [], tok, prompt, resp, device, max_len)
    return log_policy - log_ref

def get_delta(model, tok, r, device, max_len, has_stage1):
    c = dpo_reward(model, tok, r["prompt"], r["chosen"], device, max_len, has_stage1)
    r_ = dpo_reward(model, tok, r["prompt"], r["rejected"], device, max_len, has_stage1)
    return c - r_

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--member_jsonl", required=True)
    ap.add_argument("--nonmember_jsonl", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--subset_size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args()

    random.seed(args.seed)

    members = [json.loads(x) for x in open(args.member_jsonl, "r", encoding="utf-8") if x.strip()]
    if len(members) > args.subset_size:
        members = random.sample(members, args.subset_size)
    else:
        args.subset_size = len(members)

    nonmembers = [json.loads(x) for x in open(args.nonmember_jsonl, "r", encoding="utf-8") if x.strip()]
    if len(nonmembers) > args.subset_size:
        nonmembers = random.sample(nonmembers, args.subset_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, has_stage1 = load_model(args.manifest, device)

    member_deltas = []
    print("Scoring members...")
    for i, r in enumerate(members):
        member_deltas.append(get_delta(model, tok, r, device, args.max_len, has_stage1))

    nonmember_deltas = []
    print("Scoring nonmembers...")
    for i, r in enumerate(nonmembers):
        nonmember_deltas.append(get_delta(model, tok, r, device, args.max_len, has_stage1))

    labels = [1] * len(member_deltas) + [0] * len(nonmember_deltas)
    scores = member_deltas + nonmember_deltas
    auc = roc_auc_score(labels, scores)

    out = {
        "manifest": args.manifest,
        "metric": "mia_auc_roc",
        "n_members": len(member_deltas),
        "n_nonmembers": len(nonmember_deltas),
        "mean_member_margin": sum(member_deltas) / len(member_deltas),
        "mean_nonmember_margin": sum(nonmember_deltas) / len(nonmember_deltas),
        "auc": auc
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
