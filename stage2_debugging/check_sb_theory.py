import json
import numpy as np
import torch
import torch.nn.functional as F

from datasets import load_dataset
ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
subset = ds.select(range(10000, 11000))

import importlib.util, sys
spec = importlib.util.spec_from_file_location("pku_mod", "experiments/core_result_tonight/01_secure_pku_saferlhf.py")
pku_mod = importlib.util.module_from_spec(spec)
sys.modules["pku_mod"] = pku_mod
spec.loader.exec_module(pku_mod)

clean_d2 = []
for row in subset:
    item = pku_mod.normalize_row(row)
    if item is not None:
        clean_d2.append(item)

noisy_d2 = []
with open("stage2_debugging/testsets/d2_1k_subset.jsonl") as f:
    for line in f:
        noisy_d2.append(json.loads(line))
noisy_d2 = noisy_d2[:500]
clean_d2 = clean_d2[:len(noisy_d2)]

is_flipped_list = []
for clean, noisy in zip(clean_d2, noisy_d2):
    assert clean["prompt"] == noisy["prompt"]
    is_flipped_list.append(clean["chosen"] != noisy["chosen"])

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

base_name = "Qwen/Qwen2.5-3B"
tok = AutoTokenizer.from_pretrained(base_name)
if not tok.pad_token: tok.pad_token = tok.eos_token
base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, device_map="auto")

m1 = PeftModel.from_pretrained(base_model, "stage2_debugging/stage1/results_instruct_eps1.0_seed42/pku_instruct_eps1.0_s42")
m1.eval()

def compute_logprob(model, tokenizer, prompt, response, max_length=512):
    enc = tokenizer(prompt + response, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    p_len = prompt_enc["input_ids"].shape[1]
    ids = enc["input_ids"].cuda()
    mask = enc["attention_mask"].cuda()
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    shift_logits = out.logits[0, :-1, :]
    shift_labels = ids[0, 1:]
    lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    tok_lp = lp[torch.arange(len(shift_labels)), shift_labels]
    return tok_lp[p_len-1:].sum().item() if len(tok_lp) >= p_len else 0.0

m1_scores = []
for p in tqdm(noisy_d2, desc="M1 Scoring"):
    lp_c = compute_logprob(m1, tok, p["prompt"], p["chosen"])
    lp_r = compute_logprob(m1, tok, p["prompt"], p["rejected"])
    with m1.disable_adapter():
        lp_c_base = compute_logprob(m1, tok, p["prompt"], p["chosen"])
        lp_r_base = compute_logprob(m1, tok, p["prompt"], p["rejected"])
    m1_scores.append((lp_c - lp_c_base) - (lp_r - lp_r_base))

m1_scores = np.array(m1_scores)
gamma_eps = 1 / (1 + np.exp(1.0))
q = 1 / (1 + np.exp(-m1_scores))
log_odds = np.log(q / (1 - q) + 1e-12) + np.log((1 - gamma_eps) / gamma_eps)
w = 1 / (1 + np.exp(-log_odds))

rr_correct = ~np.array(is_flipped_list)
m1_gt_correct = ((m1_scores > 0) & rr_correct) | ((m1_scores < 0) & (~rr_correct))

print(f"Total pairs: {len(noisy_d2)}")
print(f"RR noise rate (is_flipped): {(~rr_correct).mean():.1%}")
print(f"M1 agreement with RR: {(m1_scores > 0).mean():.1%}")
print(f"M1 true accuracy (vs clean): {m1_gt_correct.mean():.1%}")

print("\n--- Weight Analysis ---")
print(f"Mean weight for pairs where RR is RIGHT: {w[rr_correct].mean():.3f}")
print(f"Mean weight for pairs where RR is WRONG: {w[~rr_correct].mean():.3f}")

downweighted = w < 0.5
print(f"\nSB downweighted (<0.5) {downweighted.sum()} pairs.")
if downweighted.sum() > 0:
    print(f"Of the downweighted pairs, RR was actually RIGHT on: {rr_correct[downweighted].mean():.1%}")
    print(f"Of the downweighted pairs, M1 was actually WRONG on: {(~m1_gt_correct)[downweighted].mean():.1%}")

strongly_downweighted = w < 0.3
print(f"\nSB strongly downweighted (<0.3) {strongly_downweighted.sum()} pairs.")
if strongly_downweighted.sum() > 0:
    print(f"Of these, RR was actually RIGHT on: {rr_correct[strongly_downweighted].mean():.1%}")
