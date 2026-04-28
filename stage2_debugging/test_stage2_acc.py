import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

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
    
    start = max(p_len - 1, 0)
    return float(tok_lp[start:].sum().item())

def dpo_implicit_reward(model, base_model, tokenizer, prompt, response, beta=0.1):
    lp_m1 = compute_logprob(model, tokenizer, prompt, response)
    with model.disable_adapter():
        lp_base = compute_logprob(model, tokenizer, prompt, response)
    return beta * (lp_m1 - lp_base)

test_data = []
with open("stage2_debugging/testsets/pku_secure/test_pref.jsonl") as f:
    for line in f:
        test_data.append(json.loads(line))

# The test set has 1000 items
base_name = "Qwen/Qwen2.5-3B"
tok = AutoTokenizer.from_pretrained(base_name)
if not tok.pad_token: tok.pad_token = tok.eos_token

base = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, device_map="auto")

models_to_test = [
    ("MAP-A", "stage2_debugging/stage2_d3_pku/map_retrain_instruct_m1_on_d2/checkpoint-300"),
    ("SB-A", "stage2_debugging/stage2_d3_pku/sb_fresh_instruct_m1_on_d2/checkpoint-step-300/stage2"),
    ("MAP-B", "stage2_debugging/stage2_d3_pku/map_retrain_d1d2_m1_on_d3/checkpoint-300"),
    ("SB-B", "stage2_debugging/stage2_d3_pku/sb_fresh_d1d2_m1_on_d3/checkpoint-step-300/stage2")
]

results = {}
for name, path in models_to_test:
    try:
        model = PeftModel.from_pretrained(base, path)
        model.eval()
        correct = 0
        for s in tqdm(test_data, desc=name):
            r_c = dpo_implicit_reward(model, base, tok, s["prompt"], s["chosen"])
            r_r = dpo_implicit_reward(model, base, tok, s["prompt"], s["rejected"])
            if r_c > r_r:
                correct += 1
        acc = correct / len(test_data)
        results[name] = acc
        print(f"{name}: {acc*100:.1f}%")
        del model
    except Exception as e:
        print(f"Error on {name}: {e}")

print("FINAL RESULTS")
for k, v in results.items():
    print(f"{k}: {v*100:.1f}%")
