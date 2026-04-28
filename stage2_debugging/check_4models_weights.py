import torch
import numpy as np
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

def diagnose_weights(m1_scores_on_d2, epsilon=1.0, name='Model'):
    print(f'=== {name} ===')
    m1_scores_on_d2 = np.array(m1_scores_on_d2)
    gamma_eps = 1 / (1 + np.exp(epsilon))
    q = 1 / (1 + np.exp(-m1_scores_on_d2))
    log_odds = np.log(q / (1 - q) + 1e-12) + np.log((1 - gamma_eps) / gamma_eps)
    w = 1 / (1 + np.exp(-log_odds))
    
    print(f'w: mean={w.mean():.3f}, std={w.std():.3f}')
    print(f'  > 0.7 (trust label):  {(w > 0.7).mean():.1%}')
    print(f'  < 0.3 (down-weight):  {(w < 0.3).mean():.1%}')
    print(f'  [0.3-0.7] (ambiguous):{((w>=0.3)&(w<=0.7)).mean():.1%}')
    print()

def compute_logprob(model, tokenizer, prompt, response, max_length=512):
    enc = tokenizer(prompt + response, return_tensors='pt', truncation=True, max_length=max_length)
    prompt_enc = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    p_len = prompt_enc['input_ids'].shape[1]
    
    ids = enc['input_ids'].cuda()
    mask = enc['attention_mask'].cuda()
    
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    
    shift_logits = out.logits[0, :-1, :]
    shift_labels = ids[0, 1:]
    
    lp = torch.nn.functional.log_softmax(shift_logits, dim=-1)
    tok_lp = lp[torch.arange(len(shift_labels)), shift_labels]
    
    return tok_lp[p_len-1:].mean().item() if len(tok_lp) >= p_len else 0.0

pairs = []
with open('stage2_debugging/testsets/d2_1k_subset.jsonl') as f:
    for line in f:
        pairs.append(json.loads(line))
pairs = pairs[:500] # Use 500 for speed

base_name = 'Qwen/Qwen2.5-3B'
tok = AutoTokenizer.from_pretrained(base_name)
if not tok.pad_token: tok.pad_token = tok.eos_token

base_model = AutoModelForCausalLM.from_pretrained(base_name, torch_dtype=torch.float16, device_map='auto')

models = [
    ('Base D1+D2', 'stage2_debugging/stage1/results_base_d1d2_eps1.0_seed42/pku_base_d1d2_eps1.0_s42'),
    ('Base Longepoch', 'stage2_debugging/stage1/results_base_longepoch_eps1.0_seed42/pku_base_longepoch_eps1.0_s42'),
    ('Base R64', 'stage2_debugging/stage1/results_base_r64_eps1.0_seed42/pku_base_r64_eps1.0_s42'),
    ('Instruct', 'stage2_debugging/stage1/results_instruct_eps1.0_seed42/pku_instruct_eps1.0_s42'),
]

for name, path in models:
    try:
        m1 = PeftModel.from_pretrained(base_model, path)
        m1.eval()
        deltas = []
        for p in tqdm(pairs, desc=name):
            lp_c = compute_logprob(m1, tok, p['prompt'], p['chosen'])
            lp_r = compute_logprob(m1, tok, p['prompt'], p['rejected'])
            deltas.append(lp_c - lp_r)
        diagnose_weights(deltas, 1.0, name)
        del m1
    except Exception as e:
        print(f"Error evaluating {name}: {e}")

