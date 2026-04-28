import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

device = 'cuda'

mbase = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b', torch_dtype=torch.float16)
mbase.config.use_cache = False
tok = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

prompt = "Do you like to play chess?"
resp_chosen = " Yes I play chess every day."
resp_rej = " No I hate board games."

def get_logp(m, p, r):
    pid = tok(p, add_special_tokens=False)['input_ids']
    rid = tok(r, add_special_tokens=False)['input_ids']
    x = torch.tensor([(pid + rid)[-512:]], device=device)
    plen = min(len(pid), len(x[0]) - 1)
    with torch.no_grad():
        lp = torch.log_softmax(m(input_ids=x).logits[:, :-1, :], dim=-1)
    tok_lp = lp.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    return float(tok_lp[max(plen-1, 0):].sum().item())

def margin(m):
    return get_logp(m, prompt, resp_chosen) - get_logp(m, prompt, resp_rej)

print("Base Margin:", margin(mbase.to(device)))

m1 = PeftModel.from_pretrained(mbase, '/users/jzhao7/PDPO/outputs_openllama_tonight/stage1/EleutherAI--pythia-1b_stage1_rr_eps1.0_seed42', adapter_name='stage1')
m1.set_adapter('stage1')
print("M1 Margin:", margin(m1.to(device)))

m1.load_adapter('/users/jzhao7/PDPO/outputs_openllama_tonight/stage2_mle_fix/EleutherAI--pythia-1b_stage2_mle_eps1.0_seed42_fix/stage2', adapter_name='stage2_mle')
m1.set_adapter('stage2_mle')
print("MLE Margin:", margin(m1.to(device)))

m1.load_adapter('/users/jzhao7/PDPO/outputs_openllama_tonight/stage2_softbayes/EleutherAI--pythia-1b_stage2_sb_eps1.0_seed42/stage2', adapter_name='stage2_sb')
m1.set_adapter('stage2_sb')
print("SB Margin:", margin(m1.to(device)))
