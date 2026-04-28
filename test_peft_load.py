import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

device = "cuda"

s1 = "/users/jzhao7/PDPO/outputs_openllama_tonight/stage1/EleutherAI--pythia-1b_stage1_rr_eps1.0_seed42"
s2_mle = "/users/jzhao7/PDPO/outputs_openllama_tonight/stage2_mle_fix/EleutherAI--pythia-1b_stage2_mle_eps1.0_seed42_fix/stage2"
s2_sb = "/users/jzhao7/PDPO/outputs_openllama_tonight/stage2_softbayes/EleutherAI--pythia-1b_stage2_sb_eps1.0_seed42/stage2"

m_base = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b", torch_dtype=torch.float16)
m_mle = PeftModel.from_pretrained(m_base, s1)
m_mle = PeftModel.from_pretrained(m_mle, s2_mle)

for name, param in m_mle.named_parameters():
    if "lora" in name:
        print("MLE param", name, param.sum().item())
        break

m_base2 = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-1b", torch_dtype=torch.float16)
m_sb = PeftModel.from_pretrained(m_base2, s1)
m_sb = PeftModel.from_pretrained(m_sb, s2_sb)

for name, param in m_sb.named_parameters():
    if "lora" in name:
        print("SB param", name, param.sum().item())
        break
