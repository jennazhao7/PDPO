import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16, device_map="cpu")
m1 = PeftModel.from_pretrained(base, "outputs_new_models/stage1/pku_eps1.0_s42")
print("M1 loaded")
m1 = m1.merge_and_unload()
print("M1 merged")
cfg = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
m2 = get_peft_model(m1, cfg)
print("M2 created with new adapter")
print("Trainable params:", sum(p.numel() for p in m2.parameters() if p.requires_grad))
