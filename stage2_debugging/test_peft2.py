import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16, device_map="cpu")
m1 = PeftModel.from_pretrained(base, "outputs_new_models/stage1/pku_eps1.0_s42")
m1 = m1.merge_and_unload()
cfg = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
m2 = get_peft_model(m1, cfg, adapter_name="stage2")
print("M2 adapters:", list(m2.peft_config.keys()))
