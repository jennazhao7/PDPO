#!/usr/bin/env python3
import math, sys, torch
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset

print("=" * 60)
print("  rDPO test: loss_type='robust' with label_smoothing")
print("=" * 60)

ds = Dataset.from_dict({
    "prompt":   ["Is this a test?"] * 4,
    "chosen":   [" Yes, this is a test."] * 4,
    "rejected": [" No, not a test."] * 4,
})

MODEL = "gpt2"
tok = AutoTokenizer.from_pretrained(MODEL)
tok.pad_token = tok.eos_token; tok.padding_side = "left"

SHARED = dict(
    output_dir="./rdpo_sanity",
    per_device_train_batch_size=2, max_steps=1,
    learning_rate=0.0, logging_steps=1, report_to=None,
    remove_unused_columns=False, dataloader_num_workers=0,
    beta=0.1,
    max_length=64, max_prompt_length=32, max_completion_length=32,
    save_strategy="no", no_cuda=True, use_cpu=True,
)

def get_loss(loss_type, label_smoothing):
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    m.config.use_cache = False
    cfg = DPOConfig(**SHARED, loss_type=loss_type, label_smoothing=label_smoothing)
    trainer = DPOTrainer(model=m, ref_model=None, processing_class=tok, args=cfg, train_dataset=ds)
    batch = next(iter(trainer.get_train_dataloader()))
    batch = {k: v.to("cpu") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    m.train()
    with torch.no_grad():
        return trainer.compute_loss(m, batch).item()

gamma_5 = 1.0 / (1.0 + math.exp(0.5))

loss_dpo      = get_loss("sigmoid", 0.0)
loss_rdpo_g1  = get_loss("robust", 0.0001)
loss_rdpo_g5  = get_loss("robust", gamma_5)

scaling_factor_5 = 1.0 / (1.0 - 2.0 * gamma_5)

print(f"\nStandard DPO   (sigmoid, smoothing=0.000): {loss_dpo:.6f}")
print(f"rDPO           (robust,  smoothing=0.0001): {loss_rdpo_g1:.6f}")
print(f"rDPO           (robust,  smoothing={gamma_5:.3f}): {loss_rdpo_g5:.6f}")
print(f"Expected start for smoothing={gamma_5:.3f} is ~ {loss_dpo * scaling_factor_5:.6f}\n")

if loss_rdpo_g5 > loss_dpo * 2.0:
    print("✅ PASS — rDPO scaling is active")
    sys.exit(0)
else:
    print("❌ FAIL — rDPO scaling not observed")
    sys.exit(1)
