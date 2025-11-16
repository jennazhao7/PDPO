#!/bin/bash
# Direct training script for gpt2-medium on Truthy-DPO (no qsub needed)
# Run with: bash train_gpt2_medium.sh
# Parameters match PROPS-2025 config.yaml settings:
# - batch_size: 4 (bsz)
# - gradient_accumulation_steps: 1 (ga)
# - max_length: 512 (max-len)
# - max_prompt_length: 256 (max-prompt)
# - max_target_length: ~256 (max-target, inferred from max_length - max_prompt)

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run fine-tuning ===
python train_dpo_stage1.py \
  --model gpt2-medium \
  --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
  --out ./models/truthydpo/gpt2-medium-dpo-truthydpo \
  --epochs 3 \
  --bsz 4 \
  --ga 1 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len 512

