#!/bin/bash
# Direct training script for gpt2-medium (no qsub needed)
# Run with: bash train_gpt2_medium.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run fine-tuning ===
python train_dpo_stage1.py \
  --model gpt2-medium \
  --data ./dpo_train_ready.jsonl \
  --out ./models/M1_medium \
  --epochs 3 \
  --bsz 2 \
  --ga 4 \
  --max-prompt 64 \
  --max-target 64 \
  --max-len 128

