#!/bin/bash
# Direct training script for gpt2-small (no qsub needed)
# Run with: bash train_gpt2_small.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run fine-tuning ===
python train_dpo_stage1.py \
  --model gpt2 \
  --data ./dpo_train_ready.jsonl \
  --out ./models/M1_small \
  --epochs 3 \
  --bsz 2 \
  --ga 4 \
  --max-prompt 64 \
  --max-target 64 \
  --max-len 128

