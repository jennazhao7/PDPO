#!/bin/bash
# Direct training script for gpt2-large on Truthy-DPO (no qsub needed)
# Run with: bash train_truthydpo_gpt2large.sh
# Uses the same train_dpo_stage1.py logic as gpt2-medium and gpt2-small

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run fine-tuning ===
python train_dpo_stage1.py \
  --model gpt2-large \
  --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
  --out ./models/truthydpo/gpt2-large-dpo-truthydpo \
  --epochs 3 \
  --bsz 2 \
  --ga 4 \
  --max-prompt 64 \
  --max-target 64 \
  --max-len 128

