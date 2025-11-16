#!/bin/bash
# Direct training script for pythia-1b on Truthy-DPO (no qsub needed)
# Run with: bash train_truthydpo_pythia1b.sh
# Uses the same train_dpo_stage1.py logic as gpt2-medium, gpt2-small, and gpt2-large

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run fine-tuning ===
# Single GPU settings - try PROPS-2025 defaults first, fall back to memory-optimized if OOM
# Option 1: PROPS-2025 exact settings (if you have enough GPU memory ~20GB+)
# Option 2: Memory-optimized (if you get OOM errors)
echo "🚀 Starting single-GPU training..."
echo "   Using PROPS-2025 settings: bsz=4, ga=1, max_len=512"
echo "   If you get OOM errors, use the memory-optimized script instead"

python train_dpo_stage1.py \
  --model EleutherAI/pythia-1b \
  --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
  --out ./models/truthydpo/pythia-1b-dpo-truthydpo \
  --epochs 3 \
  --bsz 4 \
  --ga 1 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len 512

