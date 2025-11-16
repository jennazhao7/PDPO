#!/bin/bash
# Single-GPU memory-optimized training script for pythia-1b
# Use this if you get OOM errors with the standard script
# Usage: bash train_truthydpo_pythia1b_single_gpu_memory_opt.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run fine-tuning ===
# Memory-optimized settings (reduced from PROPS-2025 defaults):
# - batch_size: 1 (bsz) - reduced from 4 to save memory
# - gradient_accumulation_steps: 4 (ga) - increased to maintain effective batch size = 4
# - max_length: 256 (max-len) - reduced from 512
# - max_prompt_length: 128 (max-prompt) - reduced from 256
# - max_target_length: 128 (max-target) - reduced from 256
# Effective batch size: 1 * 4 = 4 (same as PROPS-2025)
echo "🚀 Starting single-GPU memory-optimized training..."
echo "   Settings: bsz=1, ga=4, max_len=256 (reduced for memory)"

python train_dpo_stage1.py \
  --model EleutherAI/pythia-1b \
  --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
  --out ./models/truthydpo/pythia-1b-dpo-truthydpo \
  --epochs 3 \
  --bsz 1 \
  --ga 4 \
  --max-prompt 128 \
  --max-target 128 \
  --max-len 256

