#!/bin/bash
# Memory-optimized training script for pythia-1b on HH-RLHF dataset
# Optimized for 24GB GPU
# Run with: bash train_hhrlhf_pythia1b.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pdpo_env.sh
source "${SCRIPT_DIR}/scripts/pdpo_env.sh"
cd "${PDPO_ROOT}"

# === Clear GPU memory ===
echo "🧹 Clearing GPU memory..."
python clear_gpu_memory.py 2>/dev/null || echo "   (GPU memory clearing skipped)"

# === Run fine-tuning ===
# Ultra memory-optimized settings for 24GB GPU (after OOM):
# - batch_size: 1 (bsz) - minimal batch size
# - gradient_accumulation_steps: 8 (ga) - increased to maintain effective batch size = 8
# - max_length: 128 (max-len) - further reduced to save memory
# - max_prompt_length: 64 (max-prompt) - reduced from 128
# - max_target_length: 64 (max-target) - reduced from 128
# Effective batch size: 1 * 8 = 8 (maintains training stability)
echo "🚀 Starting ultra memory-optimized training pythia-1b on HH-RLHF dataset..."
echo "   Model: EleutherAI/pythia-1b"
echo "   Dataset: ./preprocessing/hhrlhf/dpo_train_ready.jsonl"
echo "   Output: ./models/hhrlhf/pythia-1b-dpo-hhrlhf"
echo "   Settings: bsz=1, ga=8, max_len=128 (ultra memory-optimized for 24GB GPU)"

python train_dpo_stage1.py \
  --model EleutherAI/pythia-1b \
  --data ./preprocessing/hhrlhf/dpo_train_ready.jsonl \
  --out ./models/hhrlhf/pythia-1b-dpo-hhrlhf \
  --epochs 3 \
  --bsz 1 \
  --ga 8 \
  --max-prompt 64 \
  --max-target 64 \
  --max-len 128

echo "✅ Training complete! Model saved to ./models/hhrlhf/pythia-1b-dpo-hhrlhf"

