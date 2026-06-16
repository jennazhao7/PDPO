#!/bin/bash
# LoRA-based training script for pythia-1b on HH-RLHF dataset
# LoRA is much more memory-efficient - only trains adapter weights (~1% of model)
# Run with: bash train_hhrlhf_pythia1b_lora.sh

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

# === Run LoRA fine-tuning ===
# LoRA settings (much more memory-efficient):
# - Only trains adapter weights (~1% of model parameters)
# - Allows larger batch sizes and sequence lengths
# - batch_size: 2 (bsz) - can be larger with LoRA
# - gradient_accumulation_steps: 4 (ga)
# - max_length: 256 (max-len) - can be longer with LoRA
# - LoRA rank: 16 (good balance of performance and memory)
# Effective batch size: 2 * 4 = 8
echo "🚀 Starting LoRA-based training pythia-1b on HH-RLHF dataset..."
echo "   Model: EleutherAI/pythia-1b"
echo "   Dataset: ./preprocessing/hhrlhf/dpo_train_ready.jsonl"
echo "   Output: ./models/hhrlhf/pythia-1b-dpo-lora-hhrlhf"
echo "   Settings: LoRA (rank=16), bsz=2, ga=4, max_len=256"

python train_dpo_lora.py \
  --model EleutherAI/pythia-1b \
  --data ./preprocessing/hhrlhf/dpo_train_ready.jsonl \
  --out ./models/hhrlhf/pythia-1b-dpo-lora-hhrlhf \
  --epochs 3 \
  --bsz 2 \
  --ga 4 \
  --max-prompt 128 \
  --max-target 128 \
  --max-len 256 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05

echo "✅ Training complete! Model saved to ./models/hhrlhf/pythia-1b-dpo-lora-hhrlhf"
echo "💡 Note: This is a LoRA adapter. To use it, load the base model and apply the adapter."

