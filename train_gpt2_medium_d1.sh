#!/bin/bash
# Training script for GPT2-medium on TruthyDPO D1 dataset
# Uses the first half (D1) of the randomly split dataset
# Run with: bash train_gpt2_medium_d1.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/pdpo_env.sh
source "${SCRIPT_DIR}/scripts/pdpo_env.sh"
cd "${PDPO_ROOT}"

# === Check if D1 dataset exists ===
D1_DATA="./preprocessing/truthydpo/dpo_train_ready_d1.jsonl"
if [ ! -f "$D1_DATA" ]; then
    echo "❌ Error: D1 dataset not found at $D1_DATA"
    echo ""
    echo "Please create D1 and D2 splits first:"
    echo "  cd preprocessing/truthydpo"
    echo "  python split_d1_d2.py"
    exit 1
fi

# === Run fine-tuning ===
echo "🚀 Starting GPT2-medium training on D1 dataset..."
echo "   Model: gpt2-medium"
echo "   Dataset: $D1_DATA"
echo "   Output: ./models/truthydpo/gpt2-medium-dpo-truthydpo-d1"
echo ""

python train_dpo_stage1.py \
  --model gpt2-medium \
  --data "$D1_DATA" \
  --out ./models/truthydpo/gpt2-medium-dpo-truthydpo-d1 \
  --epochs 3 \
  --bsz 4 \
  --ga 1 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len 512

echo ""
echo "✅ Training complete! Model saved to ./models/truthydpo/gpt2-medium-dpo-truthydpo-d1"


