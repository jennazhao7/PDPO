#!/bin/bash
# Training script for GPT2-medium M2 model on D2 + ℓ_PROPS labels
# M2 is trained starting from M1 (trained on D1)
# Run with: bash train_gpt2_medium_m2.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Configuration ===
D2_PROPS_DATA="./preprocessing/truthydpo/d2_props_labels.jsonl"
M1_MODEL="./models/truthydpo/gpt2-medium-dpo-truthydpo-d1"
M2_OUTPUT="./models/truthydpo/gpt2-medium-dpo-truthydpo-m2"

# === Check if required files exist ===
if [ ! -f "$D2_PROPS_DATA" ]; then
    echo "❌ Error: D2 PROPS labels not found at $D2_PROPS_DATA"
    echo ""
    echo "Please generate PROPS labels first:"
    echo "  1. Extract RR labels: python preprocessing/truthydpo/extract_rr_labels_d2.py"
    echo "  2. Generate M1 labels: python preprocessing/truthydpo/generate_m1_labels_d2.py"
    echo "  3. Combine labels: python preprocessing/truthydpo/combine_labels_mle.py"
    exit 1
fi

if [ ! -d "$M1_MODEL" ]; then
    echo "❌ Error: M1 model not found at $M1_MODEL"
    echo ""
    echo "Please train M1 first:"
    echo "  bash train_gpt2_medium_d1.sh"
    exit 1
fi

# === Run fine-tuning ===
echo "🚀 Starting GPT2-medium M2 training on D2 + ℓ_PROPS..."
echo "   Base model (M1): $M1_MODEL"
echo "   Dataset: $D2_PROPS_DATA"
echo "   Output: $M2_OUTPUT"
echo ""

python train_dpo_stage1.py \
  --model "$M1_MODEL" \
  --data "$D2_PROPS_DATA" \
  --out "$M2_OUTPUT" \
  --epochs 3 \
  --bsz 4 \
  --ga 1 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len 512

echo ""
echo "✅ M2 training complete! Model saved to $M2_OUTPUT"


