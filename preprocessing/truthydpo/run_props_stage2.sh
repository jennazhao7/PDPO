#!/bin/bash
# PROPS Stage 2 Pipeline: Generate ℓ_RR, ℓ_M1, combine to ℓ_PROPS, train M2
# Run this after M1 is trained on D1
# Usage: bash preprocessing/truthydpo/run_props_stage2.sh

set -e  # Exit on error

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "🚀 PROPS Stage 2 Pipeline"
echo "========================"
echo ""

# Configuration (using absolute paths)
BASE_DIR="/users/jzhao7/PDPO"
PREPROC_DIR="$BASE_DIR/preprocessing/truthydpo"
MODELS_DIR="$BASE_DIR/models/truthydpo"

D2_DATA="$PREPROC_DIR/dpo_train_ready_d2.jsonl"
PRIVATIZED_DATA="$PREPROC_DIR/dpo_privatized_dataset.jsonl"
M1_MODEL="$MODELS_DIR/gpt2-medium-dpo-truthydpo-d1"
RM_MODEL="OpenAssistant/reward-model-deberta-v3-large-v2"

# Output files
RR_LABELS="$PREPROC_DIR/d2_rr_labels.jsonl"
M1_LABELS="$PREPROC_DIR/d2_m1_labels.jsonl"
PROPS_LABELS="$PREPROC_DIR/d2_props_labels.jsonl"

# Step 1: Extract RR labels from D2 (already margin-aware flipped)
echo "📝 Step 1: Extracting RR labels (ℓ_RR) from D2..."
if [ ! -f "$D2_DATA" ]; then
    echo "❌ Error: D2 dataset not found at $D2_DATA"
    echo "   Please create D1/D2 split first: python split_d1_d2.py"
    exit 1
fi

python "$PREPROC_DIR/extract_rr_labels_d2.py" \
    --d2_file "$D2_DATA" \
    --privatized_file "$PRIVATIZED_DATA" \
    --output "$RR_LABELS"

echo ""

# Step 2: Generate M1 labels for D2
echo "📝 Step 2: Generating M1 labels (ℓ_M1) for D2..."
if [ ! -d "$M1_MODEL" ]; then
    echo "❌ Error: M1 model not found at $M1_MODEL"
    echo "   Please train M1 first: bash ../train_gpt2_medium_d1.sh"
    exit 1
fi

python "$PREPROC_DIR/generate_m1_labels_d2.py" \
    --d2_file "$D2_DATA" \
    --m1_model "$M1_MODEL" \
    --output "$M1_LABELS" \
    --rm_model "$RM_MODEL" \
    --device cuda

echo ""

# Step 3: Combine labels using MLE rule
echo "📝 Step 3: Combining ℓ_RR and ℓ_M1 using MLE rule to get ℓ_PROPS..."
python "$PREPROC_DIR/combine_labels_mle.py" \
    --rr_file "$RR_LABELS" \
    --m1_file "$M1_LABELS" \
    --output "$PROPS_LABELS" \
    --gamma_eps 0.1

echo ""

# Step 4: Train M2 on D2 + ℓ_PROPS
echo "📝 Step 4: Training M2 on D2 + ℓ_PROPS starting from M1..."
bash "$BASE_DIR/train_gpt2_medium_m2.sh"

echo ""
echo "✅ PROPS Stage 2 Pipeline Complete!"
echo "   M2 model saved to: $MODELS_DIR/gpt2-medium-dpo-truthydpo-m2"

