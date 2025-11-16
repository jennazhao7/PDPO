#!/bin/bash
# Master script to run margin-aware label flipping for Alpaca dataset
# This script follows the same pattern as truthydpo preprocessing:
# 0. Convert alpaca_eval to DPO format (if needed)
# 1. Score Alpaca dataset with reward model
# 2. Process margins (clip and normalize)
# 3. Privatize labels using margin-aware flipping

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Configuration ===
DATASET="${1:-tatsu-lab/alpaca_eval}"  # Alpaca dataset name (default: alpaca_eval)
SPLIT="${2:-eval}"                     # Dataset split
EPS_LABELS="${3:-1.0}"                # Privacy parameter for label flipping
BATCH_SIZE="${4:-32}"                 # Batch size for reward model scoring
CONVERT_FIRST="${5:-true}"            # Whether to convert alpaca_eval first (default: true)

echo "=========================================="
echo "Alpaca Dataset Margin-Aware Privatization"
echo "=========================================="
echo "Dataset: $DATASET"
echo "Split: $SPLIT"
echo "Epsilon (eps_labels): $EPS_LABELS"
echo "Batch size: $BATCH_SIZE"
echo "Convert first: $CONVERT_FIRST"
echo ""

# Step 0: Convert alpaca_eval to DPO format (if needed)
if [ "$CONVERT_FIRST" = "true" ] || [ "$CONVERT_FIRST" = "True" ]; then
    echo "Step 0/4: Converting alpaca_eval to DPO format..."
    python preprocessing/alpaca/convert_alpaca_eval_to_dpo.py \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --output_file ./alpaca_eval_dpo.jsonl
    
    if [ $? -ne 0 ]; then
        echo "❌ Error in Step 0: Conversion failed"
        exit 1
    fi
    
    echo ""
    echo "✅ Step 0 complete: Dataset converted to DPO format"
    echo ""
    
    # Use the converted file for scoring
    DPO_FILE="./alpaca_eval_dpo.jsonl"
else
    DPO_FILE=""
fi

# Step 1: Score with reward model
echo "Step 1/4: Scoring dataset with reward model..."
if [ -n "$DPO_FILE" ]; then
    python preprocessing/alpaca/score_alpaca_rm.py \
      --dpo_file "$DPO_FILE" \
      --batch_size $BATCH_SIZE \
      --output_dir ./alpaca_with_margins
else
    python preprocessing/alpaca/score_alpaca_rm.py \
      --dataset "$DATASET" \
      --split "$SPLIT" \
      --batch_size $BATCH_SIZE \
      --output_dir ./alpaca_with_margins
fi

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 1: Scoring failed"
    exit 1
fi

echo ""
echo "✅ Step 1 complete: Dataset scored with reward model"
echo ""

# Step 2: Process margins
echo "Step 2/4: Processing margins (clip and normalize)..."
python preprocessing/alpaca/process_alpaca_margins.py \
  --input_dir ./alpaca_with_margins \
  --output_dir ./alpaca_with_processed_margins

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 2: Margin processing failed"
    exit 1
fi

echo ""
echo "✅ Step 2 complete: Margins processed"
echo ""

# Step 3: Privatize labels
echo "Step 3/4: Privatizing labels with margin-aware flipping..."
python preprocessing/alpaca/privatize_alpaca_labels.py \
  --input_dir ./alpaca_with_processed_margins \
  --output_file ./alpaca_privatized_dataset.jsonl \
  --eps_labels $EPS_LABELS

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 3: Label privatization failed"
    exit 1
fi

echo ""
echo "✅ Step 3 complete: Labels privatized"
echo ""
echo "=========================================="
echo "✅ All steps completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
if [ "$CONVERT_FIRST" = "true" ] || [ "$CONVERT_FIRST" = "True" ]; then
    echo "  - Converted DPO dataset: ./alpaca_eval_dpo.jsonl"
fi
echo "  - Scored dataset: ./alpaca_with_margins"
echo "  - Processed margins: ./alpaca_with_processed_margins"
echo "  - Privatized dataset: ./alpaca_privatized_dataset.jsonl"
echo "  - Summary: ./alpaca_privatized_dataset_summary.json"
echo ""
