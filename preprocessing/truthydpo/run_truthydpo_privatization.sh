#!/bin/bash
# Master script to run margin-aware label flipping for Truthy-DPO dataset
# This script:
# 1. Scores Truthy-DPO dataset with reward model
# 2. Processes margins (clip and normalize)
# 3. Privatizes labels using margin-aware flipping
# 4. Prepares DPO training data

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Configuration ===
EPS_LABELS="${1:-1.0}"                # Privacy parameter for label flipping (default: 1.0)
BATCH_SIZE="${2:-32}"                # Batch size for reward model scoring (default: 32)

echo "=========================================="
echo "Truthy-DPO Dataset Margin-Aware Privatization"
echo "=========================================="
echo "Epsilon (eps_labels): $EPS_LABELS"
echo "Batch size: $BATCH_SIZE"
echo ""

# Step 1: Score with reward model
echo "Step 1/4: Scoring dataset with reward model..."
cd preprocessing/truthydpo
python min_rm.py

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 1: Scoring failed"
    exit 1
fi

echo ""
echo "✅ Step 1 complete: Dataset scored with reward model"
echo ""

# Step 2: Process margins
echo "Step 2/4: Processing margins (clip and normalize)..."
python process_margins.py

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 2: Margin processing failed"
    exit 1
fi

echo ""
echo "✅ Step 2 complete: Margins processed"
echo ""

# Step 3: Privatize labels
echo "Step 3/4: Privatizing labels with margin-aware flipping..."
# Update eps_labels in the script if needed, or pass as argument
python privatize_labels.py

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 3: Label privatization failed"
    exit 1
fi

echo ""
echo "✅ Step 3 complete: Labels privatized"
echo ""

# Step 4: Prepare DPO training data
echo "Step 4/4: Preparing DPO training data..."
cd ../..
python preprocessing/truthydpo/prepare_dpo_data.py

if [ $? -ne 0 ]; then
    echo "❌ Error in Step 4: DPO data preparation failed"
    exit 1
fi

echo ""
echo "✅ Step 4 complete: DPO training data prepared"
echo ""
echo "=========================================="
echo "✅ All steps completed successfully!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - Scored dataset: ./props_with_margins_truthy"
echo "  - Processed margins: ./props_with_processed_margins"
echo "  - Privatized dataset: ./dpo_privatized_dataset.jsonl"
echo "  - DPO training data: ./dpo_train_ready.jsonl"
echo "  - Summary: ./privatization_summary.json"
echo ""

