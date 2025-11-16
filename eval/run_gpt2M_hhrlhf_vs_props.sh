#!/bin/bash
# Script to run evaluation: Jennazhao7/gpt2-medium-dpo-hhrlhf vs PROPS HH-RLHF checkpoint
# Usage: bash eval/run_gpt2M_hhrlhf_vs_props.sh

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Check for OpenAI API Key ===
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ ERROR: OPENAI_API_KEY environment variable is not set!"
    echo ""
    echo "Please set it with one of these methods:"
    echo "  1. Export in current session:"
    echo "     export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    echo "  2. Add to ~/.bashrc (permanent):"
    echo "     echo 'export OPENAI_API_KEY=\"your-api-key-here\"' >> ~/.bashrc"
    echo "     source ~/.bashrc"
    echo ""
    exit 1
fi

# === Configuration ===
YOUR_MODEL="Jennazhao7/gpt2-medium-dpo-hhrlhf"  # Your HH-RLHF DPO model
BASELINE_MODEL="Setpember/HH_GPT2_DPO_props_epi_1"  # PROPS HH-RLHF checkpoint (epsilon=1.0)
OUTPUT_CSV="eval/gpt2M-hhrlhf-dpo_vs_props-ep1.csv"
NUM_PROMPTS=100
BATCH_SIZE=8  # Adjust based on GPU memory (reduce if OOM, increase for speed)
JUDGE_MODEL="gpt-4o-mini"  # Change to "gpt-4o" for better quality

# === Run evaluation ===
echo "🚀 Starting evaluation..."
echo "Your Model (A): $YOUR_MODEL"
echo "Baseline PROPS HH-RLHF (B): $BASELINE_MODEL"
echo "Judge: $JUDGE_MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Output CSV: $OUTPUT_CSV"
echo ""

python eval/truthy_gpt2M_eval.py \
  --model_a "$YOUR_MODEL" \
  --model_b "$BASELINE_MODEL" \
  --n $NUM_PROMPTS \
  --batch_size $BATCH_SIZE \
  --judge_model "$JUDGE_MODEL" \
  --out_csv "$OUTPUT_CSV" \
  --n_votes 1

echo ""
echo "✅ Evaluation complete!"
echo "📊 Results saved to: $OUTPUT_CSV"

# Move summary file to summary_results directory
SUMMARY_FILE="${OUTPUT_CSV%.csv}_summary.csv"
SUMMARY_DEST="eval/summary_results/$(basename $SUMMARY_FILE)"
if [ -f "$SUMMARY_FILE" ]; then
    mkdir -p eval/summary_results
    mv "$SUMMARY_FILE" "$SUMMARY_DEST"
    echo "📈 Summary saved to: $SUMMARY_DEST"
else
    echo "⚠️  Warning: Summary file not found at $SUMMARY_FILE"
fi

