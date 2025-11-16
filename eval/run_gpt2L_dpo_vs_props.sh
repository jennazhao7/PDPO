#!/bin/bash
# Script to run evaluation: Jennazhao7/gpt2-large-dpo-m1-v2 vs PROPS GPT-2 Large checkpoint
# Usage: bash eval/run_gpt2L_dpo_vs_props.sh

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
YOUR_MODEL="Jennazhao7/gpt2-large-dpo-m1-v2"  # Your GPT-2 Large DPO model
BASELINE_MODEL="Setpember/Jon_GPT2L_DPO_props_epi_point1"  # PROPS GPT-2 Large checkpoint (epsilon=0.1)
OUTPUT_CSV="eval/gpt2L-dpo-m1-v2_vs_props-ep01.csv"
NUM_PROMPTS=100
BATCH_SIZE=4  # Smaller batch size for large model (reduce if OOM, increase for speed if you have memory)
JUDGE_MODEL="gpt-4o-mini"  # Change to "gpt-4o" for better quality

# === Run evaluation ===
echo "🚀 Starting evaluation..."
echo "Your Model (A): $YOUR_MODEL"
echo "Baseline PROPS GPT-2 Large (B): $BASELINE_MODEL"
echo "Judge: $JUDGE_MODEL"
echo "Batch Size: $BATCH_SIZE"
echo "Output CSV: $OUTPUT_CSV"
echo ""

python eval/truthy_gpt2L_eval_updated.py \
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

