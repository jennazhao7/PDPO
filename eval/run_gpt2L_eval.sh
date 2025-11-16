#!/bin/bash
# Script to run evaluation for gpt2-large DPO model
# Usage: bash eval/run_gpt2L_eval.sh

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
    echo "  3. Set in this script (not recommended for security):"
    echo "     export OPENAI_API_KEY='your-api-key-here'"
    echo ""
    exit 1
fi

# === Configuration ===
# Option 1: Use local model path (if model is trained locally)
# YOUR_MODEL="./models/M1"

# Option 2: Use Hugging Face model (if you've pushed it to HF)
YOUR_MODEL="Jennazhao7/gpt2-large-dpo-m1"  # Your DPO model (update if different)

BASELINE_MODEL="Setpember/Jon_GPT2L_DPO_3props_epi_05"  # Baseline 3PROPS DPO model (epsilon=0.5)
OUTPUT_CSV="eval/gpt2L-dpo-m1_vs_3props-ep05.csv"
NUM_PROMPTS=100
BATCH_SIZE=4  # Smaller batch size for large model (reduce if OOM, increase for speed if you have memory)
JUDGE_MODEL="gpt-4o-mini"  # Change to "gpt-4o" for better quality, or "deepseek-chat" if using DeepSeek

# === Run evaluation ===
echo "🚀 Starting evaluation..."
echo "Your Model (A): $YOUR_MODEL"
echo "Baseline (B): $BASELINE_MODEL"
echo "Judge: $JUDGE_MODEL"
echo "Batch Size: $BATCH_SIZE"
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
echo "📈 Summary saved to: eval/summary_results/gpt2L-dpo-m1_vs_3props-ep05_summary.csv"

