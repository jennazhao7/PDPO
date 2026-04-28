#!/bin/bash
# Compare plain-LoRA gpt2M vs rr+Soft-Bayes Stage-2 gpt2M
# Uses GPT-4 as judge + reward model bench
#
# Prerequisites:
#   1. Run stage2 soft-bayes first if not done:
#      python lora/train_stage2_soft_bayes.py \
#        --model gpt2-medium \
#        --stage1_adapter outputs/stage2_lora/stage1 \
#        --data lora/preprocessing/d2_rr_flipped.jsonl \
#        --out outputs/gpt2-medium-stage2-softbayes \
#        --target_modules c_attn,c_proj,c_fc
#
#   2. export OPENAI_API_KEY=sk-...
#
# Usage:
#   bash eval/run_plain_vs_stage2_softbayes_gpt2M.sh
#
# To skip GPT-4 judging (reward only):
#   SKIP_JUDGE=1 bash eval/run_plain_vs_stage2_softbayes_gpt2M.sh
#
# To skip reward model (GPT-4 only):
#   SKIP_REWARD=1 bash eval/run_plain_vs_stage2_softbayes_gpt2M.sh

set -e
cd "$(dirname "$0")/.."

OUT_DIR="lora/eval/results/plain_vs_stage2_softbayes_gpt2M"
N_PROMPTS="${N_PROMPTS:-100}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4}"

# Soft-bayes may output to stage2_soft_bayes_* if --out was not used
SOFTBAYES_DIR="${SOFTBAYES_DIR:-outputs/gpt2-medium-stage2-softbayes}"
if [[ ! -f "$SOFTBAYES_DIR/M2_manifest.json" ]]; then
  # Try auto-derived path
  ALT="outputs/stage2_soft_bayes_gpt2-medium_stage1"
  if [[ -f "$ALT/M2_manifest.json" ]]; then
    SOFTBAYES_DIR="$ALT"
  else
    echo "Error: Need stage2 soft-bayes output. Run:"
    echo "  python lora/train_stage2_soft_bayes.py --model gpt2-medium \\"
    echo "    --stage1_adapter outputs/stage2_lora/stage1 \\"
    echo "    --data lora/preprocessing/d2_rr_flipped.jsonl \\"
    echo "    --out outputs/gpt2-medium-stage2-softbayes \\"
    echo "    --target_modules c_attn,c_proj,c_fc"
    exit 1
  fi
fi

EXTRA_FLAGS=""
[[ "${SKIP_JUDGE:-0}" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip_judge"
[[ "${SKIP_REWARD:-0}" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip_reward"

echo "=== Eval: plain-LoRA vs rr+Soft-Bayes Stage-2 (gpt2-medium) ==="
echo "Out: $OUT_DIR"
echo "Soft-Bayes: $SOFTBAYES_DIR"
echo ""

python eval/eval_compare.py \
  --model_a_type lora \
  --model_a_path outputs/plain_lora_truthy_subset \
  --model_a_base gpt2-medium \
  --model_a_label "plain-lora-gpt2M" \
  \
  --model_b_type stage2 \
  --model_b_manifest "$SOFTBAYES_DIR/M2_manifest.json" \
  --model_b_label "rr+SoftBayes-stage2-gpt2M" \
  \
  --prompts_jsonl preprocessing/truthydpo/truthy_dpo_subset.jsonl \
  --prompt_key prompt \
  --n_prompts "$N_PROMPTS" \
  --seed 42 \
  \
  --judge_model "$JUDGE_MODEL" \
  --n_votes 1 \
  \
  --out_dir "$OUT_DIR" \
  $EXTRA_FLAGS

echo ""
echo "=== Done. Results in $OUT_DIR ==="
