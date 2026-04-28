#!/bin/bash
# Compare plain-LoRA gpt2M vs rr+MLE Stage-2 gpt2M
# Uses GPT-4 as judge + reward model bench
#
# Prerequisites:
#   export OPENAI_API_KEY=sk-...
#   pip install openai
#
# Usage:
#   bash eval/run_plain_vs_stage2_gpt2M.sh
#
# To skip GPT-4 judging (reward only):
#   SKIP_JUDGE=1 bash eval/run_plain_vs_stage2_gpt2M.sh
#
# To skip reward model (GPT-4 only):
#   SKIP_REWARD=1 bash eval/run_plain_vs_stage2_gpt2M.sh

set -e
cd "$(dirname "$0")/.."

OUT_DIR="lora/eval/results/plain_vs_stage2_gpt2M"
N_PROMPTS="${N_PROMPTS:-100}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4}"

EXTRA_FLAGS=""
[[ "${SKIP_JUDGE:-0}" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip_judge"
[[ "${SKIP_REWARD:-0}" == "1" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --skip_reward"

echo "=== Eval: plain-LoRA vs rr+MLE Stage-2 (gpt2-medium) ==="
echo "Out: $OUT_DIR"
echo ""

python eval/eval_compare.py \
  --model_a_type lora \
  --model_a_path outputs/plain_lora_truthy_subset \
  --model_a_base gpt2-medium \
  --model_a_label "plain-lora-gpt2M" \
  \
  --model_b_type stage2 \
  --model_b_manifest outputs/gpt2-medium-stage2-mle/M2_manifest.json \
  --model_b_label "rr+MLE-stage2-gpt2M" \
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
