#!/bin/bash
# Run all 5 pairwise comparisons for a given model size.
# Uses GPT-4 as judge + OpenAssistant reward model.
#
# Comparisons:
#   1. rr+SoftBayes  vs  plain-LoRA
#   2. rr+SoftBayes  vs  DP-LoRA
#   3. rr+MLE        vs  plain-LoRA
#   4. rr+MLE        vs  DP-LoRA
#   5. rr+SoftBayes  vs  rr+MLE
#
# Usage:
#   bash eval/run_all_comparisons.sh gpt2-medium
#   bash eval/run_all_comparisons.sh gpt2-large
#   bash eval/run_all_comparisons.sh pythia-1b
#
# Env vars:
#   OPENAI_API_KEY    required for GPT-4 judge
#   JUDGE_MODEL       default: gpt-4
#   N_PROMPTS         default: 100
#   SKIP_JUDGE=1      skip GPT-4 judge (reward only)
#   SKIP_REWARD=1     skip reward model (judge only)

set -e
cd "$(dirname "$0")/.."

MODEL_SIZE="${1:?Usage: $0 <gpt2-medium|gpt2-large|pythia-1b>}"
N_PROMPTS="${N_PROMPTS:-100}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4}"
PROMPTS="preprocessing/truthydpo/truthy_dpo_subset.jsonl"
REWARD_MODELS="OpenAssistant/reward-model-deberta-v3-large-v2"

EXTRA=""
[[ "${SKIP_JUDGE:-0}" == "1" ]] && EXTRA="$EXTRA --skip_judge"
[[ "${SKIP_REWARD:-0}" == "1" ]] && EXTRA="$EXTRA --skip_reward"

# --- Resolve paths per model size ---
case "$MODEL_SIZE" in
  gpt2-medium|gpt2M)
    BASE=gpt2-medium
    PLAIN="outputs/plain_lora_truthy_subset"
    DPLORA="outputs/dp_lora_truthy_gpt2M"
    MLE_MANIFEST="outputs/gpt2-medium-stage2-mle/M2_manifest.json"
    SB_MANIFEST="outputs/gpt2-medium-stage2-softbayes/M2_manifest.json"
    TAG="gpt2M"
    ;;
  gpt2-large|gpt2L)
    BASE=gpt2-large
    PLAIN="outputs/plain_lora_truthy_gpt2L"
    DPLORA="outputs/dp_lora_truthy_gpt2L"
    MLE_MANIFEST="outputs/gpt2-large-stage2-mle/M2_manifest.json"
    SB_MANIFEST="outputs/gpt2-large-stage2-softbayes/M2_manifest.json"
    TAG="gpt2L"
    ;;
  pythia-1b|pythia1b)
    BASE=EleutherAI/pythia-1b
    PLAIN="outputs/plain_lora_truthy_pythia1b"
    DPLORA="outputs/dp_lora_truthy_pythia1b"
    MLE_MANIFEST="outputs/pythia1b-stage2-mle/M2_manifest.json"
    SB_MANIFEST="outputs/pythia1b-stage2-softbayes/M2_manifest.json"
    TAG="pythia1b"
    ;;
  *)
    echo "ERROR: Unknown model size: $MODEL_SIZE"
    echo "Supported: gpt2-medium, gpt2-large, pythia-1b"
    exit 1
    ;;
esac

EVAL_BASE="lora/eval/results/${TAG}"

# --- Helper: check if a model / manifest exists ---
check_lora_exists() {
  local path="$1" label="$2"
  if [[ ! -f "${path}/adapter_config.json" ]]; then
    echo "WARNING: ${label} not found at ${path} -- skipping comparisons involving it."
    return 1
  fi
  return 0
}

check_manifest_exists() {
  local path="$1" label="$2"
  if [[ ! -f "$path" ]]; then
    echo "WARNING: ${label} manifest not found at ${path} -- skipping comparisons involving it."
    return 1
  fi
  return 0
}

# --- Helper to run one comparison ---
run_compare() {
  local A_TYPE="$1" A_PATH="$2" A_BASE="$3" A_MANIFEST="$4" A_LABEL="$5"
  local B_TYPE="$6" B_PATH="$7" B_BASE="$8" B_MANIFEST="$9" B_LABEL="${10}"
  local OUT_DIR="${11}"

  echo ""
  echo "================================================================"
  echo "  ${A_LABEL}  vs  ${B_LABEL}"
  echo "  -> ${OUT_DIR}"
  echo "================================================================"

  local A_ARGS=""
  [[ "$A_TYPE" == "lora" ]] && A_ARGS="--model_a_path $A_PATH --model_a_base $A_BASE"
  [[ "$A_TYPE" == "stage2" ]] && A_ARGS="--model_a_manifest $A_MANIFEST"

  local B_ARGS=""
  [[ "$B_TYPE" == "lora" ]] && B_ARGS="--model_b_path $B_PATH --model_b_base $B_BASE"
  [[ "$B_TYPE" == "stage2" ]] && B_ARGS="--model_b_manifest $B_MANIFEST"

  python eval/eval_compare.py \
    --model_a_type "$A_TYPE" $A_ARGS --model_a_label "$A_LABEL" \
    --model_b_type "$B_TYPE" $B_ARGS --model_b_label "$B_LABEL" \
    --prompts_jsonl "$PROMPTS" --prompt_key prompt \
    --n_prompts "$N_PROMPTS" --seed 42 \
    --judge_model "$JUDGE_MODEL" --n_votes 1 \
    --reward_models "$REWARD_MODELS" \
    --out_dir "$OUT_DIR" \
    $EXTRA
}

echo "=========================================="
echo "  All comparisons for: $MODEL_SIZE ($TAG)"
echo "=========================================="

HAS_PLAIN=true
HAS_DPLORA=true
HAS_MLE=true
HAS_SB=true

check_lora_exists "$PLAIN" "plain-LoRA" || HAS_PLAIN=false
check_lora_exists "$DPLORA" "DP-LoRA" || HAS_DPLORA=false
check_manifest_exists "$MLE_MANIFEST" "rr+MLE" || HAS_MLE=false
check_manifest_exists "$SB_MANIFEST" "rr+SoftBayes" || HAS_SB=false

COMPLETED=0
SKIPPED=0

# --- 1. rr+SoftBayes vs plain-LoRA ---
if $HAS_SB && $HAS_PLAIN; then
  run_compare \
    stage2 "" "" "$SB_MANIFEST" "rr+SoftBayes" \
    lora "$PLAIN" "$BASE" "" "plain-LoRA" \
    "${EVAL_BASE}/softbayes_vs_plain"
  COMPLETED=$((COMPLETED + 1))
else
  echo "SKIP: rr+SoftBayes vs plain-LoRA (missing model)"
  SKIPPED=$((SKIPPED + 1))
fi

# --- 2. rr+SoftBayes vs DP-LoRA ---
if $HAS_SB && $HAS_DPLORA; then
  run_compare \
    stage2 "" "" "$SB_MANIFEST" "rr+SoftBayes" \
    lora "$DPLORA" "$BASE" "" "DP-LoRA" \
    "${EVAL_BASE}/softbayes_vs_dplora"
  COMPLETED=$((COMPLETED + 1))
else
  echo "SKIP: rr+SoftBayes vs DP-LoRA (missing model)"
  SKIPPED=$((SKIPPED + 1))
fi

# --- 3. rr+MLE vs plain-LoRA ---
if $HAS_MLE && $HAS_PLAIN; then
  run_compare \
    stage2 "" "" "$MLE_MANIFEST" "rr+MLE" \
    lora "$PLAIN" "$BASE" "" "plain-LoRA" \
    "${EVAL_BASE}/mle_vs_plain"
  COMPLETED=$((COMPLETED + 1))
else
  echo "SKIP: rr+MLE vs plain-LoRA (missing model)"
  SKIPPED=$((SKIPPED + 1))
fi

# --- 4. rr+MLE vs DP-LoRA ---
if $HAS_MLE && $HAS_DPLORA; then
  run_compare \
    stage2 "" "" "$MLE_MANIFEST" "rr+MLE" \
    lora "$DPLORA" "$BASE" "" "DP-LoRA" \
    "${EVAL_BASE}/mle_vs_dplora"
  COMPLETED=$((COMPLETED + 1))
else
  echo "SKIP: rr+MLE vs DP-LoRA (missing model)"
  SKIPPED=$((SKIPPED + 1))
fi

# --- 5. rr+SoftBayes vs rr+MLE ---
if $HAS_SB && $HAS_MLE; then
  run_compare \
    stage2 "" "" "$SB_MANIFEST" "rr+SoftBayes" \
    stage2 "" "" "$MLE_MANIFEST" "rr+MLE" \
    "${EVAL_BASE}/softbayes_vs_mle"
  COMPLETED=$((COMPLETED + 1))
else
  echo "SKIP: rr+SoftBayes vs rr+MLE (missing model)"
  SKIPPED=$((SKIPPED + 1))
fi

echo ""
echo "=========================================="
echo "  Done for $TAG: $COMPLETED completed, $SKIPPED skipped"
echo "  Results in: ${EVAL_BASE}/"
echo "=========================================="
echo ""
echo "Subdirectories:"
[[ $HAS_SB == true && $HAS_PLAIN == true ]] && echo "  softbayes_vs_plain/"
[[ $HAS_SB == true && $HAS_DPLORA == true ]] && echo "  softbayes_vs_dplora/"
[[ $HAS_MLE == true && $HAS_PLAIN == true ]] && echo "  mle_vs_plain/"
[[ $HAS_MLE == true && $HAS_DPLORA == true ]] && echo "  mle_vs_dplora/"
[[ $HAS_SB == true && $HAS_MLE == true ]] && echo "  softbayes_vs_mle/"
