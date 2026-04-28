#!/bin/bash
# Master evaluation: smoke test, then full 100-prompt comparisons for all 3 model sizes.
#
# Env vars:
#   OPENAI_API_KEY    required for GPT-4 judge
#   JUDGE_MODEL       default: gpt-4
#   SKIP_JUDGE=1      skip GPT-4 judge (reward only)
#   SKIP_REWARD=1     skip reward model (judge only)
#   TARGET_MODELS     comma-separated subset: gpt2M,gpt2L,pythia1b (default: all)
#   SKIP_SMOKE=1      skip smoke test when resuming partial runs
#   PROMPT_MODE       high_signal (default) or jsonl
#   N_PROMPTS         prompts per comparison (default: 40)
#   MAX_NEW_TOKENS    generation length cap (default: 96)

set -uo pipefail   # no -e: we handle errors per-comparison
cd "$(dirname "$0")/.."

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="lora/eval/logs"
RESULTS_ROOT="${RESULTS_ROOT:-lora/eval/results}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
mkdir -p "$LOG_DIR" "$RESULTS_ROOT"
LOGFILE="${LOG_DIR}/eval_all_${TIMESTAMP}.log"

# Tee all output to both console and logfile
exec > >(tee -a "$LOGFILE") 2>&1

echo "================================================================"
echo "  Master Evaluation Pipeline"
echo "  Started: $(date)"
echo "  Log: $LOGFILE"
echo "================================================================"

# Check for OPENAI_API_KEY
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARNING: OPENAI_API_KEY not set. GPT-4 judge will be skipped."
  FORCE_SKIP_JUDGE="--skip_judge"
else
  echo "OPENAI_API_KEY is set. GPT-4 judge will run."
  FORCE_SKIP_JUDGE=""
fi

PROMPTS="preprocessing/truthydpo/truthy_dpo_subset.jsonl"
REWARD_MODELS="OpenAssistant/reward-model-deberta-v3-large-v2"
PROMPT_MODE="${PROMPT_MODE:-high_signal}"
N_PROMPTS="${N_PROMPTS:-40}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"

EXTRA=""
[[ "${SKIP_JUDGE:-0}" == "1" ]] && EXTRA="$EXTRA --skip_judge"
[[ "${SKIP_REWARD:-0}" == "1" ]] && EXTRA="$EXTRA --skip_reward"
# Combine explicit skip flags with auto-detected skip
ALL_EXTRA="$EXTRA $FORCE_SKIP_JUDGE"

# Target model phases to run (default all)
TARGET_MODELS="${TARGET_MODELS:-gpt2M,gpt2L,pythia1b}"
_contains_model() {
  [[ ",${TARGET_MODELS}," == *",$1,"* ]]
}
RUN_GPT2M=false
RUN_GPT2L=false
RUN_PYTHIA=false
_contains_model "gpt2M" && RUN_GPT2M=true
_contains_model "gpt2L" && RUN_GPT2L=true
_contains_model "pythia1b" && RUN_PYTHIA=true

# Tracking arrays (bash 4+)
declare -a RESULT_DIRS=()
declare -a RESULT_LABELS=()
declare -a RESULT_STATUS=()

# ============================================================
# PHASE 0: Smoke test (5 prompts, 1 comparison, generation-only)
# ============================================================
if [[ "${SKIP_SMOKE:-0}" == "1" ]]; then
  echo "[Smoke] SKIP_SMOKE=1 -> skipping smoke test."
else
  echo ""
  echo "================================================================"
  echo "  PHASE 0: Smoke Test (5 prompts, generation-only)"
  echo "================================================================"

  SMOKE_DIR="${RESULTS_ROOT}/_smoke_test"
  SMOKE_STAGE2_MANIFEST="${SMOKE_STAGE2_MANIFEST:-${OUTPUT_ROOT}/gpt2-medium-stage2-softbayes/M2_manifest.json}"
  SMOKE_PLAIN_PATH="${SMOKE_PLAIN_PATH:-${OUTPUT_ROOT}/plain_lora_truthy_subset}"

  python eval/eval_compare.py \
    --model_a_type stage2 \
    --model_a_manifest "$SMOKE_STAGE2_MANIFEST" \
    --model_a_label "rr+SoftBayes-gpt2M" \
    --model_b_type lora \
    --model_b_path "$SMOKE_PLAIN_PATH" \
    --model_b_base gpt2-medium \
    --model_b_label "plain-LoRA-gpt2M" \
    --prompt_mode "$PROMPT_MODE" --prompts_jsonl "$PROMPTS" --prompt_key prompt \
    --n_prompts 5 --seed 42 --max_new_tokens 64 \
    --judge_model "$JUDGE_MODEL" --n_votes 1 \
    --reward_models "$REWARD_MODELS" \
    --out_dir "$SMOKE_DIR" \
    --skip_judge --skip_reward 2>&1 || true
    # smoke test: skip both judge and reward, just test model loading + generation

  SMOKE_EXIT=$?

  # Verify output files exist
  if [[ ! -f "$SMOKE_DIR/generations.jsonl" ]]; then
    echo ""
    echo "SMOKE TEST FAILED: generations.jsonl not created."
    echo "Check the error above. Fix and re-run."
    exit 1
  fi

  echo ""
  echo "SMOKE TEST PASSED (generation works). Proceeding to full evaluation."
  echo ""
  rm -rf "$SMOKE_DIR"
fi

# ============================================================
# Helper: run one comparison with error isolation
# ============================================================
run_compare() {
  local A_TYPE="$1" A_PATH="$2" A_BASE="$3" A_MANIFEST="$4" A_LABEL="$5"
  local B_TYPE="$6" B_PATH="$7" B_BASE="$8" B_MANIFEST="$9" B_LABEL="${10}"
  local OUT_DIR="${11}"
  local COMP_LABEL="${A_LABEL} vs ${B_LABEL}"

  echo ""
  echo "----------------------------------------------------------------"
  echo "  ${COMP_LABEL}"
  echo "  -> ${OUT_DIR}"
  echo "----------------------------------------------------------------"

  mkdir -p "$OUT_DIR"

  local A_ARGS=""
  [[ "$A_TYPE" == "lora" ]] && A_ARGS="--model_a_path $A_PATH --model_a_base $A_BASE"
  [[ "$A_TYPE" == "stage2" ]] && A_ARGS="--model_a_manifest $A_MANIFEST"

  local B_ARGS=""
  [[ "$B_TYPE" == "lora" ]] && B_ARGS="--model_b_path $B_PATH --model_b_base $B_BASE"
  [[ "$B_TYPE" == "stage2" ]] && B_ARGS="--model_b_manifest $B_MANIFEST"

  local RC=0
  python eval/eval_compare.py \
    --model_a_type "$A_TYPE" $A_ARGS --model_a_label "$A_LABEL" \
    --model_b_type "$B_TYPE" $B_ARGS --model_b_label "$B_LABEL" \
    --prompt_mode "$PROMPT_MODE" --prompts_jsonl "$PROMPTS" --prompt_key prompt \
    --n_prompts "$N_PROMPTS" --seed 42 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --judge_model "$JUDGE_MODEL" --n_votes 1 \
    --reward_models "$REWARD_MODELS" \
    --out_dir "$OUT_DIR" \
    $ALL_EXTRA || RC=$?

  RESULT_DIRS+=("$OUT_DIR")
  RESULT_LABELS+=("$COMP_LABEL")

  if [[ $RC -eq 0 && -f "$OUT_DIR/eval_report.json" ]]; then
    RESULT_STATUS+=("OK")
    echo "  -> DONE: ${OUT_DIR}/eval_report.json"
  else
    RESULT_STATUS+=("FAIL(rc=$RC)")
    echo "  -> FAILED (exit $RC). Continuing to next comparison."
  fi
  return 0  # always continue
}

# ============================================================
# Helper: check model existence
# ============================================================
check_lora() {
  [[ -f "${1}/adapter_config.json" ]]
}
check_manifest() {
  [[ -f "$1" ]]
}

# ============================================================
# PHASE 1: gpt2-medium
# ============================================================
if $RUN_GPT2M; then
echo "================================================================"
echo "  PHASE 1: gpt2-medium comparisons (100 prompts)"
echo "================================================================"

TAG="gpt2M"
BASE="gpt2-medium"
PLAIN="${PLAIN_GPT2M:-${OUTPUT_ROOT}/plain_lora_truthy_subset}"
DPLORA="${DPLORA_GPT2M:-${OUTPUT_ROOT}/dp_lora_truthy_gpt2M}"
MLE_MF="${MLE_MF_GPT2M:-${OUTPUT_ROOT}/gpt2-medium-stage2-mle/M2_manifest.json}"
SB_MF="${SB_MF_GPT2M:-${OUTPUT_ROOT}/gpt2-medium-stage2-softbayes/M2_manifest.json}"
EVAL_BASE="${RESULTS_ROOT}/${TAG}"

if check_manifest "$SB_MF" && check_lora "$PLAIN"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" lora "$PLAIN" "$BASE" "" "plain-LoRA" "${EVAL_BASE}/softbayes_vs_plain"
else echo "SKIP: rr+SoftBayes vs plain-LoRA ($TAG)"; fi

if check_manifest "$SB_MF" && check_lora "$DPLORA"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" lora "$DPLORA" "$BASE" "" "DP-LoRA" "${EVAL_BASE}/softbayes_vs_dplora"
else echo "SKIP: rr+SoftBayes vs DP-LoRA ($TAG) -- $DPLORA missing"; fi

if check_manifest "$MLE_MF" && check_lora "$PLAIN"; then
  run_compare stage2 "" "" "$MLE_MF" "rr+MLE" lora "$PLAIN" "$BASE" "" "plain-LoRA" "${EVAL_BASE}/mle_vs_plain"
else echo "SKIP: rr+MLE vs plain-LoRA ($TAG)"; fi

if check_manifest "$MLE_MF" && check_lora "$DPLORA"; then
  run_compare stage2 "" "" "$MLE_MF" "rr+MLE" lora "$DPLORA" "$BASE" "" "DP-LoRA" "${EVAL_BASE}/mle_vs_dplora"
else echo "SKIP: rr+MLE vs DP-LoRA ($TAG) -- $DPLORA missing"; fi

if check_manifest "$SB_MF" && check_manifest "$MLE_MF"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" stage2 "" "" "$MLE_MF" "rr+MLE" "${EVAL_BASE}/softbayes_vs_mle"
else echo "SKIP: rr+SoftBayes vs rr+MLE ($TAG)"; fi
fi

# ============================================================
# PHASE 2: gpt2-large
# ============================================================
if $RUN_GPT2L; then
echo ""
echo "================================================================"
echo "  PHASE 2: gpt2-large comparisons (100 prompts)"
echo "================================================================"

TAG="gpt2L"
BASE="gpt2-large"
PLAIN="${PLAIN_GPT2L:-${OUTPUT_ROOT}/plain_lora_truthy_gpt2L}"
DPLORA="${DPLORA_GPT2L:-${OUTPUT_ROOT}/dp_lora_truthy_gpt2L}"
MLE_MF="${MLE_MF_GPT2L:-${OUTPUT_ROOT}/gpt2-large-stage2-mle/M2_manifest.json}"
SB_MF="${SB_MF_GPT2L:-${OUTPUT_ROOT}/gpt2-large-stage2-softbayes/M2_manifest.json}"
EVAL_BASE="${RESULTS_ROOT}/${TAG}"

if check_manifest "$SB_MF" && check_lora "$PLAIN"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" lora "$PLAIN" "$BASE" "" "plain-LoRA" "${EVAL_BASE}/softbayes_vs_plain"
else echo "SKIP: rr+SoftBayes vs plain-LoRA ($TAG)"; fi

if check_manifest "$SB_MF" && check_lora "$DPLORA"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" lora "$DPLORA" "$BASE" "" "DP-LoRA" "${EVAL_BASE}/softbayes_vs_dplora"
else echo "SKIP: rr+SoftBayes vs DP-LoRA ($TAG)"; fi

if check_manifest "$MLE_MF" && check_lora "$PLAIN"; then
  run_compare stage2 "" "" "$MLE_MF" "rr+MLE" lora "$PLAIN" "$BASE" "" "plain-LoRA" "${EVAL_BASE}/mle_vs_plain"
else echo "SKIP: rr+MLE vs plain-LoRA ($TAG)"; fi

if check_manifest "$MLE_MF" && check_lora "$DPLORA"; then
  run_compare stage2 "" "" "$MLE_MF" "rr+MLE" lora "$DPLORA" "$BASE" "" "DP-LoRA" "${EVAL_BASE}/mle_vs_dplora"
else echo "SKIP: rr+MLE vs DP-LoRA ($TAG)"; fi

if check_manifest "$SB_MF" && check_manifest "$MLE_MF"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" stage2 "" "" "$MLE_MF" "rr+MLE" "${EVAL_BASE}/softbayes_vs_mle"
else echo "SKIP: rr+SoftBayes vs rr+MLE ($TAG)"; fi
fi

# ============================================================
# PHASE 3: pythia-1b
# ============================================================
if $RUN_PYTHIA; then
echo ""
echo "================================================================"
echo "  PHASE 3: pythia-1b comparisons (100 prompts)"
echo "================================================================"

TAG="pythia1b"
BASE="EleutherAI/pythia-1b"
PLAIN="${PLAIN_PYTHIA1B:-${OUTPUT_ROOT}/plain_lora_truthy_pythia1b}"
DPLORA="${DPLORA_PYTHIA1B:-${OUTPUT_ROOT}/dp_lora_truthy_pythia1b}"
MLE_MF="${MLE_MF_PYTHIA1B:-${OUTPUT_ROOT}/pythia1b-stage2-mle/M2_manifest.json}"
SB_MF="${SB_MF_PYTHIA1B:-${OUTPUT_ROOT}/pythia1b-stage2-softbayes/M2_manifest.json}"
EVAL_BASE="${RESULTS_ROOT}/${TAG}"

if check_manifest "$SB_MF" && check_lora "$PLAIN"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" lora "$PLAIN" "$BASE" "" "plain-LoRA" "${EVAL_BASE}/softbayes_vs_plain"
else echo "SKIP: rr+SoftBayes vs plain-LoRA ($TAG)"; fi

if check_manifest "$SB_MF" && check_lora "$DPLORA"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" lora "$DPLORA" "$BASE" "" "DP-LoRA" "${EVAL_BASE}/softbayes_vs_dplora"
else echo "SKIP: rr+SoftBayes vs DP-LoRA ($TAG)"; fi

if check_manifest "$MLE_MF" && check_lora "$PLAIN"; then
  run_compare stage2 "" "" "$MLE_MF" "rr+MLE" lora "$PLAIN" "$BASE" "" "plain-LoRA" "${EVAL_BASE}/mle_vs_plain"
else echo "SKIP: rr+MLE vs plain-LoRA ($TAG)"; fi

if check_manifest "$MLE_MF" && check_lora "$DPLORA"; then
  run_compare stage2 "" "" "$MLE_MF" "rr+MLE" lora "$DPLORA" "$BASE" "" "DP-LoRA" "${EVAL_BASE}/mle_vs_dplora"
else echo "SKIP: rr+MLE vs DP-LoRA ($TAG)"; fi

if check_manifest "$SB_MF" && check_manifest "$MLE_MF"; then
  run_compare stage2 "" "" "$SB_MF" "rr+SoftBayes" stage2 "" "" "$MLE_MF" "rr+MLE" "${EVAL_BASE}/softbayes_vs_mle"
else echo "SKIP: rr+SoftBayes vs rr+MLE ($TAG)"; fi
fi

# ============================================================
# Final: Status table + master summary
# ============================================================
echo ""
echo "================================================================"
echo "  STATUS TABLE"
echo "================================================================"
printf "  %-6s  %-50s\n" "STATUS" "COMPARISON"
printf "  %-6s  %-50s\n" "------" "--------------------------------------------------"

OK_COUNT=0
FAIL_COUNT=0
for i in "${!RESULT_LABELS[@]}"; do
  printf "  %-6s  %s\n" "${RESULT_STATUS[$i]}" "${RESULT_LABELS[$i]}"
  printf "  %8s %s\n" "" "${RESULT_DIRS[$i]}"
  if [[ "${RESULT_STATUS[$i]}" == "OK" ]]; then
    OK_COUNT=$((OK_COUNT+1))
  else
    FAIL_COUNT=$((FAIL_COUNT+1))
  fi
done

echo ""
echo "  Total: ${#RESULT_LABELS[@]} comparisons, $OK_COUNT OK, $FAIL_COUNT failed"

# ============================================================
# Aggregate all eval_report.json into one master summary
# ============================================================
MASTER_SUMMARY="${RESULTS_ROOT}/master_summary_${TIMESTAMP}.json"
python3 -c "
import json, glob, os

reports = {}
for path in sorted(glob.glob('${RESULTS_ROOT}/*/*/eval_report.json')):
    rel = os.path.relpath(path, '${RESULTS_ROOT}')
    key = os.path.dirname(rel)  # e.g. gpt2M/softbayes_vs_plain
    with open(path) as f:
        reports[key] = json.load(f)

summary = {
    'timestamp': '${TIMESTAMP}',
    'n_comparisons': len(reports),
    'comparisons': {}
}

for key, rpt in reports.items():
    entry = {
        'model_a': rpt.get('model_a', ''),
        'model_b': rpt.get('model_b', ''),
        'n_prompts': rpt.get('n_prompts', 0),
    }
    judge = rpt.get('gpt4_judge')
    if judge:
        entry['gpt4_win_a'] = judge.get('win_rate_a', 0)
        entry['gpt4_win_b'] = judge.get('win_rate_b', 0)
        entry['gpt4_tie'] = judge.get('tie_rate', 0)
    reward = rpt.get('reward_bench', {})
    for rm_id, rm_data in reward.items():
        slug = rm_id.replace('/', '_').lower()[:30]
        entry[f'reward_{slug}_mean_a'] = rm_data.get('score_a_mean', 0)
        entry[f'reward_{slug}_mean_b'] = rm_data.get('score_b_mean', 0)
        entry[f'reward_{slug}_win_a'] = rm_data.get('reward_win_rate_a', 0)
        entry[f'reward_{slug}_win_b'] = rm_data.get('reward_win_rate_b', 0)
    summary['comparisons'][key] = entry

with open('$MASTER_SUMMARY', 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f'Master summary: $MASTER_SUMMARY ({len(reports)} comparisons)')
" || echo "WARNING: master summary generation failed"

echo ""
echo "================================================================"
echo "  ALL DONE: $(date)"
echo "================================================================"
echo ""
echo "Results tree:"
echo "  ${RESULTS_ROOT}/"
echo "    master_summary_${TIMESTAMP}.json   <- all results in one file"
echo "    gpt2M/"
echo "      softbayes_vs_plain/    softbayes_vs_dplora/    mle_vs_plain/"
echo "      mle_vs_dplora/         softbayes_vs_mle/"
echo "    gpt2L/"
echo "      softbayes_vs_plain/    softbayes_vs_dplora/    mle_vs_plain/"
echo "      mle_vs_dplora/         softbayes_vs_mle/"
echo "    pythia1b/"
echo "      softbayes_vs_plain/    softbayes_vs_dplora/    mle_vs_plain/"
echo "      mle_vs_dplora/         softbayes_vs_mle/"
echo ""
echo "Each comparison directory contains:"
echo "  eval_report.json         Combined final report"
echo "  generations.jsonl        Prompt + both model responses"
echo "  reward_scores_*.jsonl    Per-prompt reward model scores"
echo "  reward_summary_*.json    Aggregate reward stats"
[[ -z "$FORCE_SKIP_JUDGE" ]] && echo "  gpt4_judgments.jsonl     Per-prompt GPT-4 verdicts"
[[ -z "$FORCE_SKIP_JUDGE" ]] && echo "  gpt4_summary.json        Win/tie/loss rates"
echo ""
echo "Log: $LOGFILE"
