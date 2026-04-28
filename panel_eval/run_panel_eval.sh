#!/usr/bin/env bash
# run_panel_eval.sh
# =================
# Full pipeline: generate responses → judge panel → aggregate.
# Edit the CONFIG section at the top, then run:
#   bash run_panel_eval.sh
#
# For cluster submission (SGE/PBS), wrap this in a qsub script.
# Con-J requires a GPU; GPT-4o and Gemini are API calls (no GPU needed).

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these for your run
# ─────────────────────────────────────────────────────────────────────────────

DATASET="pku"
EPSILON="0.5"
SEED="42"

# Base model (must match what the adapters were trained on)
BASE_MODEL="Qwen/Qwen2.5-3B"

# Adapter paths (directory or M2_manifest.json — resolved automatically)
SB_FRESH_PATH="/users/jzhao7/PDPO/stage2_debugging/models/sb_fresh_pku_eps0.5_seed42"
MAP_RETRAIN_PATH="/users/jzhao7/PDPO/stage2_debugging/models/map_retrain_pku_eps0.5_seed42"
MLE_FRESH_PATH="/users/jzhao7/PDPO/stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_pku_eps0.5_s42"

# Prompts file
PROMPTS="/users/jzhao7/PDPO/stage2_debugging/testsets/pku_secure/test_pref.jsonl"

# Root output directory
OUTPUT_ROOT="/users/jzhao7/PDPO/panel_eval/results/${DATASET}_eps${EPSILON}_seed${SEED}"

# Generation config
MAX_NEW_TOKENS=256
N_PROMPTS=${PANEL_N_PROMPTS_OVERRIDE:-200}   # set to 50 for smoke test, 200 for full

# Judges to run: space-separated subset of: con_j gpt4o gemini
JUDGES=${PANEL_JUDGES_OVERRIDE:-"con_j"}     # add gpt4o gemini once you have API keys

# API keys (set in environment or uncomment and fill in here)
# export OPENAI_API_KEY="sk-..."
# export GEMINI_API_KEY="AI..."

# Device for local models (Con-J, base model)
DEVICE="cuda"        # or "cuda:0", "cpu"

# Script dir (panel_eval/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ─────────────────────────────────────────────────────────────────────────────
# PART 1 — Generate responses
# ─────────────────────────────────────────────────────────────────────────────

RESP_DIR="${OUTPUT_ROOT}/responses"
mkdir -p "${RESP_DIR}"

echo "======================================================"
echo "[Part 1] Generating responses (n=${N_PROMPTS} prompts)"
echo "======================================================"

python "${SCRIPT_DIR}/generate_responses.py" \
    --base_model "${BASE_MODEL}" \
    --adapters "sb_fresh=${SB_FRESH_PATH},map_retrain=${MAP_RETRAIN_PATH},mle_fresh=${MLE_FRESH_PATH}" \
    --prompts "${PROMPTS}" \
    --output_dir "${RESP_DIR}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --n "${N_PROMPTS}" \
    --device "${DEVICE}"

echo "[Part 1] Done. Responses in ${RESP_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
# PART 2 & 3 — Judge + Aggregate for each comparison
# ─────────────────────────────────────────────────────────────────────────────

run_comparison() {
    local NAME_A="$1"
    local NAME_B="$2"
    local COMP_DIR="${OUTPUT_ROOT}/${NAME_A}_vs_${NAME_B}"
    mkdir -p "${COMP_DIR}"

    echo ""
    echo "=============================================="
    echo "[Part 2] Judging: ${NAME_A} vs ${NAME_B}"
    echo "=============================================="

    python "${SCRIPT_DIR}/judge_panel.py" \
        --responses_a "${RESP_DIR}/${NAME_A}_responses.jsonl" \
        --responses_b "${RESP_DIR}/${NAME_B}_responses.jsonl" \
        --output_dir "${COMP_DIR}" \
        --judges ${JUDGES} \
        --n "${N_PROMPTS}" \
        --device "${DEVICE}"

    echo ""
    echo "=============================================="
    echo "[Part 3] Aggregating: ${NAME_A} vs ${NAME_B}"
    echo "=============================================="

    python "${SCRIPT_DIR}/aggregate_panel.py" \
        --input_dir "${COMP_DIR}"

    echo "[Done] ${NAME_A} vs ${NAME_B} → ${COMP_DIR}"
}

# SB vs MAP (primary comparison)
run_comparison "sb_fresh" "map_retrain"

# SB vs MLE
run_comparison "sb_fresh" "mle_fresh"

# MAP vs MLE
run_comparison "map_retrain" "mle_fresh"

echo ""
echo "======================================================"
echo "[All Done] Results under: ${OUTPUT_ROOT}"
echo "======================================================"
