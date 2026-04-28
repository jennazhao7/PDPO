#!/bin/bash
#$ -N panel_eval_4gpu
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4
#$ -pe smp 8
#$ -l h_rt=12:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

# ─────────────────────────────────────────────────────────────────────────────
# Panel Eval — 4-GPU Production Job
# ─────────────────────────────────────────────────────────────────────────────
# GPU assignment (no sharing, no conflicts):
#
#   Phase 1  [parallel, ~1-2h]:
#     GPU 0  →  generate_responses  PKU
#     GPU 1  →  generate_responses  HH-RLHF
#     GPU 2  →  generate_responses  Truthy
#     GPU 3  →  idle
#
#   Phase 2  [parallel, ~2-4h]:
#     GPU 0  →  Con-J  PKU   (3 comparisons, sequential)
#     GPU 1  →  Con-J  HH    (3 comparisons, sequential)
#     GPU 2  →  Con-J  Truthy (3 comparisons, sequential)
#     GPU 3  →  idle  (GPT-4o runs on CPU alongside in background)
#
#   Phase 3  [CPU, ~1 min]:
#     aggregate all results
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

source ~/.bashrc
conda activate pdpo

cd /users/jzhao7/PDPO
SCRIPT_DIR="/users/jzhao7/PDPO/panel_eval"

echo "========================================================"
echo "[panel_eval] Job started: $(date)"
echo "[panel_eval] Host: $(hostname)"
echo "[panel_eval] CUDA devices: ${CUDA_VISIBLE_DEVICES:-<not set by scheduler — assigning per process below>}"
echo "========================================================"

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL="Qwen/Qwen2.5-3B"
N_PROMPTS=200        # prompts per dataset
MAX_NEW_TOKENS=256
EPSILON="0.5"
SEED="42"

# Results root — /users now has 37 GB free after checkpoint cleanup
RESULTS_ROOT="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS_ROOT}"

# ── PKU ──────────────────────────────────────────────────────────────────────
PKU_SB="/users/jzhao7/PDPO/stage2_debugging/models/sb_fresh_pku_eps0.5_seed42"
PKU_MAP="/users/jzhao7/PDPO/stage2_debugging/models/map_retrain_pku_eps0.5_seed42"
PKU_MLE="/users/jzhao7/PDPO/stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_pku_eps0.5_s42"
PKU_PROMPTS="/users/jzhao7/PDPO/stage2_debugging/testsets/pku_secure/test_pref.jsonl"
PKU_OUT="${RESULTS_ROOT}/pku_eps${EPSILON}_seed${SEED}"

# ── HH-RLHF ──────────────────────────────────────────────────────────────────
HH_SB="/users/jzhao7/PDPO/stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42"
HH_MAP="/users/jzhao7/PDPO/stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42"
HH_MLE="/users/jzhao7/PDPO/stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_hhrlhf_eps0.5_s42"
HH_PROMPTS="/users/jzhao7/PDPO/stage2_debugging/testsets/hhrlhf_test_pref_clean_v2.jsonl"
HH_OUT="${RESULTS_ROOT}/hhrlhf_eps${EPSILON}_seed${SEED}"

# ── TruthyDPO ────────────────────────────────────────────────────────────────
TRUTHY_SB="/users/jzhao7/PDPO/stage2_debugging/models/sb_fresh_truthy_eps0.5_seed42"
TRUTHY_MAP="/users/jzhao7/PDPO/stage2_debugging/models/map_retrain_truthy_eps0.5_seed42"
TRUTHY_MLE="/users/jzhao7/PDPO/stage2_debugging/stage2_truthy/eps_sweep_seed42/mle_fresh_truthy_eps0.5_s42"
TRUTHY_PROMPTS="/users/jzhao7/PDPO/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"
TRUTHY_OUT="${RESULTS_ROOT}/truthy_eps${EPSILON}_seed${SEED}"
# Note: Truthy only has 100 held-out prompts; N_PROMPTS capped naturally.

mkdir -p "${PKU_OUT}/responses" "${HH_OUT}/responses" "${TRUTHY_OUT}/responses"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Generate responses (3 datasets in parallel, GPUs 0/1/2)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "[Phase 1] Generating responses — $(date)"
echo "========================================================"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/generate_responses.py" \
    --base_model "${BASE_MODEL}" \
    --adapters "sb_fresh=${PKU_SB},map_retrain=${PKU_MAP},mle_fresh=${PKU_MLE}" \
    --prompts "${PKU_PROMPTS}" \
    --output_dir "${PKU_OUT}/responses" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --n ${N_PROMPTS} \
    > "${PKU_OUT}/responses/gen.log" 2>&1 &
PKU_GEN_PID=$!
echo "[Phase 1] PKU    generation started on GPU 0 (pid=${PKU_GEN_PID})"

CUDA_VISIBLE_DEVICES=1 python "${SCRIPT_DIR}/generate_responses.py" \
    --base_model "${BASE_MODEL}" \
    --adapters "sb_fresh=${HH_SB},map_retrain=${HH_MAP},mle_fresh=${HH_MLE}" \
    --prompts "${HH_PROMPTS}" \
    --output_dir "${HH_OUT}/responses" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --n ${N_PROMPTS} \
    > "${HH_OUT}/responses/gen.log" 2>&1 &
HH_GEN_PID=$!
echo "[Phase 1] HHRLHF generation started on GPU 1 (pid=${HH_GEN_PID})"

CUDA_VISIBLE_DEVICES=2 python "${SCRIPT_DIR}/generate_responses.py" \
    --base_model "${BASE_MODEL}" \
    --adapters "sb_fresh=${TRUTHY_SB},map_retrain=${TRUTHY_MAP},mle_fresh=${TRUTHY_MLE}" \
    --prompts "${TRUTHY_PROMPTS}" \
    --output_dir "${TRUTHY_OUT}/responses" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --n ${N_PROMPTS} \
    > "${TRUTHY_OUT}/responses/gen.log" 2>&1 &
TRUTHY_GEN_PID=$!
echo "[Phase 1] Truthy  generation started on GPU 2 (pid=${TRUTHY_GEN_PID})"

# Wait for all generation to complete
echo "[Phase 1] Waiting for all generation jobs to finish..."
wait ${PKU_GEN_PID}    && echo "[Phase 1] PKU    generation DONE" \
    || { echo "[Phase 1] PKU    generation FAILED — check ${PKU_OUT}/responses/gen.log"; exit 1; }
wait ${HH_GEN_PID}    && echo "[Phase 1] HHRLHF generation DONE" \
    || { echo "[Phase 1] HHRLHF generation FAILED — check ${HH_OUT}/responses/gen.log"; exit 1; }
wait ${TRUTHY_GEN_PID} && echo "[Phase 1] Truthy  generation DONE" \
    || { echo "[Phase 1] Truthy  generation FAILED — check ${TRUTHY_OUT}/responses/gen.log"; exit 1; }

echo "[Phase 1] All generation complete — $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# Helper: judge one comparison with Con-J on a given GPU
# ─────────────────────────────────────────────────────────────────────────────

judge_conj() {
    local DATASET_OUT="$1"   # e.g. ${PKU_OUT}
    local NAME_A="$2"        # e.g. sb_fresh
    local NAME_B="$3"        # e.g. map_retrain
    local GPU_ID="$4"        # 0|1|2
    local COMP_DIR="${DATASET_OUT}/${NAME_A}_vs_${NAME_B}"
    mkdir -p "${COMP_DIR}"
    echo "  [Con-J] ${NAME_A} vs ${NAME_B} on GPU ${GPU_ID}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} python "${SCRIPT_DIR}/judge_panel.py" \
        --responses_a "${DATASET_OUT}/responses/${NAME_A}_responses.jsonl" \
        --responses_b "${DATASET_OUT}/responses/${NAME_B}_responses.jsonl" \
        --output_dir "${COMP_DIR}" \
        --judges con_j \
        > "${COMP_DIR}/conj.log" 2>&1
}

# Helper: judge one comparison with GPT-4o (CPU, no GPU allocation needed)
judge_gpt4o() {
    local DATASET_OUT="$1"
    local NAME_A="$2"
    local NAME_B="$3"
    local COMP_DIR="${DATASET_OUT}/${NAME_A}_vs_${NAME_B}"
    mkdir -p "${COMP_DIR}"
    echo "  [GPT-4o] ${NAME_A} vs ${NAME_B}"
    CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/judge_panel.py" \
        --responses_a "${DATASET_OUT}/responses/${NAME_A}_responses.jsonl" \
        --responses_b "${DATASET_OUT}/responses/${NAME_B}_responses.jsonl" \
        --output_dir "${COMP_DIR}" \
        --judges gpt4o \
        > "${COMP_DIR}/gpt4o.log" 2>&1
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Judge panel (Con-J on GPUs 0/1/2 by dataset, GPT-4o on CPU)
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "[Phase 2] Judging — $(date)"
echo "========================================================"

# ── GPU 0: PKU — 3 comparisons, sequential on GPU 0 ──────────────────────────
(
    judge_conj "${PKU_OUT}" "sb_fresh" "map_retrain" 0
    judge_conj "${PKU_OUT}" "sb_fresh" "mle_fresh"   0
    judge_conj "${PKU_OUT}" "map_retrain" "mle_fresh" 0
    echo "[Phase 2] PKU    Con-J all comparisons DONE"
) > "${PKU_OUT}/conj_all.log" 2>&1 &
PKU_CONJ_PID=$!

# ── GPU 1: HH-RLHF — 3 comparisons, sequential on GPU 1 ─────────────────────
(
    judge_conj "${HH_OUT}" "sb_fresh" "map_retrain" 1
    judge_conj "${HH_OUT}" "sb_fresh" "mle_fresh"   1
    judge_conj "${HH_OUT}" "map_retrain" "mle_fresh" 1
    echo "[Phase 2] HHRLHF Con-J all comparisons DONE"
) > "${HH_OUT}/conj_all.log" 2>&1 &
HH_CONJ_PID=$!

# ── GPU 2: Truthy — 3 comparisons, sequential on GPU 2 ───────────────────────
(
    judge_conj "${TRUTHY_OUT}" "sb_fresh" "map_retrain" 2
    judge_conj "${TRUTHY_OUT}" "sb_fresh" "mle_fresh"   2
    judge_conj "${TRUTHY_OUT}" "map_retrain" "mle_fresh" 2
    echo "[Phase 2] Truthy  Con-J all comparisons DONE"
) > "${TRUTHY_OUT}/conj_all.log" 2>&1 &
TRUTHY_CONJ_PID=$!

# ── CPU: GPT-4o — all 9 comparisons in background while Con-J runs ───────────
# (API calls, no GPU; runs completely in parallel with Con-J blocks above)
echo "[Phase 2] GPT-4o judging starting on CPU..."
(
    for DS_OUT in "${PKU_OUT}" "${HH_OUT}" "${TRUTHY_OUT}"; do
        judge_gpt4o "${DS_OUT}" "sb_fresh"   "map_retrain"
        judge_gpt4o "${DS_OUT}" "sb_fresh"   "mle_fresh"
        judge_gpt4o "${DS_OUT}" "map_retrain" "mle_fresh"
    done
    echo "[Phase 2] GPT-4o all comparisons DONE"
) > "${RESULTS_ROOT}/gpt4o_all.log" 2>&1 &
GPT4O_PID=$!

echo "[Phase 2] PKU Con-J pid=${PKU_CONJ_PID} | HH Con-J pid=${HH_CONJ_PID} | Truthy Con-J pid=${TRUTHY_CONJ_PID} | GPT-4o pid=${GPT4O_PID}"
echo "[Phase 2] Waiting for all judge jobs..."

wait ${PKU_CONJ_PID}    && echo "[Phase 2] PKU    Con-J DONE"   || echo "[Phase 2] PKU    Con-J FAILED — check ${PKU_OUT}/conj_all.log"
wait ${HH_CONJ_PID}     && echo "[Phase 2] HHRLHF Con-J DONE"   || echo "[Phase 2] HHRLHF Con-J FAILED — check ${HH_OUT}/conj_all.log"
wait ${TRUTHY_CONJ_PID} && echo "[Phase 2] Truthy  Con-J DONE"  || echo "[Phase 2] Truthy  Con-J FAILED — check ${TRUTHY_OUT}/conj_all.log"
wait ${GPT4O_PID}        && echo "[Phase 2] GPT-4o  DONE"        || echo "[Phase 2] GPT-4o  FAILED — check ${RESULTS_ROOT}/gpt4o_all.log"

echo "[Phase 2] All judging complete — $(date)"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Aggregate
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "[Phase 3] Aggregating — $(date)"
echo "========================================================"

for DS_OUT in "${PKU_OUT}" "${HH_OUT}" "${TRUTHY_OUT}"; do
    for COMP in "sb_fresh_vs_map_retrain" "sb_fresh_vs_mle_fresh" "map_retrain_vs_mle_fresh"; do
        COMP_DIR="${DS_OUT}/${COMP}"
        if [ -d "${COMP_DIR}" ]; then
            echo "  Aggregating ${COMP_DIR}"
            python "${SCRIPT_DIR}/aggregate_panel.py" \
                --input_dir "${COMP_DIR}" \
                > "${COMP_DIR}/aggregate.log" 2>&1
        fi
    done
done

echo ""
echo "========================================================"
echo "[panel_eval] ALL DONE — $(date)"
echo "========================================================"
echo ""
echo "Results summary:"
for DS_OUT in "${PKU_OUT}" "${HH_OUT}" "${TRUTHY_OUT}"; do
    DS_NAME=$(basename "${DS_OUT}")
    echo "  Dataset: ${DS_NAME}"
    for COMP in "sb_fresh_vs_map_retrain" "sb_fresh_vs_mle_fresh" "map_retrain_vs_mle_fresh"; do
        SUMMARY="${DS_OUT}/${COMP}/panel_summary.json"
        if [ -f "${SUMMARY}" ]; then
            python3 -c "
import json, sys
d = json.load(open('${SUMMARY}'))
pm = d['panel_majority']
print(f'    {\"${COMP}\":35}  A={pm[\"a_wins\"]} B={pm[\"b_wins\"]} TIE={pm[\"ties\"]}  A_rate={pm[\"a_win_rate\"]}')
" 2>/dev/null || echo "    ${COMP}: (parse error)"
        else
            echo "    ${COMP}: (no summary)"
        fi
    done
done
