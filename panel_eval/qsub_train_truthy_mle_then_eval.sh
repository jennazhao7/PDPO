#!/bin/bash
#$ -N truthy_mle_then_eval
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4
#$ -pe smp 8
#$ -l h_rt=10:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

# Phase 0: Train the missing Truthy MLE-Fresh adapter (~15 min on 1 GPU)
# Then run the full recovery eval (Truthy gen + all Con-J/GPT-4o judging)

set -euo pipefail

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

SCRIPT_DIR="/users/jzhao7/PDPO/panel_eval"
RESULTS_ROOT="${SCRIPT_DIR}/results"

echo "========================================================"
echo "[job] Started: $(date)"
echo "[job] Host: $(hostname)"
echo "========================================================"

MLE_OUT="stage2_debugging/stage2_truthy/eps_sweep_seed42/mle_fresh_truthy_eps0.5_s42"
MLE_ADAPTER="${MLE_OUT}/fresh_lora"
S1_TRUTHY="stage2_debugging/stage1/results_eps0.5_seed42/truthy_eps0.5_s42"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 0: Train Truthy MLE-Fresh (GPU 0 only, ~15 min)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase 0] Training Truthy MLE-Fresh — $(date)"
CUDA_VISIBLE_DEVICES=0 python stage2_debugging/train_stage2_mle_fresh.py \
    --model "Qwen/Qwen2.5-3B" \
    --data "stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed42.jsonl" \
    --out "${MLE_OUT}" \
    --lr 2.5e-5 --epochs 1 --bsz 1 --ga 16 --max_len 1024 --bf16 \
    --seed 42
echo "[Phase 0] MLE-Fresh training DONE — $(date)"

# Verify weights were written
if [ ! -f "${MLE_ADAPTER}/adapter_model.safetensors" ]; then
    echo "[Phase 0] ERROR: adapter_model.safetensors not found in ${MLE_ADAPTER}"
    exit 1
fi
echo "[Phase 0] Verified: adapter_model.safetensors exists"

# ─────────────────────────────────────────────────────────────────────────────
# Now run the recovery eval — same logic as qsub_panel_eval_recovery.sh
# ─────────────────────────────────────────────────────────────────────────────

BASE_MODEL="Qwen/Qwen2.5-3B"
N_PROMPTS=200
MAX_NEW_TOKENS=256
EPSILON="0.5"
SEED="42"

PKU_OUT="${RESULTS_ROOT}/pku_eps${EPSILON}_seed${SEED}"
HH_OUT="${RESULTS_ROOT}/hhrlhf_eps${EPSILON}_seed${SEED}"
TRUTHY_OUT="${RESULTS_ROOT}/truthy_eps${EPSILON}_seed${SEED}"

TRUTHY_SB="/users/jzhao7/PDPO/stage2_debugging/models/sb_fresh_truthy_eps0.5_seed42"
TRUTHY_MAP="/users/jzhao7/PDPO/stage2_debugging/models/map_retrain_truthy_eps0.5_seed42"
TRUTHY_MLE="/users/jzhao7/PDPO/${MLE_OUT}"
TRUTHY_PROMPTS="/users/jzhao7/PDPO/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"

mkdir -p "${TRUTHY_OUT}/responses"

judge_conj() {
    local DATASET_OUT="$1"; local NAME_A="$2"; local NAME_B="$3"; local GPU_ID="$4"
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

judge_gpt4o() {
    local DATASET_OUT="$1"; local NAME_A="$2"; local NAME_B="$3"
    local COMP_DIR="${DATASET_OUT}/${NAME_A}_vs_${NAME_B}"
    mkdir -p "${COMP_DIR}"
    CUDA_VISIBLE_DEVICES="" python "${SCRIPT_DIR}/judge_panel.py" \
        --responses_a "${DATASET_OUT}/responses/${NAME_A}_responses.jsonl" \
        --responses_b "${DATASET_OUT}/responses/${NAME_B}_responses.jsonl" \
        --output_dir "${COMP_DIR}" \
        --judges gpt4o \
        > "${COMP_DIR}/gpt4o.log" 2>&1
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase A: Truthy gen (GPU 0) + PKU Con-J (GPU 1) + HH Con-J (GPU 2)
#          + GPT-4o PKU+HH on CPU — all parallel
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase A] Truthy gen + PKU/HH judging in parallel — $(date)"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/generate_responses.py" \
    --base_model "${BASE_MODEL}" \
    --adapters "sb_fresh=${TRUTHY_SB},map_retrain=${TRUTHY_MAP},mle_fresh=${TRUTHY_MLE}" \
    --prompts "${TRUTHY_PROMPTS}" \
    --output_dir "${TRUTHY_OUT}/responses" \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --n ${N_PROMPTS} \
    > "${TRUTHY_OUT}/responses/gen.log" 2>&1 &
TRUTHY_GEN_PID=$!
echo "[Phase A] Truthy gen on GPU 0 (pid=${TRUTHY_GEN_PID})"

(
    judge_conj "${PKU_OUT}" "sb_fresh" "map_retrain" 1
    judge_conj "${PKU_OUT}" "sb_fresh" "mle_fresh"   1
    judge_conj "${PKU_OUT}" "map_retrain" "mle_fresh" 1
    echo "[Phase A] PKU Con-J DONE"
) > "${PKU_OUT}/conj_all.log" 2>&1 &
PKU_CONJ_PID=$!

(
    judge_conj "${HH_OUT}" "sb_fresh" "map_retrain" 2
    judge_conj "${HH_OUT}" "sb_fresh" "mle_fresh"   2
    judge_conj "${HH_OUT}" "map_retrain" "mle_fresh" 2
    echo "[Phase A] HH Con-J DONE"
) > "${HH_OUT}/conj_all.log" 2>&1 &
HH_CONJ_PID=$!

(
    for DS_OUT in "${PKU_OUT}" "${HH_OUT}"; do
        judge_gpt4o "${DS_OUT}" "sb_fresh"    "map_retrain"
        judge_gpt4o "${DS_OUT}" "sb_fresh"    "mle_fresh"
        judge_gpt4o "${DS_OUT}" "map_retrain" "mle_fresh"
    done
    echo "[Phase A] GPT-4o PKU+HH DONE"
) > "${RESULTS_ROOT}/gpt4o_pku_hh.log" 2>&1 &
GPT4O_PKHH_PID=$!

echo "[Phase A] Waiting for Truthy gen..."
wait ${TRUTHY_GEN_PID} && echo "[Phase A] Truthy gen DONE" \
    || { echo "[Phase A] Truthy gen FAILED — check ${TRUTHY_OUT}/responses/gen.log"; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# Phase B: Truthy Con-J (GPU 0, now free) + GPT-4o Truthy on CPU
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase B] Truthy judging — $(date)"

(
    judge_conj "${TRUTHY_OUT}" "sb_fresh" "map_retrain" 0
    judge_conj "${TRUTHY_OUT}" "sb_fresh" "mle_fresh"   0
    judge_conj "${TRUTHY_OUT}" "map_retrain" "mle_fresh" 0
    echo "[Phase B] Truthy Con-J DONE"
) > "${TRUTHY_OUT}/conj_all.log" 2>&1 &
TRUTHY_CONJ_PID=$!

(
    judge_gpt4o "${TRUTHY_OUT}" "sb_fresh"    "map_retrain"
    judge_gpt4o "${TRUTHY_OUT}" "sb_fresh"    "mle_fresh"
    judge_gpt4o "${TRUTHY_OUT}" "map_retrain" "mle_fresh"
    echo "[Phase B] GPT-4o Truthy DONE"
) > "${RESULTS_ROOT}/gpt4o_truthy.log" 2>&1 &
GPT4O_TRUTHY_PID=$!

echo "[Phase B] Waiting for all remaining jobs..."
wait ${PKU_CONJ_PID}      && echo "PKU    Con-J DONE"   || echo "PKU    Con-J FAILED"
wait ${HH_CONJ_PID}       && echo "HH     Con-J DONE"   || echo "HH     Con-J FAILED"
wait ${TRUTHY_CONJ_PID}   && echo "Truthy Con-J DONE"   || echo "Truthy Con-J FAILED"
wait ${GPT4O_PKHH_PID}    && echo "GPT-4o PKU+HH DONE"  || echo "GPT-4o PKU+HH FAILED"
wait ${GPT4O_TRUTHY_PID}  && echo "GPT-4o Truthy DONE"  || echo "GPT-4o Truthy FAILED"

# ─────────────────────────────────────────────────────────────────────────────
# Phase C: Aggregate
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "[Phase C] Aggregating — $(date)"

for DS_OUT in "${PKU_OUT}" "${HH_OUT}" "${TRUTHY_OUT}"; do
    for COMP in "sb_fresh_vs_map_retrain" "sb_fresh_vs_mle_fresh" "map_retrain_vs_mle_fresh"; do
        COMP_DIR="${DS_OUT}/${COMP}"
        if [ -d "${COMP_DIR}" ]; then
            echo "  Aggregating $(basename ${DS_OUT})/${COMP}"
            python "${SCRIPT_DIR}/aggregate_panel.py" --input_dir "${COMP_DIR}" \
                > "${COMP_DIR}/aggregate.log" 2>&1
        fi
    done
done

echo ""
echo "========================================================"
echo "[job] ALL DONE — $(date)"
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
import json
d = json.load(open('${SUMMARY}'))
pm = d['panel_majority']
print(f'    ${COMP}: A={pm[\"a_wins\"]} B={pm[\"b_wins\"]} TIE={pm[\"ties\"]}  (A_rate={pm[\"a_win_rate\"]:.2f})')
" 2>/dev/null
        else
            echo "    ${COMP}: (no summary)"
        fi
    done
done
