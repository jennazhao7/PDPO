#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"

# Required environment variables passed by the submit script
METHOD="${METHOD}"
BASE_MODEL="${BASE_MODEL}"
STAGE1_ADAPTER="${STAGE1_ADAPTER}"
DATASET_PATH="${DATASET_PATH}"
OUT_DIR="${OUT_DIR}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[FATAL] conda not found in PATH"
  exit 2
fi
cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false

LOG_DIR="${OUT_DIR}/logs"
EVAL_DIR="${OUT_DIR}/eval"
mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${EVAL_DIR}"

echo "[RUN] METHOD=${METHOD} DATA=${DATASET_PATH} OUT=${OUT_DIR}"

LATEST_CKPT=$(ls -d "${OUT_DIR}"/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)
RESUME_FLAG=""
if [[ -n "${LATEST_CKPT}" && -d "${LATEST_CKPT}" ]]; then
  echo "[Resume] Found checkpoint: ${LATEST_CKPT}"
  RESUME_FLAG="--resume_from_checkpoint ${LATEST_CKPT}"
fi

if [[ "${METHOD}" == "sb_fresh" ]]; then
  CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/train_stage2_sb_fresh.py \
    --model "${BASE_MODEL}" \
    --stage1_adapter "${STAGE1_ADAPTER}" \
    --data "${DATASET_PATH}" \
    --out "${OUT_DIR}" \
    --epsilon "${EPS}" \
    --beta 0.5 \
    --lr 2.5e-5 \
    --seed "${SEED}" \
    --epochs 3 \
    --max_steps -1 \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bsz 1 \
    --ga 16 \
    --bf16 ${RESUME_FLAG} \
    >> "${LOG_DIR}/train_sb_fresh_s${SEED}.log" 2>&1
else
  CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/train_stage2_map_retrain.py \
    --model "${BASE_MODEL}" \
    --stage1_adapter "${STAGE1_ADAPTER}" \
    --data "${DATASET_PATH}" \
    --out "${OUT_DIR}" \
    --epsilon "${EPS}" \
    --beta 0.5 \
    --lr 2.5e-5 \
    --seed "${SEED}" \
    --epochs 3 \
    --max_steps -1 \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bsz 1 \
    --ga 16 \
    --bf16 ${RESUME_FLAG} \
    >> "${LOG_DIR}/train_map_retrain_s${SEED}.log" 2>&1
fi

MANIFEST="${OUT_DIR}/M2_manifest.json"
[ -f "${MANIFEST}" ] || { echo "[FATAL] missing manifest after train: ${MANIFEST}"; exit 3; }

CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "${MANIFEST}" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/${METHOD}_pku_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  > "${LOG_DIR}/eval_${METHOD}_s${SEED}.log" 2>&1

echo "[DONE] ${METHOD} on $(basename ${DATASET_PATH})"
