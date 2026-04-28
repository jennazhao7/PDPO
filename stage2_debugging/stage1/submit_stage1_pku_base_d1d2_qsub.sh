#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-384}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-512}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"

PREP_DIR="${PREP_DIR:-${ROOT_DIR}/stage2_debugging/preprocessing}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage1/results_base_d1d2_eps${EPS}_seed${SEED}}"
PKU_TEST="${PKU_TEST:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

SCRIPT="${ROOT_DIR}/stage2_debugging/stage1/run_stage1_pku_base_d1d2_1gpu.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

command -v qsub > /dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

mkdir -p "${OUT_ROOT}"

JOB_NAME="s1_pku_base_d1d2_eps${EPS}_s${SEED}"
QSUB_OUT="${OUT_ROOT}/qsub_${JOB_NAME}.out"
QSUB_ERR="${OUT_ROOT}/qsub_${JOB_NAME}.err"

echo "[INFO] Submitting PKU Base D1+D2 Stage 1 job (1 GPU)..."
echo "  BASE_MODEL = ${BASE_MODEL}"
echo "  OUT_ROOT   = ${OUT_ROOT}"

JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=1 \
  -pe smp 4 \
  -l h_rt=04:00:00 \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",EPS="${EPS}",SEED="${SEED}",MAX_STEPS="${MAX_STEPS}",TRAIN_MAX_LEN="${TRAIN_MAX_LEN}",EVAL_MAX_LEN="${EVAL_MAX_LEN}",SAVE_STRATEGY="${SAVE_STRATEGY}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",PREP_DIR="${PREP_DIR}",OUT_ROOT="${OUT_ROOT}",PKU_TEST="${PKU_TEST}" \
  "${SCRIPT}")

echo "[INFO] Submitted job: ${JOB_ID}"
echo "[INFO] Monitor with:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
