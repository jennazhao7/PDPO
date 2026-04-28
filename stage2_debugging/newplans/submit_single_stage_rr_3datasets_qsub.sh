#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
EPS="${EPS:-1.0}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-24:00:00}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
BETA="${BETA:-0.5}"
BSZ="${BSZ:-1}"
GA="${GA:-16}"
MAX_LEN="${MAX_LEN:-512}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/single_stage_rr_seed${SEED}_eps${EPS}}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_single_stage_rr_3datasets_on_1gpu.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for f in "${SCRIPT}" \
         "${ROOT_DIR}/stage2_debugging/train_stage2_mle_fresh.py" \
         "${ROOT_DIR}/stage2_debugging/eval_preference_accuracy.py"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/qsub_single_stage_rr_3datasets_eps${EPS}_s${SEED}.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/qsub_single_stage_rr_3datasets_eps${EPS}_s${SEED}.err"
JOB_NAME="single_rr_3ds_e${EPS}_s${SEED}"

echo "[INFO] Submitting single-stage RR (truthy+hhrlhf+pku) on 1 GPU..."
JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=1 \
  -pe smp 6 \
  -l h_rt="${H_RT}" \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",SEED="${SEED}",EPS="${EPS}",OUT_ROOT="${OUT_ROOT}",SKIP_EXISTING="${SKIP_EXISTING}",FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA}",EPOCHS="${EPOCHS}",LR="${LR}",BETA="${BETA}",BSZ="${BSZ}",GA="${GA}",MAX_LEN="${MAX_LEN}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
