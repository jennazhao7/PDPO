#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
S1_EPS="${S1_EPS:-1.0}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-09:00:00}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
D2_FRACTION="${D2_FRACTION:-0.5}"
SUBSET_SEED="${SUBSET_SEED:-42}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_stage2_hhpku_eps05_plus_mle_on_gpus123.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/qsub_s2_hhpku_eps05_plus_mle_s${SEED}.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/qsub_s2_hhpku_eps05_plus_mle_s${SEED}.err"
JOB_NAME="s2_hhpku_e05_mle_s${SEED}"

echo "[INFO] Submitting HH/PKU eps0.5 + MLE extension job (GPUs 1/2/3)..."
JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=2 \
  -pe smp 8 \
  -l h_rt="${H_RT}" \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",SEED="${SEED}",S1_EPS="${S1_EPS}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",SKIP_EXISTING="${SKIP_EXISTING}",D2_FRACTION="${D2_FRACTION}",SUBSET_SEED="${SUBSET_SEED}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
echo "[INFO] D2 subset fraction: ${D2_FRACTION} (seed ${SUBSET_SEED})"
