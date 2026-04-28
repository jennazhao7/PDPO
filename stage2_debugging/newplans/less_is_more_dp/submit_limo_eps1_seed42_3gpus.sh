#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/run_limo_eps1_seed42_3datasets_3gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
TAU_DROP="${TAU_DROP:-0.10}"
MODE="${MODE:-both}"
H_RT="${H_RT:-24:00:00}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/results_eps${EPS}_seed${SEED}}"

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/qsub_limo_${MODE}_tau${TAU_DROP}.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/qsub_limo_${MODE}_tau${TAU_DROP}.err"
JOB_NAME="limo_${MODE}_t${TAU_DROP}"

JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=3 \
  -pe smp 12 \
  -l h_rt="${H_RT}" \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",EPS="${EPS}",SEED="${SEED}",TAU_DROP="${TAU_DROP}",MODE="${MODE}",OUT_ROOT="${OUT_ROOT}" \
  "${SCRIPT}")

echo "[INFO] Submitted: ${JOB_ID}"
echo "[INFO] qstat -j ${JOB_ID}"
echo "[INFO] tail -f ${QSUB_OUT}"

