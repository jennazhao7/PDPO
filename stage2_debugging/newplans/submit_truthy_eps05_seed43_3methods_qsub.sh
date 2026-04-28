#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-43}"
EPS="${EPS:-0.5}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-08:00:00}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps1.0_seed43_truthy/truthy_eps1.0_s43}"
D2_JSONL="${D2_JSONL:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed43.jsonl}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_truthy_eps05_seed43_3methods_on_3gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for f in \
  "${S1_ADAPTER}/adapter_model.safetensors" \
  "${D2_JSONL}" \
  "${TEST_JSONL}"
do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/qsub_s2_truthy_eps05_seed43_3methods.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/qsub_s2_truthy_eps05_seed43_3methods.err"
JOB_NAME="s2_truthy_e05_s43_3m"

echo "[INFO] Submitting Truthy eps0.5 seed43 (SB/MAP/MLE on 3 GPUs)..."
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
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",SEED="${SEED}",EPS="${EPS}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",SKIP_EXISTING="${SKIP_EXISTING}",S1_ADAPTER="${S1_ADAPTER}",D2_JSONL="${D2_JSONL}",TEST_JSONL="${TEST_JSONL}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
