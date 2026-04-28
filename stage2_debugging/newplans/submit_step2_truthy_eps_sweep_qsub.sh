#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
S1_EPS="${S1_EPS:-1.0}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-08:00:00}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${S1_EPS}_seed${SEED}/truthy_eps${S1_EPS}_s${SEED}}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage2_truthy/eps_sweep_seed${SEED}}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_step2_truthy_eps_sweep_on_4gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for f in "${S1_ADAPTER}/adapter_model.safetensors" "${TEST_JSONL}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing file: ${f}"; exit 2; }
done
for eps in 0.3 0.5 1.0 2.0; do
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${eps}_seed${SEED}.jsonl"
  [ -f "${d2}" ] || { echo "[FATAL] Missing D2 file: ${d2}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }
mkdir -p "${OUT_ROOT}"

JOB_NAME="s2_truthy_sweep_s${SEED}"
QSUB_OUT="${OUT_ROOT}/qsub_${JOB_NAME}.out"
QSUB_ERR="${OUT_ROOT}/qsub_${JOB_NAME}.err"

echo "[INFO] Submitting truthy eps sweep (4 GPUs)..."
JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=4 \
  -pe smp 16 \
  -l h_rt="${H_RT}" \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",SEED="${SEED}",S1_EPS="${S1_EPS}",S1_ADAPTER="${S1_ADAPTER}",TEST_JSONL="${TEST_JSONL}",OUT_ROOT="${OUT_ROOT}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",SKIP_EXISTING="${SKIP_EXISTING}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
echo "[INFO] Final sweep table:"
echo "  ${OUT_ROOT}/TRUTHY_EPS_SWEEP_S42.md"

