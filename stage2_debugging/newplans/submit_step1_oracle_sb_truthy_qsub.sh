#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-05:00:00}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
ORACLE_DELTA="${ORACLE_DELTA:-10.0}"
TAU="${TAU:-1.0}"

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/truthy_eps${EPS}_s${SEED}}"
D2_DATA="${D2_DATA:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${EPS}_seed${SEED}.jsonl}"
ORACLE_LABELS="${ORACLE_LABELS:-}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/step1_oracle_sb/results_truthy_eps${EPS}_seed${SEED}}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_step1_oracle_sb_truthy_on_1gpu.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for f in "${S1_ADAPTER}/adapter_model.safetensors" "${D2_DATA}" "${TEST_JSONL}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing file: ${f}"; exit 2; }
done
if [[ -n "${ORACLE_LABELS}" ]]; then
  [ -f "${ORACLE_LABELS}" ] || { echo "[FATAL] ORACLE_LABELS set but missing file: ${ORACLE_LABELS}"; exit 2; }
fi

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }
mkdir -p "${OUT_ROOT}"

JOB_NAME="s1_oraclesb_truthy_eps${EPS}_s${SEED}"
QSUB_OUT="${OUT_ROOT}/qsub_${JOB_NAME}.out"
QSUB_ERR="${OUT_ROOT}/qsub_${JOB_NAME}.err"

echo "[INFO] Submitting Step1 Oracle SB job..."
JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=1 \
  -pe smp 4 \
  -l h_rt="${H_RT}" \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",EPS="${EPS}",SEED="${SEED}",S1_ADAPTER="${S1_ADAPTER}",D2_DATA="${D2_DATA}",ORACLE_LABELS="${ORACLE_LABELS}",TEST_JSONL="${TEST_JSONL}",OUT_ROOT="${OUT_ROOT}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",ORACLE_DELTA="${ORACLE_DELTA}",TAU="${TAU}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
echo "[INFO] Result summary:"
echo "  ${OUT_ROOT}/STEP1_ORACLE_SB_SUMMARY.md"
if [[ -z "${ORACLE_LABELS}" ]]; then
  echo "[INFO] ORACLE_LABELS not provided: worker will reconstruct flips from D2 RR ids."
fi

