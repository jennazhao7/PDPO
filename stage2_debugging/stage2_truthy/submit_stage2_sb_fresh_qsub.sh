#!/bin/bash
# Submit the SB-Fresh Stage-2 variant as a single-GPU qsub job.
#
# Usage:
#   DATASET=truthy bash submit_stage2_sb_fresh_qsub.sh
#   DATASET=hhrlhf bash submit_stage2_sb_fresh_qsub.sh
#   DATASET=pku    bash submit_stage2_sb_fresh_qsub.sh
#
# Runs independently of the 4-variant job — can be submitted in parallel or later.
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
DATASET="${DATASET:-truthy}"   # truthy | hhrlhf | pku
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-05:00:00}"       # 5 h: scoring + 3 epochs + eval on single GPU
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# ── Resolve paths ─────────────────────────────────────────────────────────
case "${DATASET}" in
  truthy)
    DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"
    ;;
  hhrlhf)
    DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/test_pref.jsonl"
    ;;
  pku)
    DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl"
    ;;
  *)
    echo "[FATAL] Unsupported DATASET=${DATASET}. Use one of: truthy, hhrlhf, pku"
    exit 2
    ;;
esac

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${DATASET}_eps${EPS}_s${SEED}}"
D2_DATA="${D2_DATA:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${DATASET}_eps${EPS}_seed${SEED}.jsonl}"
TEST_JSONL="${TEST_JSONL:-${DEFAULT_TEST}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage2_${DATASET}/results_eps${EPS}_seed${SEED}}"

SCRIPT="${ROOT_DIR}/stage2_debugging/stage2_truthy/run_stage2_sb_fresh_on_1gpu.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

# ── Preflight ─────────────────────────────────────────────────────────────
for f in "${S1_ADAPTER}/adapter_model.safetensors" "${D2_DATA}" "${TEST_JSONL}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing file: ${f}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }
mkdir -p "${OUT_ROOT}"

JOB_NAME="s2_sbfresh_${DATASET}_eps${EPS}_s${SEED}"
QSUB_OUT="${OUT_ROOT}/qsub_${JOB_NAME}.out"
QSUB_ERR="${OUT_ROOT}/qsub_${JOB_NAME}.err"

echo "[INFO] Submitting SB-Fresh job (1 GPU, ${H_RT}) for DATASET=${DATASET} EPS=${EPS} SEED=${SEED}..."

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
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",DATASET="${DATASET}",EPS="${EPS}",SEED="${SEED}",S1_ADAPTER="${S1_ADAPTER}",D2_DATA="${D2_DATA}",TEST_JSONL="${TEST_JSONL}",OUT_ROOT="${OUT_ROOT}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo ""
echo "[Monitor]"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
echo ""
echo "[Results will be written to]"
echo "  ${OUT_ROOT}/sb_fresh_${DATASET}_eps${EPS}_s${SEED}/"
echo "  ${OUT_ROOT}/eval/sb_fresh_${DATASET}_eps${EPS}_s${SEED}_eval.json"
echo "  ${OUT_ROOT}/STAGE2_${DATASET^^}_SUMMARY_WITH_SBFRESH.md"
