#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
SEED="${SEED:-42}"
EPS="${EPS:-1.0}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-08:00:00}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_LEN="${MAX_LEN:-512}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/crossdata_eval/truthy_eps${EPS}_s${SEED}_on_hhrlhf}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/test_pref.jsonl}"
MANIFEST_SB_FRESH="${MANIFEST_SB_FRESH:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/sb_fresh_truthy_eps1.0_s42/M2_manifest.json}"
MANIFEST_MAP_RETRAIN="${MANIFEST_MAP_RETRAIN:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/map_retrain_truthy_eps1.0_s42/M2_manifest.json}"
MANIFEST_MLE_DPO="${MANIFEST_MLE_DPO:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/mle_dpo_truthy_eps1.0_s42/M2_manifest.json}"
MANIFEST_SOFT_BAYES="${MANIFEST_SOFT_BAYES:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/soft_bayes_truthy_eps1.0_s42/M2_manifest.json}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_crossdata_eval_truthy_eps1_on_hhrlhf_4gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for f in "${SCRIPT}" "${ROOT_DIR}/stage2_debugging/eval_preference_accuracy.py" "${TEST_JSONL}" \
         "${MANIFEST_SB_FRESH}" "${MANIFEST_MAP_RETRAIN}" "${MANIFEST_MLE_DPO}" "${MANIFEST_SOFT_BAYES}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/qsub_crossdata_truthy_on_hhrlhf_4gpus.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/qsub_crossdata_truthy_on_hhrlhf_4gpus.err"
JOB_NAME="xeval_tyhhrl_4g"

echo "[INFO] Submitting cross-data eval (4 GPUs)..."
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
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",SEED="${SEED}",EPS="${EPS}",OUT_ROOT="${OUT_ROOT}",SKIP_EXISTING="${SKIP_EXISTING}",MAX_LEN="${MAX_LEN}",TEST_JSONL="${TEST_JSONL}",MANIFEST_SB_FRESH="${MANIFEST_SB_FRESH}",MANIFEST_MAP_RETRAIN="${MANIFEST_MAP_RETRAIN}",MANIFEST_MLE_DPO="${MANIFEST_MLE_DPO}",MANIFEST_SOFT_BAYES="${MANIFEST_SOFT_BAYES}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
