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
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}}"

DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/test_pref.jsonl"
TRUTHY_TEST="${TRUTHY_TEST:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"
HHRLHF_TEST="${HHRLHF_TEST:-${ROOT_DIR}/stage2_debugging/test_pref.jsonl}"
PKU_TEST="${PKU_TEST:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

SCRIPT="${ROOT_DIR}/stage2_debugging/stage1/run_stage1_3datasets_on_3gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

resolve_test_path() {
  local requested="$1"
  local fallback="$2"
  local label="$3"
  if [[ -f "${requested}" ]]; then
    echo "${requested}"
    return
  fi
  if [[ -f "${fallback}" ]]; then
    echo "[WARN] ${label} test missing at ${requested}; falling back to ${fallback}" >&2
    echo "${fallback}"
    return
  fi
  echo "[FATAL] ${label} test missing. Tried: ${requested} and fallback: ${fallback}" >&2
  exit 2
}

TRUTHY_TEST="$(resolve_test_path "${TRUTHY_TEST}" "${DEFAULT_TEST}" "TruthyDPO")"
HHRLHF_TEST="$(resolve_test_path "${HHRLHF_TEST}" "${DEFAULT_TEST}" "HH-RLHF")"
PKU_TEST="$(resolve_test_path "${PKU_TEST}" "${DEFAULT_TEST}" "PKU-SafeRLHF")"

for f in \
  "${PREP_DIR}/d1_rr_flipped_truthy_eps${EPS}_seed${SEED}.jsonl" \
  "${PREP_DIR}/d1_rr_flipped_hhrlhf_eps${EPS}_seed${SEED}.jsonl" \
  "${PREP_DIR}/d1_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl" \
  "${TRUTHY_TEST}" "${HHRLHF_TEST}" "${PKU_TEST}"
do
  [ -f "${f}" ] || { echo "[FATAL] Missing file: ${f}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

mkdir -p "${OUT_ROOT}"

JOB_NAME="s1_3ds_eps${EPS}_s${SEED}"
QSUB_OUT="${OUT_ROOT}/qsub_${JOB_NAME}.out"
QSUB_ERR="${OUT_ROOT}/qsub_${JOB_NAME}.err"

echo "[INFO] Submitting one qsub job that uses 3 GPUs..."
JOB_ID=$(qsub -terse \
  -q gpu@@jung_gpu \
  -l gpu_card=3 \
  -pe smp 12 \
  -l h_rt=05:00:00 \
  -M "${MAIL_USER}" \
  -m bea \
  -N "${JOB_NAME}" \
  -o "${QSUB_OUT}" \
  -e "${QSUB_ERR}" \
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",EPS="${EPS}",SEED="${SEED}",MAX_STEPS="${MAX_STEPS}",TRAIN_MAX_LEN="${TRAIN_MAX_LEN}",EVAL_MAX_LEN="${EVAL_MAX_LEN}",SAVE_STRATEGY="${SAVE_STRATEGY}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",PREP_DIR="${PREP_DIR}",OUT_ROOT="${OUT_ROOT}",TRUTHY_TEST="${TRUTHY_TEST}",HHRLHF_TEST="${HHRLHF_TEST}",PKU_TEST="${PKU_TEST}" \
  "${SCRIPT}")

echo "[INFO] Submitted job: ${JOB_ID}"
echo "[INFO] Monitor with:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
