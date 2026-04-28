#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-18:00:00}"

DATA_JSONL="${DATA_JSONL:-${ROOT_DIR}/preprocessing/truthydpo/truthy_dpo_subset.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/results_truthy_dpsgd_plain_seed${SEED}}"

DELTA="${DELTA:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EPOCHS="${EPOCHS:-3}"
LR_DP="${LR_DP:-5e-5}"
LR_PLAIN="${LR_PLAIN:-5e-5}"
DP_PER_DEVICE_BSZ="${DP_PER_DEVICE_BSZ:-1}"
DP_GRAD_ACCUM="${DP_GRAD_ACCUM:-32}"
PLAIN_PER_DEVICE_BSZ="${PLAIN_PER_DEVICE_BSZ:-1}"
PLAIN_GRAD_ACCUM="${PLAIN_GRAD_ACCUM:-16}"
PLAIN_MAX_STEPS="${PLAIN_MAX_STEPS:-600}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-200}"
USE_BF16_PLAIN="${USE_BF16_PLAIN:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_truthy_dpsgd_plain_eps_on_2gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for f in "${SCRIPT}" "${DATA_JSONL}" \
         "${ROOT_DIR}/lora/dp-lora/train_dp_lora.py" \
         "${ROOT_DIR}/lora/plain-lora/train_plain_lora.py"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/qsub_truthy_dpsgd_plain_3gpus.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/qsub_truthy_dpsgd_plain_3gpus.err"
JOB_NAME="truthy_dp_plain_3g"

echo "[INFO] Submitting 3-GPU Truthy DP-SGD + plain LoRA sweep..."
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
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",SEED="${SEED}",DATA_JSONL="${DATA_JSONL}",OUT_ROOT="${OUT_ROOT}",DELTA="${DELTA}",MAX_GRAD_NORM="${MAX_GRAD_NORM}",EPOCHS="${EPOCHS}",LR_DP="${LR_DP}",LR_PLAIN="${LR_PLAIN}",DP_PER_DEVICE_BSZ="${DP_PER_DEVICE_BSZ}",DP_GRAD_ACCUM="${DP_GRAD_ACCUM}",PLAIN_PER_DEVICE_BSZ="${PLAIN_PER_DEVICE_BSZ}",PLAIN_GRAD_ACCUM="${PLAIN_GRAD_ACCUM}",PLAIN_MAX_STEPS="${PLAIN_MAX_STEPS}",LORA_R="${LORA_R}",LORA_ALPHA="${LORA_ALPHA}",MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH}",MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES}",USE_BF16_PLAIN="${USE_BF16_PLAIN}",SKIP_EXISTING="${SKIP_EXISTING}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
