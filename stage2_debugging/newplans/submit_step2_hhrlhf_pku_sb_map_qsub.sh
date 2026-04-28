#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
cd "${ROOT_DIR}"

CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAIL_USER="${MAIL_USER:-jzhao7@nd.edu}"
H_RT="${H_RT:-07:00:00}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_RERUN_HHRLHF_SB="${FORCE_RERUN_HHRLHF_SB:-1}"
D2_FRACTION="${D2_FRACTION:-0.5}"
SUBSET_SEED="${SUBSET_SEED:-42}"

SCRIPT="${ROOT_DIR}/stage2_debugging/newplans/run_step2_hhrlhf_pku_sb_map_on_4gpus.sh"
[ -x "${SCRIPT}" ] || chmod +x "${SCRIPT}"

for ds in hhrlhf pku; do
  s1="${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${ds}_eps${EPS}_s${SEED}/adapter_model.safetensors"
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  [ -f "${s1}" ] || { echo "[FATAL] Missing stage1 adapter: ${s1}"; exit 2; }
  [ -f "${d2}" ] || { echo "[FATAL] Missing D2 data: ${d2}"; exit 2; }
done

command -v qsub >/dev/null 2>&1 || { echo "[FATAL] qsub not found in PATH"; exit 2; }

QSUB_OUT="${ROOT_DIR}/stage2_debugging/newplans/qsub_s2_hhrlhf_pku_sbmap_eps${EPS}_s${SEED}.out"
QSUB_ERR="${ROOT_DIR}/stage2_debugging/newplans/qsub_s2_hhrlhf_pku_sbmap_eps${EPS}_s${SEED}.err"
JOB_NAME="s2_hp_sbmap_eps${EPS}_s${SEED}"

echo "[INFO] Submitting HH+PKU SB-Fresh/MAP-Retrain job (4 GPUs)..."
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
  -v ROOT_DIR="${ROOT_DIR}",CONDA_ENV="${CONDA_ENV}",BASE_MODEL="${BASE_MODEL}",EPS="${EPS}",SEED="${SEED}",SAVE_STEPS="${SAVE_STEPS}",SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT}",SKIP_EXISTING="${SKIP_EXISTING}",FORCE_RERUN_HHRLHF_SB="${FORCE_RERUN_HHRLHF_SB}",D2_FRACTION="${D2_FRACTION}",SUBSET_SEED="${SUBSET_SEED}" \
  "${SCRIPT}")

echo "[INFO] Submitted job ID: ${JOB_ID}"
echo "[INFO] Monitor:"
echo "  qstat -j ${JOB_ID}"
echo "  tail -f ${QSUB_OUT}"
echo "  tail -f ${QSUB_ERR}"
echo "[INFO] D2 subset fraction: ${D2_FRACTION} (seed ${SUBSET_SEED})"
echo
echo "[INFO] Output summaries:"
echo "  /users/jzhao7/PDPO/stage2_debugging/stage2_hhrlhf/results_eps${EPS}_seed${SEED}/SBFRESH_MAPRETRAIN_HHRLHF_SUMMARY.md"
echo "  /users/jzhao7/PDPO/stage2_debugging/stage2_pku/results_eps${EPS}_seed${SEED}/SBFRESH_MAPRETRAIN_PKU_SUMMARY.md"

