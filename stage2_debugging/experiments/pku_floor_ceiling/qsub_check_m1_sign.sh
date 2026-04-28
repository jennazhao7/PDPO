#!/bin/bash
# Check M1 Sign Diagnostic (Check 1 & 2, ~60-90 min on GPU for N=5082 x 3 models)
#$ -S /bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -l h_rt=2:00:00
#$ -pe smp 4
#$ -M jzhao7@nd.edu
#$ -m bea
#$ -v CONDA_ENV=pdpo

set -euo pipefail
echo "[INFO] host=$(hostname)  date=$(date)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<not set>}"

ROOT_DIR="/users/jzhao7/PDPO"
CONDA_ENV="${CONDA_ENV:-pdpo}"
EXP="${ROOT_DIR}/stage2_debugging/experiments/pku_floor_ceiling"

source ~/.bashrc
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${ROOT_DIR}"
echo "[Check] Running M1 sign diagnostic..."

TOKENIZERS_PARALLELISM=false python ${EXP}/check_m1_sign.py \
  2>&1 | tee ${EXP}/logs/check_m1_sign.log

echo "[Check] Done at $(date)"
