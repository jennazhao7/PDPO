#!/bin/bash
# Job 1 — MLE-Fresh (runs on GPU node; submitted via submit_jobs.sh)
#$ -S /bin/bash

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
echo "[Job 1] Training MLE-Fresh..."

python stage2_debugging/train_stage2_mle_fresh.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data  stage2_debugging/preprocessing/d2_rr_flipped_pku_eps1.0_seed42.jsonl \
  --out   ${EXP}/models/pku_mle_fresh \
  --epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4

echo "[Job 1] Training done. Running eval..."

python stage2_debugging/eval_preference_accuracy.py \
  --manifest   ${EXP}/models/pku_mle_fresh/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/pku_secure/test_pref.jsonl \
  --out_json   ${EXP}/results/job1.json

python3 -c "import json; d=json.load(open('${EXP}/results/job1.json')); print(f'[RESULT] Job 1 (MLE-Fresh, floor): {d[\"accuracy\"]:.4f}')"
echo "[Job 1] Done at $(date)"
