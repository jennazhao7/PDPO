#!/bin/bash
# Job 2 — Oracle SB Base Ref (runs on GPU node; submitted via submit_jobs.sh)
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
echo "[Job 2] Training Oracle SB Base Ref..."

python ${EXP}/train_job2_oracle_sb_baseref.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --data  ${EXP}/d2_pku_with_flipped_eps1.0_seed42.jsonl \
  --out   ${EXP}/models/pku_oracle_sb_baseref \
  --epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4

echo "[Job 2] Training done. Running eval..."

python stage2_debugging/eval_preference_accuracy.py \
  --manifest   ${EXP}/models/pku_oracle_sb_baseref/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/pku_secure/test_pref.jsonl \
  --out_json   ${EXP}/results/job2.json

python3 -c "import json; d=json.load(open('${EXP}/results/job2.json')); print(f'[RESULT] Job 2 (Oracle SB base ref, ceiling): {d[\"accuracy\"]:.4f}')"
echo "[Job 2] Done at $(date)"
