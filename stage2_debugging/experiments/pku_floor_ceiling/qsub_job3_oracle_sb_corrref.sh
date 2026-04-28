#!/bin/bash
# Job 3 — Oracle SB Interpolated Ref (runs on GPU node; submitted via submit_jobs.sh)
#$ -S /bin/bash

set -euo pipefail
echo "[INFO] host=$(hostname)  date=$(date)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<not set>}"

ROOT_DIR="/users/jzhao7/PDPO"
CONDA_ENV="${CONDA_ENV:-pdpo}"
EXP="${ROOT_DIR}/stage2_debugging/experiments/pku_floor_ceiling"
M1="${ROOT_DIR}/stage2_debugging/stage1/results_instruct_eps1.0_seed42/pku_eps1.0_s42"

source ~/.bashrc
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${ROOT_DIR}"
echo "[Job 3] Training Oracle SB Interpolated Ref..."

python ${EXP}/train_job3_oracle_sb_corrref.py \
  --model Qwen/Qwen2.5-3B-Instruct \
  --m1    ${M1} \
  --data  ${EXP}/d2_pku_with_flipped_eps1.0_seed42.jsonl \
  --out   ${EXP}/models/pku_oracle_sb_corrref \
  --epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4

echo "[Job 3] Training done. Running eval..."

python stage2_debugging/eval_preference_accuracy.py \
  --manifest   ${EXP}/models/pku_oracle_sb_corrref/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/pku_secure/test_pref.jsonl \
  --out_json   ${EXP}/results/job3.json

python3 -c "import json; d=json.load(open('${EXP}/results/job3.json')); print(f'[RESULT] Job 3 (Oracle SB interp ref, method): {d[\"accuracy\"]:.4f}')"
echo "[Job 3] Done at $(date)"
