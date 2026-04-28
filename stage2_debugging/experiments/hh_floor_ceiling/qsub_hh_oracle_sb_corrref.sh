#!/bin/bash
# HH-RLHF Oracle SB Interpolated Ref — Oracle ceiling experiment
#$ -S /bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -l h_rt=3:00:00
#$ -pe smp 4
#$ -M jzhao7@nd.edu
#$ -m bea
#$ -v CONDA_ENV=pdpo

set -euo pipefail
echo "[INFO] host=$(hostname)  date=$(date)"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<not set>}"

ROOT_DIR="/users/jzhao7/PDPO"
CONDA_ENV="${CONDA_ENV:-pdpo}"
EXP="${ROOT_DIR}/stage2_debugging/experiments/hh_floor_ceiling"
M1="${ROOT_DIR}/stage2_debugging/stage1/results_eps1.0_seed42/hhrlhf_eps1.0_s42"
DATA="${ROOT_DIR}/stage2_debugging/newplans/stage1_followup_hhpku_seed42/data/d2_rr_flipped_hhrlhf_eps1.0_seed42_frac0.5_subseed42.jsonl"
TEST_JSONL="${ROOT_DIR}/stage2_debugging/test_pref.jsonl"

source ~/.bashrc
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi

cd "${ROOT_DIR}"
echo "[HH Oracle SB Interp] Training..."

TOKENIZERS_PARALLELISM=false python ${EXP}/train_oracle_sb_corrref.py \
  --model Qwen/Qwen2.5-3B \
  --m1    ${M1} \
  --data  ${DATA} \
  --out   ${EXP}/models/hh_oracle_sb_corrref \
  --epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4 --bf16 \
  2>&1 | tee ${EXP}/logs/hh_oracle_sb_corrref.log

echo "[HH Oracle SB Interp] Training done. Running eval..."

python stage2_debugging/eval_preference_accuracy.py \
  --manifest   ${EXP}/models/hh_oracle_sb_corrref/M2_manifest.json \
  --test_jsonl ${TEST_JSONL} \
  --out_json   ${EXP}/results/hh_oracle_sb_corrref_eval.json

python3 -c "
import json
d = json.load(open('${EXP}/results/hh_oracle_sb_corrref_eval.json'))
print(f'[RESULT] HH Oracle SB Interp Ref: acc={d[\"accuracy\"]:.4f}  margin={d[\"mean_margin\"]:.4f}')
"
echo "[HH Oracle SB Interp] Done at $(date)"
