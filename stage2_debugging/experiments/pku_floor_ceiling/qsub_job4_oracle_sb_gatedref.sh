#!/bin/bash
# Job 4 — Oracle SB Gated Ref (submitted via qsub or run directly)
#$ -S /bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -l h_rt=6:00:00
#$ -pe smp 4
#$ -M jzhao7@nd.edu
#$ -m bea

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
echo "[Job 4] Training Oracle SB Gated Ref (tau=1.0)..."

python ${EXP}/train_job4_oracle_sb_gatedref.py \
  --model   Qwen/Qwen2.5-3B-Instruct \
  --m1      ${M1} \
  --data    ${EXP}/d2_pku_with_flipped_eps1.0_seed42.jsonl \
  --out     ${EXP}/models/pku_oracle_sb_gatedref \
  --resume_from ${EXP}/models/pku_oracle_sb_gatedref/checkpoint-step-300 \
  --epsilon 1.0 --beta 0.5 --tau 1.0 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4 --bf16

echo "[Job 4] Training done. Running eval..."

python stage2_debugging/eval_preference_accuracy.py \
  --manifest   ${EXP}/models/pku_oracle_sb_gatedref/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/pku_secure/test_pref.jsonl \
  --out_json   ${EXP}/results/job4.json

python3 -c "
import json
d = json.load(open('${EXP}/results/job4.json'))
acc = d['accuracy']
j2  = json.load(open('${EXP}/results/job2.json'))['accuracy']
j3  = json.load(open('${EXP}/results/job3.json'))['accuracy']
print(f'[RESULT] Job 4 (Oracle SB gated ref): {acc:.4f}')
print(f'[COMPARE] Job2 ceiling={j2:.4f}  Job3 (old method)={j3:.4f}  Job4 (gated)={acc:.4f}')
if acc > j2:
    print('[VERDICT] ✅ Gated ref BEATS base-ref ceiling — corrected reference story holds.')
elif acc > j3:
    print('[VERDICT] ⚠️  Gated ref improves over Job 3 but does not beat ceiling.')
    print('          M1 gamma=0.56 may be at the quality threshold for reference correction.')
else:
    print('[VERDICT] ❌ Gated ref does not help — M1 too weak for any reference interpolation.')
"
echo "[Job 4] Done at $(date)"
