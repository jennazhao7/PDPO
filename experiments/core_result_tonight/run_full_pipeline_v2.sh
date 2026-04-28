#!/usr/bin/env bash
# run_full_pipeline_v2.sh
set -euo pipefail

MODEL=$1
SEED=$2
GPU=$3

# Use the new directory
export OUT_ROOT="outputs_new_models"
OUT_TXT="${OUT_ROOT}/logs/${MODEL//\//--}_seed${SEED}.txt"

export CUDA_VISIBLE_DEVICES=$GPU
export SEED=$SEED
export GPU_ID=$GPU

mkdir -p "${OUT_ROOT}/logs"

echo "==========================================================" > "$OUT_TXT"
echo "Starting pipeline for $MODEL (Seed $SEED) on GPU $GPU" >> "$OUT_TXT"
echo "Output Directory: $OUT_ROOT" >> "$OUT_TXT"
echo "==========================================================" >> "$OUT_TXT"

# set -e ensures that if ANY step fails, the script STOPS immediately.
# This guarantees partial results (like Stage 1) are perfectly preserved 
# and not overwritten or corrupted by later failing stages.

echo "" >> "$OUT_TXT"
echo "[$(date)] === STAGE 0: DATA PREP ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_data_prep_task.sh >> "$OUT_TXT" 2>&1

echo "" >> "$OUT_TXT"
echo "[$(date)] === STAGE 1: RR-SFT ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_stage1_task.sh "$MODEL" >> "$OUT_TXT" 2>&1

echo "" >> "$OUT_TXT"
echo "[$(date)] === STAGE 2: RR-MLE ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_stage2_mle_task.sh "$MODEL" >> "$OUT_TXT" 2>&1

echo "" >> "$OUT_TXT"
echo "[$(date)] === STAGE 2: RR-SoftBayes ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_stage2_sb_task.sh "$MODEL" >> "$OUT_TXT" 2>&1

echo "" >> "$OUT_TXT"
echo "[$(date)] === PIPELINE COMPLETION ===" >> "$OUT_TXT"
echo "All steps finished successfully!" >> "$OUT_TXT"
