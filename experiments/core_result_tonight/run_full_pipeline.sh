#!/usr/bin/env bash
set -euo pipefail

MODEL=$1
SEED=$2
GPU=$3
OUT_TXT="outputs_openllama_tonight/logs/${MODEL//\//--}_seed${SEED}.txt"

export CUDA_VISIBLE_DEVICES=$GPU
export SEED=$SEED
export GPU_ID=$GPU

mkdir -p outputs_openllama_tonight/logs

echo "Starting pipeline for $MODEL (Seed $SEED) on GPU $GPU" > "$OUT_TXT"

echo "=== DATA PREP ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_data_prep.sh >> "$OUT_TXT" 2>&1 || true

echo "=== STAGE 1 (RR-SFT) ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_stage1_task.sh "$MODEL" >> "$OUT_TXT" 2>&1

echo "=== STAGE 2 (RR-MLE) ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_stage2_mle_task.sh "$MODEL" >> "$OUT_TXT" 2>&1

echo "=== STAGE 2 (RR-SoftBayes) ===" >> "$OUT_TXT"
bash experiments/core_result_tonight/run_stage2_sb_task.sh "$MODEL" >> "$OUT_TXT" 2>&1

echo "Pipeline finished!" >> "$OUT_TXT"
