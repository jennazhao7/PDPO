#!/usr/bin/env bash
# run_eval_pipeline.sh
set -euo pipefail

MODEL=$1
SEED=$2
GPU=$3

# Paths
export OUT_ROOT="outputs_new_models"
STAGE2_DIR="${OUT_ROOT}/stage2_softbayes/${MODEL//\//--}_stage2_sb_eps1.0_seed${SEED}"
MANIFEST="${STAGE2_DIR}/M2_manifest.json"
TEST_JSONL="data/pku_saferlhf_secure/test_pref.jsonl"
OUT_JSON="${STAGE2_DIR}/eval_pref_accuracy.json"
LOG_TXT="${OUT_ROOT}/logs/eval_${MODEL//\//--}_seed${SEED}.txt"

export CUDA_VISIBLE_DEVICES=$GPU

echo "Starting evaluation for $MODEL (Seed $SEED) on GPU $GPU" > "$LOG_TXT"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Error: Manifest not found at $MANIFEST" >> "$LOG_TXT"
  exit 1
fi

echo "[$(date)] Running Preference Accuracy Evaluation" >> "$LOG_TXT"
# User noted we don't need conda run here because they launch it from within the active env
python eval/eval_preference_accuracy.py \
  --manifest "$MANIFEST" \
  --test_jsonl "$TEST_JSONL" \
  --out_json "$OUT_JSON" >> "$LOG_TXT" 2>&1

echo "[$(date)] Evaluation finished successfully!" >> "$LOG_TXT"
