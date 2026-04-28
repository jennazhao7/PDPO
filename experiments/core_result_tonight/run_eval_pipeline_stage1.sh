#!/usr/bin/env bash
# run_eval_pipeline_stage1.sh
set -euo pipefail

MODEL=$1
SEED=$2
GPU=$3

export OUT_ROOT="outputs_new_models"
STAGE1_DIR="${OUT_ROOT}/stage1/${MODEL//\//--}_stage1_rr_eps1.0_seed${SEED}"
TEST_JSONL="data/pku_saferlhf_secure/test_pref.jsonl"
OUT_JSON="${STAGE1_DIR}/eval_pref_accuracy.json"
LOG_TXT="${OUT_ROOT}/logs/eval_stage1_${MODEL//\//--}_seed${SEED}.txt"

export CUDA_VISIBLE_DEVICES=$GPU

echo "Starting Stage 1 evaluation for $MODEL (Seed $SEED) on GPU $GPU" > "$LOG_TXT"

if [[ ! -d "$STAGE1_DIR" ]]; then
  echo "Error: Stage 1 dir not found at $STAGE1_DIR" >> "$LOG_TXT"
  exit 1
fi

# Build a temporary manifest pointing to Stage 1 - eval_preference_accuracy.py expects a manifest
# that points to stage1 as M1 only (no M2 stage2 adapter)
TMP_MANIFEST="${STAGE1_DIR}/M1_manifest.json"
cat > "$TMP_MANIFEST" << EOF
{
  "base_model": "$MODEL",
  "stage1_adapter": "$STAGE1_DIR",
  "stage2_adapter": null,
  "method": "stage1_only"
}
EOF

echo "[$(date)] Running Preference Accuracy Evaluation (Stage 1 baseline)" >> "$LOG_TXT"
conda run -n pdpo python eval/eval_preference_accuracy.py \
  --manifest "$TMP_MANIFEST" \
  --test_jsonl "$TEST_JSONL" \
  --out_json "$OUT_JSON" >> "$LOG_TXT" 2>&1

echo "[$(date)] Stage 1 evaluation finished!" >> "$LOG_TXT"
