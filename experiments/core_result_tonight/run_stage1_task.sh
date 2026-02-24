#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${1:?Usage: run_stage1_task.sh <model_id>}"
GPU_ID="${GPU_ID:-0}"
EPSILON="${EPSILON:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"
OUT_ROOT="${OUT_ROOT:-outputs_openllama_tonight}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

SAFE_MODEL="${MODEL//\//--}"
D1_OUT="$OUT_ROOT/preprocessing/d1_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
OUT_DIR="$OUT_ROOT/stage1/${SAFE_MODEL}_stage1_rr_eps${EPSILON}_seed${SEED}"

if [[ ! -f "$D1_OUT" ]]; then
  echo "Missing $D1_OUT. Run data prep first."
  exit 1
fi

if [[ -f "$OUT_DIR/adapter_config.json" ]]; then
  echo "SKIP stage1: already done -> $OUT_DIR"
  exit 0
fi

if [[ "$MODEL" == EleutherAI/pythia-* ]]; then
  TARGET_MODS="query_key_value,dense_h_to_4h,dense_4h_to_h"
  GA=16
  MAX_LEN=512
else
  TARGET_MODS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  GA=32
  # RTX6000-safe default for OpenLLaMA
  MAX_LEN="${MAX_LEN:-384}"
fi

python lora/preprocessing/train_truthy_stage1_lora.py \
  --model "$MODEL" \
  --data "$D1_OUT" \
  --out "$OUT_DIR" \
  --max-steps "$MAX_STEPS" \
  --epochs 3.0 \
  --bsz 1 \
  --ga "$GA" \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len "$MAX_LEN" \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules "$TARGET_MODS" \
  --seed "$SEED"
