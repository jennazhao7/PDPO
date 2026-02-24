#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${1:?Usage: run_stage2_mle_task.sh <model_id>}"
GPU_ID="${GPU_ID:-0}"
EPSILON="${EPSILON:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"
OUT_ROOT="${OUT_ROOT:-outputs_openllama_tonight}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_ID}"

SAFE_MODEL="${MODEL//\//--}"
D2_OUT="$OUT_ROOT/preprocessing/d2_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
STAGE1_DIR="$OUT_ROOT/stage1/${SAFE_MODEL}_stage1_rr_eps${EPSILON}_seed${SEED}"
OUT_DIR="$OUT_ROOT/stage2_mle/${SAFE_MODEL}_stage2_mle_eps${EPSILON}_seed${SEED}"
MANIFEST="$OUT_DIR/M2_manifest.json"

if [[ ! -f "$D2_OUT" ]]; then
  echo "Missing $D2_OUT. Run data prep first."
  exit 1
fi
if [[ ! -f "$STAGE1_DIR/adapter_config.json" ]]; then
  echo "Missing stage1 adapter: $STAGE1_DIR"
  exit 1
fi
if [[ -f "$MANIFEST" ]]; then
  echo "SKIP stage2_mle: already done -> $MANIFEST"
  exit 0
fi

if [[ "$MODEL" == EleutherAI/pythia-* ]]; then
  TARGET_MODS="query_key_value,dense_h_to_4h,dense_4h_to_h"
  GA=16
  MAX_LEN=512
else
  TARGET_MODS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
  GA=32
  MAX_LEN="${MAX_LEN:-384}"
fi

python lora/train_stage2_lora.py \
  --model "$MODEL" \
  --stage1_adapter "$STAGE1_DIR" \
  --data "$D2_OUT" \
  --out "$OUT_DIR" \
  --manifest_out "$MANIFEST" \
  --target_modules "$TARGET_MODS" \
  --max_len "$MAX_LEN" \
  --bsz 1 \
  --ga "$GA" \
  --lr 1e-5 \
  --max_steps "$MAX_STEPS" \
  --epochs 1.0 \
  --max_grad_norm 1.0 \
  --warmup_ratio 0.03 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --seed "$SEED" \
  --no_smoke_test
