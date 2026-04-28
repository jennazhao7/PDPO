#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

D1_DATA="${D1_DATA:-outputs_eps05/preprocessing/d1_rr_flipped_eps05.jsonl}"
STEPS="${STEPS:-300}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-outputs_eps05/stage1_rr}"

mkdir -p "$OUT_ROOT"

COMMON_ARGS=(
  --data "$D1_DATA"
  --max-steps "$STEPS"
  --seed "$SEED"
  --epochs 3.0
  --bsz 1
  --ga 16
  --lr 1e-4
  --warmup-ratio 0.03
  --max-prompt 256
  --max-target 256
  --max-len 512
  --lora-r 16
  --lora-alpha 32
  --lora-dropout 0.05
)

echo "[eps05] Stage1 RR gpt2-medium"
if [[ -f "$OUT_ROOT/gpt2-medium-truthy-stage1-rr-eps05/adapter_config.json" ]]; then
  echo "[eps05] SKIP gpt2-medium stage1 (adapter already exists)"
else
  python lora/preprocessing/train_truthy_stage1_lora.py \
    --model gpt2-medium \
    --target-modules c_attn,c_proj,c_fc \
    --out "$OUT_ROOT/gpt2-medium-truthy-stage1-rr-eps05" \
    "${COMMON_ARGS[@]}"
fi

echo "[eps05] Stage1 RR gpt2-large"
if [[ -f "$OUT_ROOT/gpt2-large-truthy-stage1-rr-eps05/adapter_config.json" ]]; then
  echo "[eps05] SKIP gpt2-large stage1 (adapter already exists)"
else
  python lora/preprocessing/train_truthy_stage1_lora.py \
    --model gpt2-large \
    --target-modules c_attn,c_proj,c_fc \
    --out "$OUT_ROOT/gpt2-large-truthy-stage1-rr-eps05" \
    "${COMMON_ARGS[@]}"
fi

echo "[eps05] Stage1 RR EleutherAI/pythia-1b"
if [[ -f "$OUT_ROOT/pythia-1b-truthy-stage1-rr-eps05/adapter_config.json" ]]; then
  echo "[eps05] SKIP pythia-1b stage1 (adapter already exists)"
else
  python lora/preprocessing/train_truthy_stage1_lora.py \
    --model EleutherAI/pythia-1b \
    --target-modules query_key_value,dense_h_to_4h,dense_4h_to_h \
    --out "$OUT_ROOT/pythia-1b-truthy-stage1-rr-eps05" \
    "${COMMON_ARGS[@]}"
fi

echo "[eps05] Stage1 RR done for all models."
