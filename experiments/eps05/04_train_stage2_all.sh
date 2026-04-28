#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

D2_DATA="${D2_DATA:-outputs_eps05/preprocessing/d2_rr_flipped_eps05.jsonl}"
STEPS="${STEPS:-300}"
SEED="${SEED:-42}"
EPSILON="${EPSILON:-0.5}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs_eps05/stage1_rr}"

COMMON_MLE_ARGS=(
  --data "$D2_DATA"
  --max_len 512
  --bsz 1
  --ga 16
  --lr 1e-5
  --max_steps "$STEPS"
  --epochs 1.0
  --max_grad_norm 1.0
  --warmup_ratio 0.03
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --seed "$SEED"
  --no_smoke_test
)

COMMON_SB_ARGS=(
  --data "$D2_DATA"
  --epsilon "$EPSILON"
  --max_len 512
  --bsz 1
  --ga 16
  --lr 1e-5
  --max_steps "$STEPS"
  --epochs 3
  --max_grad_norm 1.0
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --seed "$SEED"
)

echo "[eps05] Stage2 MLE gpt2-medium"
if [[ -f "outputs_eps05/gpt2-medium-stage2-mle/M2_manifest.json" ]]; then
  echo "[eps05] SKIP gpt2-medium stage2-mle (manifest exists)"
else
  python lora/train_stage2_lora.py \
    --model gpt2-medium \
    --stage1_adapter "$STAGE1_ROOT/gpt2-medium-truthy-stage1-rr-eps05" \
    --target_modules c_attn,c_proj,c_fc \
    --out outputs_eps05/gpt2-medium-stage2-mle \
    --manifest_out outputs_eps05/gpt2-medium-stage2-mle/M2_manifest.json \
    "${COMMON_MLE_ARGS[@]}"
fi

echo "[eps05] Stage2 Soft-Bayes gpt2-medium"
if [[ -f "outputs_eps05/gpt2-medium-stage2-softbayes/M2_manifest.json" ]]; then
  echo "[eps05] SKIP gpt2-medium stage2-softbayes (manifest exists)"
else
  python lora/train_stage2_soft_bayes.py \
    --model gpt2-medium \
    --stage1_adapter "$STAGE1_ROOT/gpt2-medium-truthy-stage1-rr-eps05" \
    --target_modules c_attn,c_proj,c_fc \
    --out outputs_eps05/gpt2-medium-stage2-softbayes \
    --manifest_out outputs_eps05/gpt2-medium-stage2-softbayes/M2_manifest.json \
    "${COMMON_SB_ARGS[@]}"
fi

echo "[eps05] Stage2 MLE gpt2-large"
if [[ -f "outputs_eps05/gpt2-large-stage2-mle/M2_manifest.json" ]]; then
  echo "[eps05] SKIP gpt2-large stage2-mle (manifest exists)"
else
  python lora/train_stage2_lora.py \
    --model gpt2-large \
    --stage1_adapter "$STAGE1_ROOT/gpt2-large-truthy-stage1-rr-eps05" \
    --target_modules c_attn,c_proj,c_fc \
    --out outputs_eps05/gpt2-large-stage2-mle \
    --manifest_out outputs_eps05/gpt2-large-stage2-mle/M2_manifest.json \
    "${COMMON_MLE_ARGS[@]}"
fi

echo "[eps05] Stage2 Soft-Bayes gpt2-large"
if [[ -f "outputs_eps05/gpt2-large-stage2-softbayes/M2_manifest.json" ]]; then
  echo "[eps05] SKIP gpt2-large stage2-softbayes (manifest exists)"
else
  python lora/train_stage2_soft_bayes.py \
    --model gpt2-large \
    --stage1_adapter "$STAGE1_ROOT/gpt2-large-truthy-stage1-rr-eps05" \
    --target_modules c_attn,c_proj,c_fc \
    --out outputs_eps05/gpt2-large-stage2-softbayes \
    --manifest_out outputs_eps05/gpt2-large-stage2-softbayes/M2_manifest.json \
    "${COMMON_SB_ARGS[@]}"
fi

echo "[eps05] Stage2 MLE pythia-1b"
if [[ -f "outputs_eps05/pythia1b-stage2-mle/M2_manifest.json" ]]; then
  echo "[eps05] SKIP pythia1b stage2-mle (manifest exists)"
else
  python lora/train_stage2_lora.py \
    --model EleutherAI/pythia-1b \
    --stage1_adapter "$STAGE1_ROOT/pythia-1b-truthy-stage1-rr-eps05" \
    --target_modules query_key_value,dense_h_to_4h,dense_4h_to_h \
    --out outputs_eps05/pythia1b-stage2-mle \
    --manifest_out outputs_eps05/pythia1b-stage2-mle/M2_manifest.json \
    "${COMMON_MLE_ARGS[@]}"
fi

echo "[eps05] Stage2 Soft-Bayes pythia-1b"
if [[ -f "outputs_eps05/pythia1b-stage2-softbayes/M2_manifest.json" ]]; then
  echo "[eps05] SKIP pythia1b stage2-softbayes (manifest exists)"
else
  python lora/train_stage2_soft_bayes.py \
    --model EleutherAI/pythia-1b \
    --stage1_adapter "$STAGE1_ROOT/pythia-1b-truthy-stage1-rr-eps05" \
    --target_modules query_key_value,dense_h_to_4h,dense_4h_to_h \
    --out outputs_eps05/pythia1b-stage2-softbayes \
    --manifest_out outputs_eps05/pythia1b-stage2-softbayes/M2_manifest.json \
    "${COMMON_SB_ARGS[@]}"
fi

echo "[eps05] Stage2 MLE + Soft-Bayes done for all models."
