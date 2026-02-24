#!/usr/bin/env bash
set -euo pipefail

# Tonight core run (4 models) with RR -> Stage1 -> Stage2(MLE/SB)
# Models:
#   - EleutherAI/pythia-1b
#   - EleutherAI/pythia-2.8b
#   - openlm-research/open_llama_3b_v2
#   - openlm-research/open_llama_7b_v2

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

EPSILON="${EPSILON:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"

DATA_ROOT="${DATA_ROOT:-data/pku_saferlhf_secure}"
OUT_ROOT="${OUT_ROOT:-outputs_openllama_tonight}"

mkdir -p "$OUT_ROOT"/{logs,preprocessing,stage1,stage2_mle,stage2_softbayes}

log() {
  echo "[$(date +'%F %T')] $*"
}

run_if_missing() {
  local target="$1"
  shift
  if [[ -e "$target" ]]; then
    log "SKIP (exists): $target"
  else
    "$@"
  fi
}

############################
# 0) Secure dataset snapshot
############################
TRAIN_JSONL="${DATA_ROOT}/train_pref.jsonl"
TEST_JSONL="${DATA_ROOT}/test_pref.jsonl"

if [[ ! -f "$TRAIN_JSONL" || ! -f "$TEST_JSONL" ]]; then
  log "Preparing secure PKU-SafeRLHF snapshot..."
  python experiments/core_result_tonight/01_secure_pku_saferlhf.py \
    --dataset_id PKU-Alignment/PKU-SafeRLHF \
    --dataset_config default \
    --output_dir "$DATA_ROOT" \
    --train_rows 10000 \
    --test_rows 1000
else
  log "Secure dataset exists: $DATA_ROOT"
fi

##############################################
# 1) RR preprocessing split for stage1/stage2
##############################################
D1_OUT="$OUT_ROOT/preprocessing/d1_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
D2_OUT="$OUT_ROOT/preprocessing/d2_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
D1_AUDIT="$OUT_ROOT/preprocessing/d1_rr_audit_eps${EPSILON}_seed${SEED}.jsonl"
D2_AUDIT="$OUT_ROOT/preprocessing/d2_rr_audit_eps${EPSILON}_seed${SEED}.jsonl"

run_if_missing "$D1_OUT" \
  python lora/preprocessing/rr_stream_flip.py \
    --input_jsonl "$TRAIN_JSONL" \
    --epsilon "$EPSILON" \
    --seed "$SEED" \
    --partition_count 2 \
    --partition_index 0 \
    --write_out "$D1_OUT" \
    --audit_out "$D1_AUDIT" \
    --max_audit 100000

run_if_missing "$D2_OUT" \
  python lora/preprocessing/rr_stream_flip.py \
    --input_jsonl "$TRAIN_JSONL" \
    --epsilon "$EPSILON" \
    --seed "$SEED" \
    --partition_count 2 \
    --partition_index 1 \
    --write_out "$D2_OUT" \
    --audit_out "$D2_AUDIT" \
    --max_audit 100000

###########################################
# 2) Model matrix (RTX-6000-safe defaults)
###########################################
MODELS=(
  "EleutherAI/pythia-1b"
  "EleutherAI/pythia-2.8b"
  "openlm-research/open_llama_3b_v2"
  "openlm-research/open_llama_7b_v2"
)

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL="${MODEL//\//--}"
  log "===== START MODEL: $MODEL ====="

  # LoRA target modules
  if [[ "$MODEL" == EleutherAI/pythia-* ]]; then
    TARGET_MODS="query_key_value,dense_h_to_4h,dense_4h_to_h"
    GA_STAGE1=16
    GA_STAGE2=16
  else
    # OpenLLaMA family
    TARGET_MODS="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    GA_STAGE1=32
    GA_STAGE2=32
  fi

  STAGE1_OUT="$OUT_ROOT/stage1/${SAFE_MODEL}_stage1_rr_eps${EPSILON}_seed${SEED}"
  STAGE2_MLE_OUT="$OUT_ROOT/stage2_mle/${SAFE_MODEL}_stage2_mle_eps${EPSILON}_seed${SEED}"
  STAGE2_SB_OUT="$OUT_ROOT/stage2_softbayes/${SAFE_MODEL}_stage2_sb_eps${EPSILON}_seed${SEED}"

  # 2.1 Stage1 RR-DPO
  run_if_missing "$STAGE1_OUT/adapter_config.json" \
    python lora/preprocessing/train_truthy_stage1_lora.py \
      --model "$MODEL" \
      --data "$D1_OUT" \
      --out "$STAGE1_OUT" \
      --max-steps "$MAX_STEPS" \
      --epochs 3.0 \
      --bsz 1 \
      --ga "$GA_STAGE1" \
      --lr 1e-4 \
      --warmup-ratio 0.03 \
      --max-prompt 256 \
      --max-target 256 \
      --max-len 512 \
      --lora-r 16 \
      --lora-alpha 32 \
      --lora-dropout 0.05 \
      --target-modules "$TARGET_MODS" \
      --seed "$SEED"

  # 2.2 Stage2 MLE
  run_if_missing "$STAGE2_MLE_OUT/M2_manifest.json" \
    python lora/train_stage2_lora.py \
      --model "$MODEL" \
      --stage1_adapter "$STAGE1_OUT" \
      --data "$D2_OUT" \
      --out "$STAGE2_MLE_OUT" \
      --manifest_out "$STAGE2_MLE_OUT/M2_manifest.json" \
      --target_modules "$TARGET_MODS" \
      --max_len 512 \
      --bsz 1 \
      --ga "$GA_STAGE2" \
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

  # 2.3 Stage2 Soft-Bayes
  run_if_missing "$STAGE2_SB_OUT/M2_manifest.json" \
    python lora/train_stage2_soft_bayes.py \
      --model "$MODEL" \
      --stage1_adapter "$STAGE1_OUT" \
      --data "$D2_OUT" \
      --out "$STAGE2_SB_OUT" \
      --manifest_out "$STAGE2_SB_OUT/M2_manifest.json" \
      --target_modules "$TARGET_MODS" \
      --epsilon "$EPSILON" \
      --max_len 512 \
      --bsz 1 \
      --ga "$GA_STAGE2" \
      --lr 1e-5 \
      --max_steps "$MAX_STEPS" \
      --epochs 3 \
      --max_grad_norm 1.0 \
      --lora_r 16 \
      --lora_alpha 32 \
      --lora_dropout 0.05 \
      --seed "$SEED"

  log "===== DONE MODEL: $MODEL ====="
done

log "All 4 models complete."
