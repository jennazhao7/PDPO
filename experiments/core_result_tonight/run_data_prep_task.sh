#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

EPSILON="${EPSILON:-1.0}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-data/pku_saferlhf_secure}"
OUT_ROOT="${OUT_ROOT:-outputs_openllama_tonight}"

mkdir -p "$OUT_ROOT/preprocessing"

TRAIN_JSONL="${DATA_ROOT}/train_pref.jsonl"
TEST_JSONL="${DATA_ROOT}/test_pref.jsonl"

if [[ ! -f "$TRAIN_JSONL" || ! -f "$TEST_JSONL" ]]; then
  python experiments/core_result_tonight/01_secure_pku_saferlhf.py \
    --dataset_id PKU-Alignment/PKU-SafeRLHF \
    --dataset_config default \
    --output_dir "$DATA_ROOT" \
    --train_rows 10000 \
    --test_rows 1000
fi

D1_OUT="$OUT_ROOT/preprocessing/d1_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
D2_OUT="$OUT_ROOT/preprocessing/d2_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
D1_AUDIT="$OUT_ROOT/preprocessing/d1_rr_audit_eps${EPSILON}_seed${SEED}.jsonl"
D2_AUDIT="$OUT_ROOT/preprocessing/d2_rr_audit_eps${EPSILON}_seed${SEED}.jsonl"

if [[ ! -f "$D1_OUT" ]]; then
  python lora/preprocessing/rr_stream_flip.py \
    --input_jsonl "$TRAIN_JSONL" \
    --epsilon "$EPSILON" \
    --seed "$SEED" \
    --partition_count 2 \
    --partition_index 0 \
    --write_out "$D1_OUT" \
    --audit_out "$D1_AUDIT" \
    --max_audit 100000
fi

if [[ ! -f "$D2_OUT" ]]; then
  python lora/preprocessing/rr_stream_flip.py \
    --input_jsonl "$TRAIN_JSONL" \
    --epsilon "$EPSILON" \
    --seed "$SEED" \
    --partition_count 2 \
    --partition_index 1 \
    --write_out "$D2_OUT" \
    --audit_out "$D2_AUDIT" \
    --max_audit 100000
fi

echo "Data prep ready:"
echo "  $D1_OUT"
echo "  $D2_OUT"
