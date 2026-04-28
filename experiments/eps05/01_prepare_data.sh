#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

INPUT_DATA="${INPUT_DATA:-preprocessing/truthydpo/truthy_dpo_subset.jsonl}"
EPSILON="${EPSILON:-0.5}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-outputs_eps05/preprocessing}"

mkdir -p "$OUT_DIR"

D1_OUT="$OUT_DIR/d1_rr_flipped_eps05.jsonl"
D2_OUT="$OUT_DIR/d2_rr_flipped_eps05.jsonl"
D1_AUDIT="$OUT_DIR/d1_rr_audit_eps05.jsonl"
D2_AUDIT="$OUT_DIR/d2_rr_audit_eps05.jsonl"

echo "[eps05] Building D1 RR data -> $D1_OUT"
python lora/preprocessing/rr_stream_flip.py \
  --input_jsonl "$INPUT_DATA" \
  --epsilon "$EPSILON" \
  --seed "$SEED" \
  --partition_count 2 \
  --partition_index 0 \
  --write_out "$D1_OUT" \
  --audit_out "$D1_AUDIT" \
  --max_audit 100000

echo "[eps05] Building D2 RR data -> $D2_OUT"
python lora/preprocessing/rr_stream_flip.py \
  --input_jsonl "$INPUT_DATA" \
  --epsilon "$EPSILON" \
  --seed "$SEED" \
  --partition_count 2 \
  --partition_index 1 \
  --write_out "$D2_OUT" \
  --audit_out "$D2_AUDIT" \
  --max_audit 100000

echo "[eps05] Done. Outputs in $OUT_DIR"
