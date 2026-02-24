#!/bin/bash
# Run rr_stream_flip to produce D1 and D2 (partitioned, RR-flipped) for Stage-1 and Stage-2.
# Usage: bash lora/preprocessing/prepare_d1_d2.sh [--input truthy_dpo_subset.jsonl] [--epsilon 1] [--seed 0]

set -e
cd "$(dirname "$0")/../.."
OUT_DIR="lora/preprocessing"
INPUT="${INPUT:-preprocessing/truthydpo/truthy_dpo_subset.jsonl}"
EPS="${EPS:-1.0}"
SEED="${SEED:-0}"

D1="${OUT_DIR}/d1_rr_flipped.jsonl"
D2="${OUT_DIR}/d2_rr_flipped.jsonl"

echo "Preparing D1 and D2 from $INPUT (epsilon=$EPS seed=$SEED)"
echo ""

echo "Writing D1 (partition 0) to $D1 ..."
python lora/preprocessing/rr_stream_flip.py \
  --input_jsonl "$INPUT" \
  --epsilon "$EPS" \
  --seed "$SEED" \
  --partition_count 2 \
  --partition_index 0 \
  --write_out "$D1" \
  --max_audit 100000

echo ""
echo "Writing D2 (partition 1) to $D2 ..."
python lora/preprocessing/rr_stream_flip.py \
  --input_jsonl "$INPUT" \
  --epsilon "$EPS" \
  --seed "$SEED" \
  --partition_count 2 \
  --partition_index 1 \
  --write_out "$D2" \
  --max_audit 100000

echo ""
echo "Done. Stage-1 loads from $D1, Stage-2 from $D2"
