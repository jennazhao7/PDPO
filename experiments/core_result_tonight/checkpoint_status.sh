#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

EPSILON="${EPSILON:-1.0}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-outputs_openllama_tonight}"

MODELS=(
  "EleutherAI/pythia-1b"
  "EleutherAI/pythia-2.8b"
  "openlm-research/open_llama_3b_v2"
  "openlm-research/open_llama_7b_v2"
)

echo "=== Overnight pipeline status (eps=${EPSILON}, seed=${SEED}) ==="
echo "OUT_ROOT=${OUT_ROOT}"
echo

for MODEL in "${MODELS[@]}"; do
  SAFE_MODEL="${MODEL//\//--}"
  S1="${OUT_ROOT}/stage1/${SAFE_MODEL}_stage1_rr_eps${EPSILON}_seed${SEED}/adapter_config.json"
  MLE="${OUT_ROOT}/stage2_mle/${SAFE_MODEL}_stage2_mle_eps${EPSILON}_seed${SEED}/M2_manifest.json"
  SB="${OUT_ROOT}/stage2_softbayes/${SAFE_MODEL}_stage2_sb_eps${EPSILON}_seed${SEED}/M2_manifest.json"

  [[ -f "$S1" ]] && S1_ST="DONE" || S1_ST="PENDING"
  [[ -f "$MLE" ]] && MLE_ST="DONE" || MLE_ST="PENDING"
  [[ -f "$SB" ]] && SB_ST="DONE" || SB_ST="PENDING"

  printf "%-42s  stage1=%-7s  stage2_mle=%-7s  stage2_sb=%-7s\n" "$MODEL" "$S1_ST" "$MLE_ST" "$SB_ST"
done

echo
echo "Resume priority:"
echo "  1) openlm-research/open_llama_3b_v2 stage2_sb"
echo "  2) openlm-research/open_llama_7b_v2 stage1"
echo "  3) openlm-research/open_llama_7b_v2 stage2_mle"
echo "  4) openlm-research/open_llama_7b_v2 stage2_sb"
