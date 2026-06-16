#!/usr/bin/env bash
# Prepare data for quick_sb_test if missing (run on VM after bootstrap).
set -euo pipefail

PDPO_ROOT="${PDPO_ROOT:-${HOME}/PDPO}"
cd "${PDPO_ROOT}"

EPSILON="${EPSILON:-1.0}"
SEED="${SEED:-42}"
OUT_ROOT="${OUT_ROOT:-outputs_new_models}"

mkdir -p "${OUT_ROOT}/preprocessing" "${OUT_ROOT}/stage1"

# PKU test/train data
if [[ ! -f data/pku_saferlhf_secure/test_pref.jsonl ]]; then
  echo "Generating PKU-SafeRLHF secure split..."
  python experiments/core_result_tonight/01_secure_pku_saferlhf.py \
    --dataset_id PKU-Alignment/PKU-SafeRLHF \
    --dataset_config default \
    --output_dir data/pku_saferlhf_secure \
    --train_rows 10000 \
    --test_rows 1000
fi

D2_OUT="${OUT_ROOT}/preprocessing/d2_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
if [[ ! -f "${D2_OUT}" ]]; then
  echo "Running RR flip for D2..."
  bash experiments/core_result_tonight/run_data_prep_task.sh
  mkdir -p "${OUT_ROOT}/preprocessing"
  cp "outputs_openllama_tonight/preprocessing/d2_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl" "${D2_OUT}" 2>/dev/null || \
  python lora/preprocessing/rr_stream_flip.py \
    --input_jsonl data/pku_saferlhf_secure/train_pref.jsonl \
    --epsilon "${EPSILON}" --seed "${SEED}" \
    --partition_count 2 --partition_index 1 \
    --write_out "${D2_OUT}" \
    --audit_out "${OUT_ROOT}/preprocessing/d2_rr_audit_eps${EPSILON}_seed${SEED}.jsonl" \
    --max_audit 100000
fi

S1_DIR="${OUT_ROOT}/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps${EPSILON}_seed${SEED}"
if [[ ! -f "${S1_DIR}/adapter_config.json" ]]; then
  echo "Training minimal Stage1 adapter for smoke test (50 steps)..."
  D1_OUT="${OUT_ROOT}/preprocessing/d1_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
  if [[ ! -f "${D1_OUT}" ]]; then
    python lora/preprocessing/rr_stream_flip.py \
      --input_jsonl data/pku_saferlhf_secure/train_pref.jsonl \
      --epsilon "${EPSILON}" --seed "${SEED}" \
      --partition_count 2 --partition_index 0 \
      --write_out "${D1_OUT}" \
      --audit_out "${OUT_ROOT}/preprocessing/d1_rr_audit_eps${EPSILON}_seed${SEED}.jsonl" \
      --max_audit 100000
  fi
  OUT_ROOT="${OUT_ROOT}" GPU_ID=0 MAX_STEPS=50 bash -c '
    export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
    python lora/preprocessing/train_truthy_stage1_lora.py \
      --model Qwen/Qwen2.5-3B \
      --data "'"${D1_OUT}"'" \
      --out "'"${S1_DIR}"'" \
      --max-steps 50 --epochs 1.0 --bsz 1 --ga 32 --lr 1e-4 \
      --max-prompt 256 --max-target 256 --max-len 256 \
      --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
      --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
      --seed '"${SEED}"'
  '
fi

echo "Smoke test prerequisites ready."
