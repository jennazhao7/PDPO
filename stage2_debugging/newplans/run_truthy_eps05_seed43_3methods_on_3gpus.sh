#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-43}"
EPS="${EPS:-0.5}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps1.0_seed43_truthy/truthy_eps1.0_s43}"
D2_JSONL="${D2_JSONL:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed43.jsonl}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps0.5_seed43}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${EVAL_DIR}"

source ~/.bashrc
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[FATAL] conda not found in PATH"
  exit 2
fi

cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/users/jzhao7/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
assert torch.cuda.device_count() >= 3, "Need at least 3 visible GPUs"
PY

nvidia-smi || true

for f in \
  "${S1_ADAPTER}/adapter_model.safetensors" \
  "${D2_JSONL}" \
  "${TEST_JSONL}"
do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

run_one() {
  local method="$1" gpu="$2"
  local out_dir manifest train_log eval_log eval_json

  out_dir="${OUT_ROOT}/${method}_truthy_eps${EPS}_s${SEED}"
  manifest="${out_dir}/M2_manifest.json"
  train_log="${LOG_DIR}/train_${method}_truthy_s${SEED}.log"
  eval_log="${LOG_DIR}/eval_${method}_truthy_s${SEED}.log"
  eval_json="${EVAL_DIR}/${method}_truthy_eps${EPS}_s${SEED}_eval.json"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${manifest}" ]]; then
    echo "[SKIP] ${method}: ${manifest}" | tee -a "${train_log}"
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[RUN] gpu=${gpu} method=${method} out=${out_dir}" | tee -a "${train_log}"

  if [[ "${method}" == "sb_fresh" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_sb_fresh.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${S1_ADAPTER}" \
      --data "${D2_JSONL}" \
      --out "${out_dir}" \
      --epsilon "${EPS}" \
      --beta 0.5 \
      --lr 2.5e-5 \
      --seed "${SEED}" \
      --epochs 3 \
      --max_steps -1 \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT}" \
      --bsz 1 \
      --ga 16 \
      --bf16 \
      >> "${train_log}" 2>&1
  elif [[ "${method}" == "map_retrain" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_map_retrain.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${S1_ADAPTER}" \
      --data "${D2_JSONL}" \
      --out "${out_dir}" \
      --epsilon "${EPS}" \
      --beta 0.5 \
      --lr 2.5e-5 \
      --seed "${SEED}" \
      --epochs 3 \
      --max_steps -1 \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT}" \
      --bsz 1 \
      --ga 16 \
      --bf16 \
      >> "${train_log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_mle_fresh.py \
      --model "${BASE_MODEL}" \
      --data "${D2_JSONL}" \
      --out "${out_dir}" \
      --epsilon "${EPS}" \
      --beta 0.5 \
      --lr 2.5e-5 \
      --seed "${SEED}" \
      --epochs 3 \
      --max_steps -1 \
      --save_steps "${SAVE_STEPS}" \
      --save_total_limit "${SAVE_TOTAL_LIMIT}" \
      --bsz 1 \
      --ga 16 \
      --bf16 \
      >> "${train_log}" 2>&1
  fi

  [ -f "${manifest}" ] || { echo "[FATAL] missing manifest: ${manifest}"; return 3; }

  CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/eval_preference_accuracy.py \
    --manifest "${manifest}" \
    --test_jsonl "${TEST_JSONL}" \
    --out_json "${eval_json}" \
    --max_len 512 \
    >> "${eval_log}" 2>&1

  echo "[DONE] ${method}"
}

( run_one sb_fresh 0 ) & P1=$!
( run_one map_retrain 1 ) & P2=$!
( run_one mle_fresh 2 ) & P3=$!

FAILED=0
for p in "${P1}" "${P2}" "${P3}"; do
  if ! wait "${p}"; then
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "[FATAL] one or more methods failed"
  exit 1
fi

echo "[DONE] Truthy eps0.5 seed43 3-method job complete: ${OUT_ROOT}"
