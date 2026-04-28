#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-384}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-512}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

PREP_DIR="${PREP_DIR:-${ROOT_DIR}/stage2_debugging/preprocessing}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage1/results_base_d1d2_eps${EPS}_seed${SEED}}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"

PKU_TEST="${PKU_TEST:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${EVAL_DIR}"

source ~/.bashrc
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[FATAL] conda not found in PATH"; exit 2
fi
cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false

D1_PKU="${PREP_DIR}/d1_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl"
D2_PKU="${PREP_DIR}/d2_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl"

[ -f "${D1_PKU}" ]   || { echo "[FATAL] Missing: ${D1_PKU}";   exit 2; }
[ -f "${D2_PKU}" ]   || { echo "[FATAL] Missing: ${D2_PKU}";   exit 2; }
[ -f "${PKU_TEST}" ] || { echo "[FATAL] Missing: ${PKU_TEST}"; exit 2; }

# Combine D1+D2 into /tmp on the compute node (avoids /users quota)
COMBINED_DATA="/tmp/d1d2_pku_eps${EPS}_seed${SEED}_$$.jsonl"
echo "[INFO] Combining D1+D2 -> ${COMBINED_DATA}"
cat "${D1_PKU}" "${D2_PKU}" > "${COMBINED_DATA}"
N_PAIRS=$(wc -l < "${COMBINED_DATA}")
echo "[INFO] Combined: ${N_PAIRS} pairs"

OUT_PKU="${OUT_ROOT}/pku_base_d1d2_eps${EPS}_s${SEED}"

python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), "gpus:", torch.cuda.device_count())
assert torch.cuda.is_available() and torch.cuda.device_count() >= 1
PY

nvidia-smi || true

echo "[INFO] Training: BASE=${BASE_MODEL}  DATA=${N_PAIRS}p  OUT=${OUT_PKU}"

CUDA_VISIBLE_DEVICES=0 python -u lora/preprocessing/train_truthy_stage1_lora.py \
  --model "${BASE_MODEL}" \
  --data "${COMBINED_DATA}" \
  --out "${OUT_PKU}" \
  --max-steps "${MAX_STEPS}" \
  --save-strategy "${SAVE_STRATEGY}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --epochs 3.0 --bsz 1 --ga 32 --lr 1e-4 --warmup-ratio 0.03 \
  --max-prompt 256 --max-target 256 --max-len "${TRAIN_MAX_LEN}" \
  --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
  --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --seed "${SEED}" \
  > "${LOG_DIR}/train_pku_base_d1d2_s${SEED}.log" 2>&1

rm -f "${COMBINED_DATA}"

[ -f "${OUT_PKU}/adapter_model.safetensors" ] || { echo "[FATAL] No adapter in ${OUT_PKU}"; exit 3; }

echo "[INFO] Training done. Evaluating..."

CUDA_VISIBLE_DEVICES=0 python -u eval/eval_stage1_accuracy.py \
  --base_model "${BASE_MODEL}" \
  --stage1_adapter "${OUT_PKU}" \
  --test_jsonl "${PKU_TEST}" \
  --out_json "${EVAL_DIR}/pku_base_d1d2_eps${EPS}_s${SEED}_eval.json" \
  --max_len "${EVAL_MAX_LEN}" \
  > "${LOG_DIR}/eval_pku_base_d1d2_s${SEED}.log" 2>&1

python - <<PY
import json, os
p = "${EVAL_DIR}/pku_base_d1d2_eps${EPS}_s${SEED}_eval.json"
if not os.path.exists(p):
    print("[WARN] Eval file missing:", p)
else:
    d = json.load(open(p))
    acc = d.get("accuracy", "N/A")
    print(f"=== PKU Base (D1+D2, ~{N_PAIRS} pairs) ===")
    print(f"  Accuracy : {acc}  (N={d.get('n','N/A')})")
    print(f"  Above 50%? {'YES ✅' if isinstance(acc, float) and acc > 0.5 else 'NO ❌'}")
PY

echo "[DONE] ${OUT_ROOT}"
