#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-300}"
TRAIN_MAX_LEN="${TRAIN_MAX_LEN:-384}"
EVAL_MAX_LEN="${EVAL_MAX_LEN:-512}"
SAVE_STRATEGY="${SAVE_STRATEGY:-steps}"
SAVE_STEPS="${SAVE_STEPS:-50}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

PREP_DIR="${PREP_DIR:-${ROOT_DIR}/stage2_debugging/preprocessing}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage1/results_instruct_eps${EPS}_seed${SEED}}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"

PKU_TEST="${PKU_TEST:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${EVAL_DIR}"

source ~/.bashrc
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
else
  echo "[FATAL] conda not found in PATH"
  exit 2
fi
cd "${ROOT_DIR}"
export TOKENIZERS_PARALLELISM=false

D1_PKU="${PREP_DIR}/d1_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl"

[ -f "${D1_PKU}" ]  || { echo "[FATAL] Missing training data: ${D1_PKU}";  exit 2; }
[ -f "${PKU_TEST}" ] || { echo "[FATAL] Missing test file:     ${PKU_TEST}"; exit 2; }

OUT_PKU="${OUT_ROOT}/pku_eps${EPS}_s${SEED}"

python - <<'PY'
import torch, peft, trl
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this job"
assert torch.cuda.device_count() >= 1, "Need at least 1 visible GPU"
PY

nvidia-smi || true

echo "[INFO] Starting Stage 1 training on PKU with instruct base (single GPU)..."
echo "  BASE_MODEL  = ${BASE_MODEL}"
echo "  D1_PKU      = ${D1_PKU}"
echo "  OUT_PKU     = ${OUT_PKU}"

CUDA_VISIBLE_DEVICES=0 python -u lora/preprocessing/train_truthy_stage1_lora.py \
  --model "${BASE_MODEL}" \
  --data "${D1_PKU}" \
  --out "${OUT_PKU}" \
  --max-steps "${MAX_STEPS}" \
  --save-strategy "${SAVE_STRATEGY}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --epochs 3.0 \
  --bsz 1 \
  --ga 32 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len "${TRAIN_MAX_LEN}" \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --seed "${SEED}" \
  > "${LOG_DIR}/train_pku_instruct_s${SEED}.log" 2>&1

[ -f "${OUT_PKU}/adapter_model.safetensors" ] || { echo "[FATAL] Missing adapter weights in ${OUT_PKU}"; exit 3; }

echo "[INFO] Training complete. Starting evaluation..."

CUDA_VISIBLE_DEVICES=0 python -u eval/eval_stage1_accuracy.py \
  --base_model "${BASE_MODEL}" \
  --stage1_adapter "${OUT_PKU}" \
  --test_jsonl "${PKU_TEST}" \
  --out_json "${EVAL_DIR}/pku_instruct_eps${EPS}_s${SEED}_eval.json" \
  --max_len "${EVAL_MAX_LEN}" \
  > "${LOG_DIR}/eval_pku_instruct_s${SEED}.log" 2>&1

echo "[INFO] Eval JSON written to ${EVAL_DIR}/pku_instruct_eps${EPS}_s${SEED}_eval.json"

python - <<PY
import json, os
p = "${EVAL_DIR}/pku_instruct_eps${EPS}_s${SEED}_eval.json"
if not os.path.exists(p):
    print(f"[WARN] Eval file not found: {p}")
else:
    d = json.load(open(p))
    acc = d.get("accuracy", "N/A")
    n   = d.get("n", "N/A")
    print(f"=== PKU Instruct M1 Result ===")
    print(f"  Base model   : ${BASE_MODEL}")
    print(f"  Accuracy     : {acc}")
    print(f"  N            : {n}")
    print(f"  Above 50%?   : {'YES ✅' if isinstance(acc, float) and acc > 0.5 else 'NO ❌'}")
PY

echo "[DONE] Stage1 instruct train+eval complete: ${OUT_ROOT}"
