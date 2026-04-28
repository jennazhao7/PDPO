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
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"

DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/test_pref.jsonl"
TRUTHY_TEST="${TRUTHY_TEST:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"
HHRLHF_TEST="${HHRLHF_TEST:-${ROOT_DIR}/stage2_debugging/test_pref.jsonl}"
PKU_TEST="${PKU_TEST:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

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

PIDS=()
cleanup_children() {
  for p in "${PIDS[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
}
trap cleanup_children EXIT INT TERM

D1_TRUTHY="${PREP_DIR}/d1_rr_flipped_truthy_eps${EPS}_seed${SEED}.jsonl"
D1_HHRLHF="${PREP_DIR}/d1_rr_flipped_hhrlhf_eps${EPS}_seed${SEED}.jsonl"
D1_PKU="${PREP_DIR}/d1_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl"

resolve_test_path() {
  local requested="$1"
  local fallback="$2"
  local label="$3"
  if [[ -f "${requested}" ]]; then
    echo "${requested}"
    return
  fi
  if [[ -f "${fallback}" ]]; then
    echo "[WARN] ${label} test missing at ${requested}; falling back to ${fallback}" >&2
    echo "${fallback}"
    return
  fi
  echo "[FATAL] ${label} test missing. Tried: ${requested} and fallback: ${fallback}" >&2
  exit 2
}

TRUTHY_TEST="$(resolve_test_path "${TRUTHY_TEST}" "${DEFAULT_TEST}" "TruthyDPO")"
HHRLHF_TEST="$(resolve_test_path "${HHRLHF_TEST}" "${DEFAULT_TEST}" "HH-RLHF")"
PKU_TEST="$(resolve_test_path "${PKU_TEST}" "${DEFAULT_TEST}" "PKU-SafeRLHF")"

OUT_TRUTHY="${OUT_ROOT}/truthy_eps${EPS}_s${SEED}"
OUT_HHRLHF="${OUT_ROOT}/hhrlhf_eps${EPS}_s${SEED}"
OUT_PKU="${OUT_ROOT}/pku_eps${EPS}_s${SEED}"

for f in "$D1_TRUTHY" "$D1_HHRLHF" "$D1_PKU" "$TRUTHY_TEST" "$HHRLHF_TEST" "$PKU_TEST"; do
  [ -f "$f" ] || { echo "[FATAL] Missing file: $f"; exit 2; }
done

echo "[INFO] Eval test files:"
echo "  TruthyDPO -> ${TRUTHY_TEST}"
echo "  HH-RLHF   -> ${HHRLHF_TEST}"
echo "  PKU       -> ${PKU_TEST}"

python - <<'PY'
import torch, peft, trl
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
assert torch.cuda.device_count() >= 3, "Need at least 3 visible GPUs for this script"
PY

nvidia-smi || true

echo "[INFO] Starting stage1 training on 3 GPUs..."

CUDA_VISIBLE_DEVICES=0 python -u lora/preprocessing/train_truthy_stage1_lora.py \
  --model "${BASE_MODEL}" \
  --data "${D1_TRUTHY}" \
  --out "${OUT_TRUTHY}" \
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
  > "${LOG_DIR}/train_truthy_s${SEED}.log" 2>&1 &
PID_TRAIN_TRUTHY=$!
PIDS+=(${PID_TRAIN_TRUTHY})

CUDA_VISIBLE_DEVICES=1 python -u lora/preprocessing/train_truthy_stage1_lora.py \
  --model "${BASE_MODEL}" \
  --data "${D1_HHRLHF}" \
  --out "${OUT_HHRLHF}" \
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
  > "${LOG_DIR}/train_hhrlhf_s${SEED}.log" 2>&1 &
PID_TRAIN_HHRLHF=$!
PIDS+=(${PID_TRAIN_HHRLHF})

CUDA_VISIBLE_DEVICES=2 python -u lora/preprocessing/train_truthy_stage1_lora.py \
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
  > "${LOG_DIR}/train_pku_s${SEED}.log" 2>&1 &
PID_TRAIN_PKU=$!
PIDS+=(${PID_TRAIN_PKU})

wait ${PID_TRAIN_TRUTHY} ${PID_TRAIN_HHRLHF} ${PID_TRAIN_PKU}

for out in "$OUT_TRUTHY" "$OUT_HHRLHF" "$OUT_PKU"; do
  [ -f "${out}/adapter_model.safetensors" ] || { echo "[FATAL] Missing adapter weights in ${out}"; exit 3; }
done

echo "[INFO] Training complete. Starting evaluation..."

CUDA_VISIBLE_DEVICES=0 python -u eval/eval_stage1_accuracy.py \
  --base_model "${BASE_MODEL}" \
  --stage1_adapter "${OUT_TRUTHY}" \
  --test_jsonl "${TRUTHY_TEST}" \
  --out_json "${EVAL_DIR}/truthy_eps${EPS}_s${SEED}_eval.json" \
  --max_len "${EVAL_MAX_LEN}" \
  > "${LOG_DIR}/eval_truthy_s${SEED}.log" 2>&1 &
PID_EVAL_TRUTHY=$!
PIDS+=(${PID_EVAL_TRUTHY})

CUDA_VISIBLE_DEVICES=1 python -u eval/eval_stage1_accuracy.py \
  --base_model "${BASE_MODEL}" \
  --stage1_adapter "${OUT_HHRLHF}" \
  --test_jsonl "${HHRLHF_TEST}" \
  --out_json "${EVAL_DIR}/hhrlhf_eps${EPS}_s${SEED}_eval.json" \
  --max_len "${EVAL_MAX_LEN}" \
  > "${LOG_DIR}/eval_hhrlhf_s${SEED}.log" 2>&1 &
PID_EVAL_HHRLHF=$!
PIDS+=(${PID_EVAL_HHRLHF})

CUDA_VISIBLE_DEVICES=2 python -u eval/eval_stage1_accuracy.py \
  --base_model "${BASE_MODEL}" \
  --stage1_adapter "${OUT_PKU}" \
  --test_jsonl "${PKU_TEST}" \
  --out_json "${EVAL_DIR}/pku_eps${EPS}_s${SEED}_eval.json" \
  --max_len "${EVAL_MAX_LEN}" \
  > "${LOG_DIR}/eval_pku_s${SEED}.log" 2>&1 &
PID_EVAL_PKU=$!
PIDS+=(${PID_EVAL_PKU})

wait ${PID_EVAL_TRUTHY} ${PID_EVAL_HHRLHF} ${PID_EVAL_PKU}

python - <<PY
import json, os
out_root = "${OUT_ROOT}"
eval_dir = os.path.join(out_root, "eval")
paths = {
    "TruthyDPO": os.path.join(eval_dir, "truthy_eps${EPS}_s${SEED}_eval.json"),
    "HH-RLHF": os.path.join(eval_dir, "hhrlhf_eps${EPS}_s${SEED}_eval.json"),
    "PKU-SafeRLHF": os.path.join(eval_dir, "pku_eps${EPS}_s${SEED}_eval.json"),
}
summary = {}
for name, p in paths.items():
    if not os.path.exists(p):
        summary[name] = {"error": f"missing: {p}"}
        continue
    d = json.load(open(p))
    summary[name] = {
        "accuracy": d.get("accuracy"),
        "ece": d.get("ece"),
        "n": d.get("n"),
        "stage1_adapter": d.get("stage1_adapter"),
        "eval_json": p,
    }

json_path = os.path.join(out_root, "STAGE1_ACCURACY_SUMMARY.json")
md_path = os.path.join(out_root, "STAGE1_ACCURACY_SUMMARY.md")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Stage1 Accuracy Summary (eps=${EPS}, seed=${SEED})\\n\\n")
    f.write("| Dataset | Accuracy | ECE | N | Eval JSON |\\n")
    f.write("|---|---:|---:|---:|---|\\n")
    for ds in ("TruthyDPO", "HH-RLHF", "PKU-SafeRLHF"):
        s = summary.get(ds, {})
        if "error" in s:
            f.write(f"| {ds} | ERR | ERR | ERR | `{s['error']}` |\\n")
        else:
            f.write(f"| {ds} | {s['accuracy']:.4f} | {s['ece']:.4f} | {s['n']} | `{s['eval_json']}` |\\n")

print("Wrote", json_path)
print("Wrote", md_path)
PY

trap - EXIT INT TERM
echo "[DONE] Stage1 train+eval complete: ${OUT_ROOT}"
