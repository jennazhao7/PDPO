#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
SEED="${SEED:-42}"
EPS="${EPS:-1.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
MAX_LEN="${MAX_LEN:-512}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/crossdata_eval/truthy_eps${EPS}_s${SEED}_on_hhrlhf}"
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
assert torch.cuda.device_count() >= 4, "Need at least 4 visible GPUs"
PY

nvidia-smi || true

TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/test_pref.jsonl}"
MANIFEST_SB_FRESH="${MANIFEST_SB_FRESH:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/sb_fresh_truthy_eps1.0_s42/M2_manifest.json}"
MANIFEST_MAP_RETRAIN="${MANIFEST_MAP_RETRAIN:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/map_retrain_truthy_eps1.0_s42/M2_manifest.json}"
MANIFEST_MLE_DPO="${MANIFEST_MLE_DPO:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/mle_dpo_truthy_eps1.0_s42/M2_manifest.json}"
MANIFEST_SOFT_BAYES="${MANIFEST_SOFT_BAYES:-${ROOT_DIR}/stage2_debugging/stage2_truthy/results_eps1.0_seed42/soft_bayes_truthy_eps1.0_s42/M2_manifest.json}"

for f in "${ROOT_DIR}/stage2_debugging/eval_preference_accuracy.py" "${TEST_JSONL}" \
         "${MANIFEST_SB_FRESH}" "${MANIFEST_MAP_RETRAIN}" "${MANIFEST_MLE_DPO}" "${MANIFEST_SOFT_BAYES}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

run_eval() {
  local method="$1"
  local manifest="$2"
  local gpu="$3"
  local out_json="${EVAL_DIR}/${method}_truthy_eps${EPS}_s${SEED}_on_hhrlhf_eval.json"
  local log_file="${LOG_DIR}/eval_${method}.log"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_json}" ]]; then
    echo "[SKIP] ${method}: ${out_json}" | tee -a "${log_file}"
    return 0
  fi

  echo "[RUN] gpu=${gpu} method=${method}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/eval_preference_accuracy.py \
    --manifest "${manifest}" \
    --test_jsonl "${TEST_JSONL}" \
    --out_json "${out_json}" \
    --max_len "${MAX_LEN}" \
    >> "${log_file}" 2>&1
  echo "[DONE] ${method}" | tee -a "${log_file}"
}

( run_eval "sb_fresh" "${MANIFEST_SB_FRESH}" 0 ) & P1=$!
( run_eval "map_retrain" "${MANIFEST_MAP_RETRAIN}" 1 ) & P2=$!
( run_eval "mle_dpo" "${MANIFEST_MLE_DPO}" 2 ) & P3=$!
( run_eval "soft_bayes" "${MANIFEST_SOFT_BAYES}" 3 ) & P4=$!
wait "${P1}" "${P2}" "${P3}" "${P4}"

python - <<'PY' "${EVAL_DIR}" "${OUT_ROOT}" "${EPS}" "${SEED}"
import json, os, sys
eval_dir, out_root, eps, seed = sys.argv[1:5]
methods = ["sb_fresh", "map_retrain", "mle_dpo", "soft_bayes"]
rows = []
for m in methods:
    p = os.path.join(eval_dir, f"{m}_truthy_eps{eps}_s{seed}_on_hhrlhf_eval.json")
    if not os.path.isfile(p):
        rows.append({"method": m, "status": "missing_eval"})
        continue
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    rows.append({
        "method": m,
        "accuracy": j.get("accuracy"),
        "ece": j.get("ece"),
        "n": j.get("n"),
    })

summary_json = os.path.join(out_root, "crossdata_eval_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

summary_md = os.path.join(out_root, "crossdata_eval_summary.md")
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("| Method | Accuracy (on HH-RLHF test) | ECE | N |\n")
    f.write("|---|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r.get('method')} | {r.get('accuracy')} | {r.get('ece')} | {r.get('n')} |\n")

print("[DONE] wrote", summary_json)
print("[DONE] wrote", summary_md)
PY

echo "[DONE] cross-data eval complete: ${OUT_ROOT}"
