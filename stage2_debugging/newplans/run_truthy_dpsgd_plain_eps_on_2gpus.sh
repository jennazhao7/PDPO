#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"

# TruthyDPO unflipped (direct from fetch, no RR)
DATA_JSONL="${DATA_JSONL:-${ROOT_DIR}/preprocessing/truthydpo/truthy_dpo_subset.jsonl}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/results_truthy_dpsgd_plain_seed${SEED}}"
LOG_DIR="${OUT_ROOT}/logs"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

# DP-SGD alignment params
DELTA="${DELTA:-1e-5}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
EPOCHS="${EPOCHS:-3}"
LR_DP="${LR_DP:-5e-5}"
LR_PLAIN="${LR_PLAIN:-5e-5}"

# More training steps:
# - DP uses smaller effective batch for more optimizer updates while honoring epsilon budget.
# - Plain gets a large fixed max_steps budget.
DP_PER_DEVICE_BSZ="${DP_PER_DEVICE_BSZ:-1}"
DP_GRAD_ACCUM="${DP_GRAD_ACCUM:-32}"         # effective batch 32
PLAIN_PER_DEVICE_BSZ="${PLAIN_PER_DEVICE_BSZ:-1}"
PLAIN_GRAD_ACCUM="${PLAIN_GRAD_ACCUM:-16}"   # effective batch 16
PLAIN_MAX_STEPS="${PLAIN_MAX_STEPS:-600}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
# Keep explicit qkvo to match stage-2 fresh runs and avoid qsub -v comma parsing issues.
TARGET_MODULES_QKVO="q_proj,k_proj,v_proj,o_proj"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-512}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-200}"
USE_BF16_PLAIN="${USE_BF16_PLAIN:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

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

for f in "${DATA_JSONL}" \
         "${ROOT_DIR}/lora/dp-lora/train_dp_lora.py" \
         "${ROOT_DIR}/lora/plain-lora/train_plain_lora.py"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

run_dp() {
  local eps="$1"
  local gpu="$2"
  local tag="dpsgd_truthy_eps${eps}_s${SEED}"
  local out_dir="${OUT_ROOT}/${tag}"
  local log_file="${LOG_DIR}/train_${tag}.log"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${tag} summary exists: ${out_dir}/summary.json" | tee -a "${log_file}"
    return 0
  fi

  echo "[RUN] gpu=${gpu} ${tag}" | tee -a "${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u lora/dp-lora/train_dp_lora.py \
    --model_id "${BASE_MODEL}" \
    --dataset_path "${DATA_JSONL}" \
    --output_dir "${out_dir}" \
    --seed "${SEED}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR_DP}" \
    --per_device_train_batch_size "${DP_PER_DEVICE_BSZ}" \
    --gradient_accumulation_steps "${DP_GRAD_ACCUM}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --target_modules "${TARGET_MODULES_QKVO}" \
    --max_grad_norm "${MAX_GRAD_NORM}" \
    --delta "${DELTA}" \
    --epsilon "${eps}" \
    --dp \
    --max_eval_samples "${MAX_EVAL_SAMPLES}" \
    --logging_steps 10 \
    >> "${log_file}" 2>&1
}

run_plain() {
  local eps_tag="$1"
  local gpu="$2"
  local tag="plain_lora_truthy_eps${eps_tag}_s${SEED}"
  local out_dir="${OUT_ROOT}/${tag}"
  local log_file="${LOG_DIR}/train_${tag}.log"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${out_dir}/summary.json" ]]; then
    echo "[SKIP] ${tag} summary exists: ${out_dir}/summary.json" | tee -a "${log_file}"
    return 0
  fi

  echo "[RUN] gpu=${gpu} ${tag}" | tee -a "${log_file}"
  if [[ "${USE_BF16_PLAIN}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u lora/plain-lora/train_plain_lora.py \
      --model_id "${BASE_MODEL}" \
      --dataset_path "${DATA_JSONL}" \
      --output_dir "${out_dir}" \
      --seed "${SEED}" \
      --num_train_epochs "${EPOCHS}" \
      --max_steps "${PLAIN_MAX_STEPS}" \
      --learning_rate "${LR_PLAIN}" \
      --per_device_train_batch_size "${PLAIN_PER_DEVICE_BSZ}" \
      --gradient_accumulation_steps "${PLAIN_GRAD_ACCUM}" \
      --max_seq_length "${MAX_SEQ_LENGTH}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --target_modules "${TARGET_MODULES_QKVO}" \
      --max_grad_norm "${MAX_GRAD_NORM}" \
      --max_eval_samples "${MAX_EVAL_SAMPLES}" \
      --logging_steps 10 \
      --bf16 \
      >> "${log_file}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -u lora/plain-lora/train_plain_lora.py \
      --model_id "${BASE_MODEL}" \
      --dataset_path "${DATA_JSONL}" \
      --output_dir "${out_dir}" \
      --seed "${SEED}" \
      --num_train_epochs "${EPOCHS}" \
      --max_steps "${PLAIN_MAX_STEPS}" \
      --learning_rate "${LR_PLAIN}" \
      --per_device_train_batch_size "${PLAIN_PER_DEVICE_BSZ}" \
      --gradient_accumulation_steps "${PLAIN_GRAD_ACCUM}" \
      --max_seq_length "${MAX_SEQ_LENGTH}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --target_modules "${TARGET_MODULES_QKVO}" \
      --max_grad_norm "${MAX_GRAD_NORM}" \
      --max_eval_samples "${MAX_EVAL_SAMPLES}" \
      --logging_steps 10 \
      >> "${log_file}" 2>&1
  fi

  python - <<'PY' "${out_dir}" "${log_file}"
import json, os, sys
out_dir, log_file = sys.argv[1], sys.argv[2]
metrics = os.path.join(out_dir, "metrics.jsonl")
first = None
last = None
if os.path.isfile(metrics):
    with open(metrics, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if "loss" in row:
                v = float(row["loss"])
                if first is None:
                    first = v
                last = v
with open(log_file, "a", encoding="utf-8") as f:
    if first is None or last is None:
        f.write("[CHECK] plain training loss not found in metrics.jsonl\n")
    else:
        delta = last - first
        f.write(f"[CHECK] plain loss first={first:.6f} last={last:.6f} delta={delta:.6f}\n")
        if delta < -0.01:
            f.write("[CHECK] PASS: plain loss decreased.\n")
        else:
            f.write("[CHECK] WARN: plain loss did not decrease enough.\n")
PY
}

# Three-GPU queues (no conflict):
#   GPU0: DP eps=1.0
#   GPU1: DP eps=0.5
#   GPU2: Plain eps1.0 then plain eps0.5 (same config, separate tags)
(
  run_dp "1.0" "0"
) &
PID0=$!

(
  run_dp "0.5" "1"
) &
PID1=$!

(
  run_plain "1.0" "2"
  run_plain "0.5" "2"
) &
PID2=$!

wait "${PID0}" "${PID1}" "${PID2}"

python - <<'PY' "${OUT_ROOT}" "${SEED}"
import json, os, sys
out_root = sys.argv[1]
seed = sys.argv[2]
rows = [
    ("dpsgd_lora", "1.0", f"dpsgd_truthy_eps1.0_s{seed}"),
    ("dpsgd_lora", "0.5", f"dpsgd_truthy_eps0.5_s{seed}"),
    ("plain_lora", "1.0", f"plain_lora_truthy_eps1.0_s{seed}"),
    ("plain_lora", "0.5", f"plain_lora_truthy_eps0.5_s{seed}"),
]
table = []
for method, eps, tag in rows:
    s = os.path.join(out_root, tag, "summary.json")
    if not os.path.isfile(s):
        table.append({"method": method, "epsilon": eps, "status": "missing_summary"})
        continue
    with open(s, "r", encoding="utf-8") as f:
        j = json.load(f)
    table.append({
        "method": method,
        "epsilon": eps,
        "pairwise_accuracy": j.get("pairwise_accuracy"),
        "train_n": j.get("train_n"),
        "val_n": j.get("val_n"),
        "epsilon_spent": j.get("epsilon_spent"),
        "dp_enabled": j.get("dp_enabled"),
    })

summary_json = os.path.join(out_root, "comparison_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(table, f, indent=2, ensure_ascii=False)

summary_md = os.path.join(out_root, "comparison_summary.md")
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("| Method | eps | pairwise_accuracy | epsilon_spent | train_n | val_n |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for r in table:
        f.write(
            f"| {r.get('method')} | {r.get('epsilon')} | "
            f"{r.get('pairwise_accuracy')} | {r.get('epsilon_spent')} | "
            f"{r.get('train_n')} | {r.get('val_n')} |\n"
        )
print("[DONE] wrote", summary_json)
print("[DONE] wrote", summary_md)
PY

echo "[DONE] 3-GPU Truthy DP-SGD + plain LoRA sweep complete: ${OUT_ROOT}"
