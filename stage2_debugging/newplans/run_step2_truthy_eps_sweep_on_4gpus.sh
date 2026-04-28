#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
S1_EPS="${S1_EPS:-1.0}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"  # 1=true, 0=false

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${S1_EPS}_seed${SEED}/truthy_eps${S1_EPS}_s${SEED}}"
TEST_JSONL="${TEST_JSONL:-${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage2_truthy/eps_sweep_seed${SEED}}"
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

for f in "${S1_ADAPTER}/adapter_model.safetensors" "${TEST_JSONL}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

for eps in 0.3 0.5 1.0 2.0; do
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${eps}_seed${SEED}.jsonl"
  [ -f "${d2}" ] || {
    echo "[FATAL] Missing D2 for eps=${eps}: ${d2}"
    echo "        Generate it first via rr_stream_flip before submitting this sweep."
    exit 2
  }
done

python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
assert torch.cuda.device_count() >= 4, "Need at least 4 visible GPUs"
PY

nvidia-smi || true

run_train() {
  local method="$1" eps="$2" gpu="$3"
  local d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${eps}_seed${SEED}.jsonl"
  local out
  local log
  case "${method}" in
    sb)
      out="${OUT_ROOT}/sb_fresh_truthy_eps${eps}_s${SEED}"
      log="${LOG_DIR}/train_sb_fresh_eps${eps}.log"
      ;;
    map)
      out="${OUT_ROOT}/map_retrain_truthy_eps${eps}_s${SEED}"
      log="${LOG_DIR}/train_map_retrain_eps${eps}.log"
      ;;
    mle)
      out="${OUT_ROOT}/mle_fresh_truthy_eps${eps}_s${SEED}"
      log="${LOG_DIR}/train_mle_fresh_eps${eps}.log"
      ;;
    *)
      echo "[FATAL] unknown method=${method}"
      return 2
      ;;
  esac

  if [[ "${SKIP_EXISTING}" == "1" ]] && [[ -f "${out}/M2_manifest.json" ]]; then
    echo "[SKIP] ${method} eps=${eps} already has manifest: ${out}/M2_manifest.json" | tee -a "${log}"
    return 0
  fi

  mkdir -p "${out}"
  echo "[RUN] gpu=${gpu} method=${method} eps=${eps} out=${out}" | tee -a "${log}"

  if [[ "${method}" == "sb" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_sb_fresh.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${S1_ADAPTER}" \
      --data "${d2}" \
      --out "${out}" \
      --epsilon "${eps}" \
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
      >> "${log}" 2>&1
  elif [[ "${method}" == "map" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_map_retrain.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${S1_ADAPTER}" \
      --data "${d2}" \
      --out "${out}" \
      --epsilon "${eps}" \
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
      >> "${log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_mle_fresh.py \
      --model "${BASE_MODEL}" \
      --data "${d2}" \
      --out "${out}" \
      --epsilon "${eps}" \
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
      >> "${log}" 2>&1
  fi
}

# Balanced per-GPU queues (approx equal total walltime).
# Existing truthy eps=1.0 map/sb are typically skipped when SKIP_EXISTING=1.
queue_gpu0() {
  run_train sb 0.3 0
  run_train mle 0.3 0
  run_train mle 2.0 0
}
queue_gpu1() {
  run_train sb 0.5 1
  run_train map 0.3 1
}
queue_gpu2() {
  run_train sb 2.0 2
  run_train map 0.5 2
}
queue_gpu3() {
  run_train map 2.0 3
  run_train mle 0.5 3
  run_train mle 1.0 3
}

PIDS=()
NAMES=()
launch_queue() {
  local name="$1"
  shift
  ( "$@" ) > "${LOG_DIR}/${name}.queue.log" 2>&1 &
  PIDS+=($!)
  NAMES+=("${name}")
}

launch_queue "gpu0" queue_gpu0
launch_queue "gpu1" queue_gpu1
launch_queue "gpu2" queue_gpu2
launch_queue "gpu3" queue_gpu3

FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if ! wait "${pid}"; then
    echo "[FATAL] queue ${name} failed (pid=${pid}); see ${LOG_DIR}/${name}.queue.log"
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  exit 4
fi

echo "[INFO] Training queues complete. Running eval sweep..."

eval_one() {
  local method="$1" eps="$2"
  local out manifest eval_json log
  out="${OUT_ROOT}/${method}_truthy_eps${eps}_s${SEED}"
  manifest="${out}/M2_manifest.json"
  eval_json="${EVAL_DIR}/${method}_truthy_eps${eps}_s${SEED}_eval.json"
  log="${LOG_DIR}/eval_${method}_eps${eps}.log"

  if [[ ! -f "${manifest}" ]]; then
    echo "[WARN] missing manifest for ${method} eps=${eps}: ${manifest}" | tee -a "${log}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
    --manifest "${manifest}" \
    --test_jsonl "${TEST_JSONL}" \
    --out_json "${eval_json}" \
    --max_len 512 \
    >> "${log}" 2>&1
}

for eps in 0.3 0.5 1.0 2.0; do
  eval_one sb_fresh "${eps}"
  eval_one map_retrain "${eps}"
  eval_one mle_fresh "${eps}"
done

export OUT_ROOT EVAL_DIR SEED
python - <<'PY'
import json, os

out_root = os.environ["OUT_ROOT"]
eval_dir = os.environ["EVAL_DIR"]
seed = os.environ["SEED"]
eps_list = ["0.3", "0.5", "1.0", "2.0"]
methods = ["sb_fresh", "map_retrain", "mle_fresh"]

table = {}
for eps in eps_list:
    table[eps] = {}
    for m in methods:
        p = os.path.join(eval_dir, f"{m}_truthy_eps{eps}_s{seed}_eval.json")
        if os.path.exists(p):
            d = json.load(open(p, "r", encoding="utf-8"))
            table[eps][m] = {
                "accuracy": d.get("accuracy"),
                "ece": d.get("ece"),
                "mean_margin": d.get("mean_margin"),
                "n": d.get("n"),
                "eval_json": p,
            }
        else:
            table[eps][m] = {"error": f"missing: {p}"}

json_path = os.path.join(out_root, "TRUTHY_EPS_SWEEP_S42.json")
md_path = os.path.join(out_root, "TRUTHY_EPS_SWEEP_S42.md")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(table, f, indent=2)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Truthy Epsilon Sweep (seed=42)\n\n")
    f.write("| epsilon | SB-Fresh acc | MAP-Retrain acc | MLE-Fresh acc |\n")
    f.write("|---|---:|---:|---:|\n")
    for eps in eps_list:
        row = table[eps]
        def acc(method):
            v = row.get(method, {})
            if "error" in v or v.get("accuracy") is None:
                return "N/A"
            return f"{v['accuracy']:.4f}"
        f.write(f"| {eps} | {acc('sb_fresh')} | {acc('map_retrain')} | {acc('mle_fresh')} |\n")

print("Wrote", json_path)
print("Wrote", md_path)
PY

echo "[DONE] Truthy eps sweep complete: ${OUT_ROOT}"

