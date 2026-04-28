#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
DATASET="${DATASET:-truthy}"   # truthy | hhrlhf | pku
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

case "${DATASET}" in
  truthy)
    DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"
    ;;
  hhrlhf)
    DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/test_pref.jsonl"
    ;;
  pku)
    DEFAULT_TEST="${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl"
    ;;
  *)
    echo "[FATAL] Unsupported DATASET=${DATASET}. Use one of: truthy, hhrlhf, pku"
    exit 2
    ;;
esac

S1_ADAPTER="${S1_ADAPTER:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${DATASET}_eps${EPS}_s${SEED}}"
D2_DATA="${D2_DATA:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${DATASET}_eps${EPS}_seed${SEED}.jsonl}"
TEST_JSONL="${TEST_JSONL:-${DEFAULT_TEST}}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/stage2_${DATASET}/results_eps${EPS}_seed${SEED}}"
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
export D2_DATA

for f in "${S1_ADAPTER}/adapter_model.safetensors" "${D2_DATA}" "${TEST_JSONL}"; do
  [ -f "$f" ] || { echo "[FATAL] Missing required file: $f"; exit 2; }
done

python - <<'PY'
import os
import torch, peft, trl
from datasets import load_dataset
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
assert torch.cuda.device_count() >= 4, "Need at least 4 visible GPUs"
path = os.environ["D2_DATA"]
ds = load_dataset("json", data_files=path)["train"]
cols = set(ds.column_names)
required = {"prompt", "chosen", "rejected"}
missing = sorted(required - cols)
assert len(ds) > 0, f"D2 dataset is empty: {path}"
assert not missing, f"D2 missing required columns {missing}; columns={sorted(cols)}"
print("d2_rows:", len(ds), "d2_cols:", sorted(cols))
PY

nvidia-smi || true

PIDS=()
NAMES=()
cleanup_children() {
  for p in "${PIDS[@]:-}"; do
    if kill -0 "$p" 2>/dev/null; then
      kill "$p" 2>/dev/null || true
    fi
  done
}
trap cleanup_children EXIT INT TERM

OUT_MLE="${OUT_ROOT}/mle_dpo_${DATASET}_eps${EPS}_s${SEED}"
OUT_PROPS="${OUT_ROOT}/props_map_${DATASET}_eps${EPS}_s${SEED}"
OUT_SB="${OUT_ROOT}/soft_bayes_${DATASET}_eps${EPS}_s${SEED}"
OUT_RETRAIN="${OUT_ROOT}/map_retrain_${DATASET}_eps${EPS}_s${SEED}"

echo "[INFO] Launching 4 Stage2 variants for dataset=${DATASET}..."

CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/train_stage2_dpo.py \
  --model "${BASE_MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${OUT_MLE}" \
  --seed "${SEED}" \
  --epochs 3 \
  --max_steps -1 \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bsz 1 \
  --ga 16 \
  --bf16 \
  > "${LOG_DIR}/train_mle_dpo_${DATASET}_s${SEED}.log" 2>&1 &
PIDS+=($!)
NAMES+=("MLE-DPO")

CUDA_VISIBLE_DEVICES=1 python -u stage2_debugging/train_stage2_props_map.py \
  --model "${BASE_MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${OUT_PROPS}" \
  --epsilon "${EPS}" \
  --seed "${SEED}" \
  --epochs 3 \
  --max_steps -1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bsz 1 \
  --ga 16 \
  --bf16 \
  > "${LOG_DIR}/train_props_map_${DATASET}_s${SEED}.log" 2>&1 &
PIDS+=($!)
NAMES+=("PROPS-MAP")

CUDA_VISIBLE_DEVICES=2 python -u stage2_debugging/train_stage2_soft_bayes.py \
  --model "${BASE_MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${OUT_SB}" \
  --epsilon "${EPS}" \
  --beta 0.5 \
  --lr 2.5e-5 \
  --enforce_calibrated \
  --seed "${SEED}" \
  --epochs 3 \
  --max_steps -1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bsz 1 \
  --ga 16 \
  --bf16 \
  > "${LOG_DIR}/train_soft_bayes_${DATASET}_s${SEED}.log" 2>&1 &
PIDS+=($!)
NAMES+=("Soft-Bayes")

CUDA_VISIBLE_DEVICES=3 python -u stage2_debugging/train_stage2_map_retrain.py \
  --model "${BASE_MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${OUT_RETRAIN}" \
  --epsilon "${EPS}" \
  --seed "${SEED}" \
  --epochs 3 \
  --max_steps -1 \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --bsz 1 \
  --ga 16 \
  --bf16 \
  > "${LOG_DIR}/train_map_retrain_${DATASET}_s${SEED}.log" 2>&1 &
PIDS+=($!)
NAMES+=("MAP-Retrain")

FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if ! wait "${pid}"; then
    echo "[FATAL] ${name} process failed (pid=${pid}). See logs in ${LOG_DIR}"
    FAILED=1
    break
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  exit 4
fi

for f in \
  "${OUT_MLE}/M2_manifest.json" \
  "${OUT_PROPS}/M2_manifest.json" \
  "${OUT_SB}/M2_manifest.json" \
  "${OUT_RETRAIN}/M2_manifest.json"; do
  [ -f "$f" ] || { echo "[FATAL] Missing manifest: $f"; exit 3; }
done

echo "[INFO] Training complete. Running eval..."

# Eval A/B/baseline stacked variants with stage2 metric
CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "${OUT_MLE}/M2_manifest.json" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/mle_dpo_${DATASET}_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  > "${LOG_DIR}/eval_mle_dpo_${DATASET}_s${SEED}.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "${OUT_PROPS}/M2_manifest.json" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/props_map_${DATASET}_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  > "${LOG_DIR}/eval_props_map_${DATASET}_s${SEED}.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "${OUT_SB}/M2_manifest.json" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/soft_bayes_${DATASET}_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  > "${LOG_DIR}/eval_soft_bayes_${DATASET}_s${SEED}.log" 2>&1

# Eval retrain variant as stage1-style (single adapter vs base)
CUDA_VISIBLE_DEVICES=0 python -u eval/eval_stage1_accuracy.py \
  --base_model "${BASE_MODEL}" \
  --stage1_adapter "${OUT_RETRAIN}/fresh_lora" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/map_retrain_${DATASET}_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  > "${LOG_DIR}/eval_map_retrain_${DATASET}_s${SEED}.log" 2>&1

export OUT_ROOT DATASET EPS SEED EVAL_DIR
python - <<'PY'
import json
import os

out_root = os.environ["OUT_ROOT"]
eval_dir = os.environ["EVAL_DIR"]
dataset = os.environ["DATASET"]
eps = os.environ["EPS"]
seed = os.environ["SEED"]

paths = {
    "MLE-DPO": os.path.join(eval_dir, f"mle_dpo_{dataset}_eps{eps}_s{seed}_eval.json"),
    "PROPS-MAP": os.path.join(eval_dir, f"props_map_{dataset}_eps{eps}_s{seed}_eval.json"),
    "Soft-Bayes": os.path.join(eval_dir, f"soft_bayes_{dataset}_eps{eps}_s{seed}_eval.json"),
    "MAP-Retrain": os.path.join(eval_dir, f"map_retrain_{dataset}_eps{eps}_s{seed}_eval.json"),
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
        "eval_json": p,
    }

dataset_tag = dataset.upper()
json_path = os.path.join(out_root, f"STAGE2_{dataset_tag}_SUMMARY.json")
md_path = os.path.join(out_root, f"STAGE2_{dataset_tag}_SUMMARY.md")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Stage2 {dataset} Summary (eps={eps}, seed={seed})\\n\\n")
    f.write("| Variant | Accuracy | ECE | N | Eval JSON |\\n")
    f.write("|---|---:|---:|---:|---|\\n")
    for k in ("MLE-DPO", "PROPS-MAP", "Soft-Bayes", "MAP-Retrain"):
        s = summary.get(k, {})
        if "error" in s:
            f.write(f"| {k} | ERR | ERR | ERR | {s['error']} |\\n")
        else:
            f.write(
                f"| {k} | {s['accuracy']:.4f} | {s['ece']:.4f} | {s['n']} | {s['eval_json']} |\\n"
            )

print("Wrote", json_path)
print("Wrote", md_path)
PY

trap - EXIT INT TERM
echo "[DONE] Stage2 ${DATASET} 4-variant run complete: ${OUT_ROOT}"
