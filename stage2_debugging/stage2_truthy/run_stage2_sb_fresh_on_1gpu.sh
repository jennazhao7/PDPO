#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname)  date=$(date)"

# ── Environment (all overridable via qsub -v) ──────────────────────────────
ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
DATASET="${DATASET:-truthy}"   # truthy | hhrlhf | pku
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

# ── Resolve dataset-specific defaults ─────────────────────────────────────
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

# ── Activate conda ────────────────────────────────────────────────────────
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

# ── Preflight checks ──────────────────────────────────────────────────────
for f in "${S1_ADAPTER}/adapter_model.safetensors" "${D2_DATA}" "${TEST_JSONL}"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

python - <<'PY'
import os
import torch, peft, trl
from datasets import load_dataset
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
assert torch.cuda.device_count() >= 1, "Need at least 1 visible GPU"
path = os.environ["D2_DATA"]
ds = load_dataset("json", data_files=path)["train"]
cols = set(ds.column_names)
required = {"prompt", "chosen", "rejected"}
missing = sorted(required - cols)
assert len(ds) > 0, f"D2 dataset is empty: {path}"
assert not missing, f"D2 missing required columns {missing}; got {sorted(cols)}"
print("d2_rows:", len(ds), "d2_cols:", sorted(cols))
PY

nvidia-smi || true

# ── Output directory for this variant ────────────────────────────────────
OUT_SB_FRESH="${OUT_ROOT}/sb_fresh_${DATASET}_eps${EPS}_s${SEED}"

echo "[INFO] Launching SB-Fresh on GPU 0 for dataset=${DATASET} eps=${EPS} seed=${SEED}..."

CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/train_stage2_sb_fresh.py \
  --model "${BASE_MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${OUT_SB_FRESH}" \
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
  2>&1 | tee "${LOG_DIR}/train_sb_fresh_${DATASET}_s${SEED}.log"

TRAIN_EXIT=${PIPESTATUS[0]}
if [[ "${TRAIN_EXIT}" -ne 0 ]]; then
  echo "[FATAL] SB-Fresh training exited with code ${TRAIN_EXIT}. See ${LOG_DIR}/train_sb_fresh_${DATASET}_s${SEED}.log"
  exit "${TRAIN_EXIT}"
fi

# Verify manifest was created
MANIFEST="${OUT_SB_FRESH}/M2_manifest.json"
[ -f "${MANIFEST}" ] || { echo "[FATAL] Missing manifest: ${MANIFEST}"; exit 3; }

echo "[INFO] Training done. Running eval..."

# eval_preference_accuracy.py detects single-adapter manifest → has_stage1=False
# → policy = base+stage2, ref = base  (consistent with SB-Fresh training)
CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "${MANIFEST}" \
  --test_jsonl "${TEST_JSONL}" \
  --out_json "${EVAL_DIR}/sb_fresh_${DATASET}_eps${EPS}_s${SEED}_eval.json" \
  --max_len 512 \
  2>&1 | tee "${LOG_DIR}/eval_sb_fresh_${DATASET}_s${SEED}.log"

EVAL_EXIT=${PIPESTATUS[0]}
if [[ "${EVAL_EXIT}" -ne 0 ]]; then
  echo "[FATAL] SB-Fresh eval exited with code ${EVAL_EXIT}"
  exit "${EVAL_EXIT}"
fi

# ── Write comparison summary (appends SB-Fresh alongside existing 4-variant results) ──
export OUT_ROOT DATASET EPS SEED EVAL_DIR OUT_SB_FRESH
python - <<'PY'
import json, os

out_root = os.environ["OUT_ROOT"]
eval_dir = os.environ["EVAL_DIR"]
dataset  = os.environ["DATASET"]
eps      = os.environ["EPS"]
seed     = os.environ["SEED"]

# All six result paths (original 4 + SB-Fresh + MAP-Retrain comparison)
paths = {
    "MLE-DPO":    os.path.join(eval_dir, f"mle_dpo_{dataset}_eps{eps}_s{seed}_eval.json"),
    "PROPS-MAP":  os.path.join(eval_dir, f"props_map_{dataset}_eps{eps}_s{seed}_eval.json"),
    "Soft-Bayes": os.path.join(eval_dir, f"soft_bayes_{dataset}_eps{eps}_s{seed}_eval.json"),
    "MAP-Retrain":os.path.join(eval_dir, f"map_retrain_{dataset}_eps{eps}_s{seed}_eval.json"),
    "SB-Fresh":   os.path.join(eval_dir, f"sb_fresh_{dataset}_eps{eps}_s{seed}_eval.json"),
}

summary = {}
for name, p in paths.items():
    if not os.path.exists(p):
        summary[name] = {"error": f"missing: {p}"}
        continue
    d = json.load(open(p))
    summary[name] = {
        "accuracy":    d.get("accuracy"),
        "ece":         d.get("ece"),
        "mean_margin": d.get("mean_margin"),
        "n":           d.get("n"),
        "eval_json":   p,
    }

dataset_tag = dataset.upper()
json_path = os.path.join(out_root, f"STAGE2_{dataset_tag}_SUMMARY_WITH_SBFRESH.json")
md_path   = os.path.join(out_root, f"STAGE2_{dataset_tag}_SUMMARY_WITH_SBFRESH.md")

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

with open(md_path, "w", encoding="utf-8") as f:
    f.write(f"# Stage2 {dataset} Summary — 5 variants (eps={eps}, seed={seed})\n\n")
    f.write("| Variant | Accuracy | ECE | Mean margin | N | Eval JSON |\n")
    f.write("|---|---:|---:|---:|---:|---|\n")
    order = ("SB-Fresh", "MAP-Retrain", "Soft-Bayes", "PROPS-MAP", "MLE-DPO")
    for k in order:
        s = summary.get(k, {})
        if "error" in s:
            f.write(f"| {k} | ERR | ERR | ERR | ERR | {s['error']} |\n")
        else:
            acc  = f"{s['accuracy']:.4f}"   if s.get('accuracy')    is not None else "N/A"
            ece  = f"{s['ece']:.4f}"        if s.get('ece')         is not None else "N/A"
            mm   = f"{s['mean_margin']:.4f}" if s.get('mean_margin') is not None else "N/A"
            n    = s.get('n', 'N/A')
            evp  = s.get('eval_json', '')
            f.write(f"| {k} | {acc} | {ece} | {mm} | {n} | {evp} |\n")

print(f"[Summary] {json_path}")
print(f"[Summary] {md_path}")
PY

echo "[DONE] SB-Fresh run complete: ${OUT_SB_FRESH}"
