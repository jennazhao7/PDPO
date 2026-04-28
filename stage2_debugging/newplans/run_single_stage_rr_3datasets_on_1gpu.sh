#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
EPS="${EPS:-1.0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
FORCE_REBUILD_DATA="${FORCE_REBUILD_DATA:-0}"

EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
BETA="${BETA:-0.5}"
BSZ="${BSZ:-1}"
GA="${GA:-16}"
MAX_LEN="${MAX_LEN:-512}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/single_stage_rr_seed${SEED}_eps${EPS}}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"
DATA_DIR="${OUT_ROOT}/data"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${EVAL_DIR}" "${DATA_DIR}"

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
assert torch.cuda.device_count() >= 1, "Need at least 1 visible GPU"
PY

nvidia-smi || true

for f in \
  "${ROOT_DIR}/stage2_debugging/train_stage2_mle_fresh.py" \
  "${ROOT_DIR}/stage2_debugging/eval_preference_accuracy.py" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d1_rr_flipped_truthy_eps${EPS}_seed${SEED}.jsonl" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${EPS}_seed${SEED}.jsonl" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d1_rr_flipped_hhrlhf_eps${EPS}_seed${SEED}.jsonl" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_hhrlhf_eps${EPS}_seed${SEED}.jsonl" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d1_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl" \
  "${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_pku_eps${EPS}_seed${SEED}.jsonl" \
  "${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl" \
  "${ROOT_DIR}/stage2_debugging/test_pref.jsonl" \
  "${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl"; do
  [ -f "${f}" ] || { echo "[FATAL] Missing required file: ${f}"; exit 2; }
done

build_full_data() {
  local ds="$1"
  local d1="${ROOT_DIR}/stage2_debugging/preprocessing/d1_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  local d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  local out="${DATA_DIR}/d_full_${ds}_eps${EPS}_seed${SEED}.jsonl"

  if [[ "${FORCE_REBUILD_DATA}" == "0" && -f "${out}" ]]; then
    echo "${out}"
    return 0
  fi

  python - <<'PY' "${d1}" "${d2}" "${out}" "${ds}"
import json, sys
d1, d2, out, ds = sys.argv[1:5]
rows = []
for p in (d1, d2):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
with open(out, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[DATA] {ds}: wrote {len(rows)} rows -> {out}", file=sys.stderr)
PY
  echo "${out}"
}

test_jsonl_for() {
  local ds="$1"
  case "${ds}" in
    truthy) echo "${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl" ;;
    hhrlhf) echo "${ROOT_DIR}/stage2_debugging/test_pref.jsonl" ;;
    pku) echo "${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl" ;;
    *) echo "[FATAL] unknown dataset: ${ds}" >&2; exit 2 ;;
  esac
}

run_one() {
  local ds="$1"
  local full_data test_json out_dir train_log eval_log manifest eval_json

  full_data="$(build_full_data "${ds}")"
  test_json="$(test_jsonl_for "${ds}")"
  out_dir="${OUT_ROOT}/single_stage_rr_${ds}_eps${EPS}_s${SEED}"
  train_log="${LOG_DIR}/train_${ds}.log"
  eval_log="${LOG_DIR}/eval_${ds}.log"
  manifest="${out_dir}/M2_manifest.json"
  eval_json="${EVAL_DIR}/single_stage_rr_${ds}_eps${EPS}_s${SEED}_eval.json"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${manifest}" && -f "${eval_json}" ]]; then
    echo "[SKIP] ${ds}: manifest+eval exist" | tee -a "${train_log}"
    return 0
  fi

  mkdir -p "${out_dir}"
  echo "[RUN] single-stage-rr ${ds} (gpu=0)" | tee -a "${train_log}"
  CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/train_stage2_mle_fresh.py \
    --model "${BASE_MODEL}" \
    --data "${full_data}" \
    --out "${out_dir}" \
    --epsilon "${EPS}" \
    --beta "${BETA}" \
    --lr "${LR}" \
    --epochs "${EPOCHS}" \
    --max_steps -1 \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --bsz "${BSZ}" \
    --ga "${GA}" \
    --max_len "${MAX_LEN}" \
    --bf16 \
    --seed "${SEED}" \
    >> "${train_log}" 2>&1

  [ -f "${manifest}" ] || { echo "[FATAL] missing manifest: ${manifest}" | tee -a "${train_log}"; exit 3; }

  CUDA_VISIBLE_DEVICES=0 python -u stage2_debugging/eval_preference_accuracy.py \
    --manifest "${manifest}" \
    --test_jsonl "${test_json}" \
    --out_json "${eval_json}" \
    --max_len "${MAX_LEN}" \
    >> "${eval_log}" 2>&1

  echo "[DONE] ${ds}"
}

run_one truthy
run_one hhrlhf
run_one pku

python - <<'PY' "${EVAL_DIR}" "${OUT_ROOT}" "${EPS}" "${SEED}"
import json, os, sys
eval_dir, out_root, eps, seed = sys.argv[1:5]
rows = []
for ds in ["truthy", "hhrlhf", "pku"]:
    p = os.path.join(eval_dir, f"single_stage_rr_{ds}_eps{eps}_s{seed}_eval.json")
    if not os.path.isfile(p):
        rows.append({"dataset": ds, "status": "missing_eval"})
        continue
    with open(p, "r", encoding="utf-8") as f:
        j = json.load(f)
    rows.append({
        "dataset": ds,
        "accuracy": j.get("accuracy"),
        "ece": j.get("ece"),
        "n": j.get("n"),
    })
summary_json = os.path.join(out_root, "single_stage_rr_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)
summary_md = os.path.join(out_root, "single_stage_rr_summary.md")
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("| Dataset | Accuracy | ECE | N |\n")
    f.write("|---|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r.get('dataset')} | {r.get('accuracy')} | {r.get('ece')} | {r.get('n')} |\n")
print("[DONE] wrote", summary_json)
print("[DONE] wrote", summary_md)
PY

echo "[DONE] single-stage RR 3-dataset run complete: ${OUT_ROOT}"
