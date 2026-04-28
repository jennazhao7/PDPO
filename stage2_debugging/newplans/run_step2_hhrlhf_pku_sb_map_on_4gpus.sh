#!/bin/bash
#$ -S /bin/bash
set -euo pipefail

echo "[INFO] host=$(hostname) date=$(date)"

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"   # 1: skip when M2_manifest exists
FORCE_RERUN_HHRLHF_SB="${FORCE_RERUN_HHRLHF_SB:-1}"  # 1: rerun hhrlhf sb_fresh even if partial exists
D2_FRACTION="${D2_FRACTION:-0.5}"     # 0<frac<=1 => deterministic subsample fraction
SUBSET_SEED="${SUBSET_SEED:-42}"

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

python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available(), "gpu_count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available in this qsub job"
assert torch.cuda.device_count() >= 4, "Need at least 4 visible GPUs"
PY

nvidia-smi || true

dataset_test_path() {
  local ds="$1"
  case "${ds}" in
    hhrlhf) echo "${ROOT_DIR}/stage2_debugging/test_pref.jsonl" ;;
    pku)    echo "${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl" ;;
    *) echo ""; return 1 ;;
  esac
}

dataset_stage1_adapter() {
  local ds="$1"
  echo "${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${ds}_eps${EPS}_s${SEED}"
}

dataset_d2() {
  local ds="$1"
  echo "${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
}

dataset_out_root() {
  local ds="$1"
  echo "${ROOT_DIR}/stage2_debugging/stage2_${ds}/results_eps${EPS}_seed${SEED}"
}

prepare_d2_subset() {
  local ds="$1" full_d2="$2" out_root="$3"
  local data_dir subset_d2
  data_dir="${out_root}/data"
  mkdir -p "${data_dir}"
  subset_d2="${data_dir}/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}_frac${D2_FRACTION}_subseed${SUBSET_SEED}.jsonl"

  if [[ -f "${subset_d2}" ]]; then
    echo "${subset_d2}"
    return 0
  fi

  export FULL_D2="${full_d2}" SUBSET_D2="${subset_d2}" D2_FRACTION SUBSET_SEED
  python - <<'PY'
import json, os, random, sys
src = os.environ["FULL_D2"]
dst = os.environ["SUBSET_D2"]
frac = float(os.environ["D2_FRACTION"])
seed = int(os.environ["SUBSET_SEED"])

rows = [json.loads(x) for x in open(src, "r", encoding="utf-8") if x.strip()]
n = len(rows)
if n == 0:
    raise RuntimeError(f"Empty D2: {src}")
if not (0 < frac <= 1.0):
    raise RuntimeError(f"D2_FRACTION must be in (0,1], got {frac}")
target = max(1, int(round(n * frac)))
rng = random.Random(seed)
rng.shuffle(rows)
rows = rows[:target]
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
with open(dst, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[Subset] {src} -> {dst} rows={target}/{n} frac={frac} seed={seed}", file=sys.stderr)
PY

  echo "${subset_d2}"
}

run_one() {
  local ds="$1" method="$2" gpu="$3"
  local s1 d2 out_root out_dir log_dir eval_dir test_jsonl manifest
  s1="$(dataset_stage1_adapter "${ds}")"
  out_root="$(dataset_out_root "${ds}")"
  if [[ "${ds}" == "hhrlhf" ]]; then
    d2="${SUB_D2_HHRLHF}"
  else
    d2="${SUB_D2_PKU}"
  fi
  out_dir="${out_root}/${method}_${ds}_eps${EPS}_s${SEED}"
  log_dir="${out_root}/logs"
  eval_dir="${out_root}/eval"
  test_jsonl="$(dataset_test_path "${ds}")"
  manifest="${out_dir}/M2_manifest.json"

  mkdir -p "${out_root}" "${log_dir}" "${eval_dir}"
  for f in "${s1}/adapter_model.safetensors" "${d2}" "${test_jsonl}"; do
    [ -f "${f}" ] || { echo "[FATAL] missing file for ${ds}/${method}: ${f}"; return 2; }
  done

  # Optional skip behavior.
  if [[ "${SKIP_EXISTING}" == "1" && -f "${manifest}" ]]; then
    if [[ "${ds}" == "hhrlhf" && "${method}" == "sb_fresh" && "${FORCE_RERUN_HHRLHF_SB}" == "1" ]]; then
      echo "[INFO] FORCE_RERUN_HHRLHF_SB=1, ignoring existing manifest for hhrlhf sb_fresh"
    else
      echo "[SKIP] ${ds}/${method} already complete: ${manifest}"
      return 0
    fi
  fi

  # For forced rerun, clear old output folder to avoid confusion.
  if [[ "${ds}" == "hhrlhf" && "${method}" == "sb_fresh" && "${FORCE_RERUN_HHRLHF_SB}" == "1" ]]; then
    rm -rf "${out_dir}"
    mkdir -p "${out_dir}"
  fi

  echo "[RUN] gpu=${gpu} ${ds}/${method} d2=${d2}"
  if [[ "${method}" == "sb_fresh" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_sb_fresh.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
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
      > "${log_dir}/train_sb_fresh_${ds}_s${SEED}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_map_retrain.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
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
      > "${log_dir}/train_map_retrain_${ds}_s${SEED}.log" 2>&1
  fi

  [ -f "${manifest}" ] || { echo "[FATAL] missing manifest after train: ${manifest}"; return 3; }

  CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/eval_preference_accuracy.py \
    --manifest "${manifest}" \
    --test_jsonl "${test_jsonl}" \
    --out_json "${eval_dir}/${method}_${ds}_eps${EPS}_s${SEED}_eval.json" \
    --max_len 512 \
    > "${log_dir}/eval_${method}_${ds}_s${SEED}.log" 2>&1

  echo "[DONE] ${ds}/${method}"
}

# Build subsets ONCE before parallel launch to avoid races.
SUB_D2_HHRLHF="$(prepare_d2_subset "hhrlhf" "$(dataset_d2 hhrlhf)" "$(dataset_out_root hhrlhf)")"
SUB_D2_PKU="$(prepare_d2_subset "pku" "$(dataset_d2 pku)" "$(dataset_out_root pku)")"
export SUB_D2_HHRLHF SUB_D2_PKU
echo "[INFO] Using subset d2 hhrlhf: ${SUB_D2_HHRLHF}"
echo "[INFO] Using subset d2 pku: ${SUB_D2_PKU}"

# Balanced queue:
# gpu0: hhrlhf/sb_fresh (rerun completion)
# gpu1: pku/sb_fresh
# gpu2: pku/map_retrain
# gpu3: hhrlhf/map_retrain (likely skipped if already complete)
PIDS=()
NAMES=()

( run_one hhrlhf sb_fresh 0 ) & PIDS+=($!); NAMES+=("hhrlhf_sb_fresh")
( run_one pku sb_fresh 1 ) & PIDS+=($!); NAMES+=("pku_sb_fresh")
( run_one pku map_retrain 2 ) & PIDS+=($!); NAMES+=("pku_map_retrain")
( run_one hhrlhf map_retrain 3 ) & PIDS+=($!); NAMES+=("hhrlhf_map_retrain")

FAILED=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  name="${NAMES[$i]}"
  if ! wait "${pid}"; then
    echo "[FATAL] task failed: ${name} (pid=${pid})"
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  exit 4
fi

export ROOT_DIR EPS SEED
python - <<'PY'
import json, os

root = os.environ["ROOT_DIR"]
eps = os.environ["EPS"]
seed = os.environ["SEED"]
for ds in ["hhrlhf", "pku"]:
    out_root = f"{root}/stage2_debugging/stage2_{ds}/results_eps{eps}_seed{seed}"
    eval_dir = f"{out_root}/eval"
    paths = {
        "SB-Fresh": f"{eval_dir}/sb_fresh_{ds}_eps{eps}_s{seed}_eval.json",
        "MAP-Retrain": f"{eval_dir}/map_retrain_{ds}_eps{eps}_s{seed}_eval.json",
    }
    summary = {}
    for name, p in paths.items():
        if os.path.exists(p):
            d = json.load(open(p, "r", encoding="utf-8"))
            summary[name] = {
                "accuracy": d.get("accuracy"),
                "ece": d.get("ece"),
                "mean_margin": d.get("mean_margin"),
                "n": d.get("n"),
                "eval_json": p,
            }
        else:
            summary[name] = {"error": f"missing: {p}"}
    md = f"{out_root}/SBFRESH_MAPRETRAIN_{ds.upper()}_SUMMARY.md"
    js = f"{out_root}/SBFRESH_MAPRETRAIN_{ds.upper()}_SUMMARY.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# {ds} SB-Fresh vs MAP-Retrain (eps={eps}, seed={seed})\n\n")
        f.write("| Method | Accuracy | ECE | Mean margin | N | Eval JSON |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for k in ["SB-Fresh", "MAP-Retrain"]:
            s = summary.get(k, {})
            if "error" in s:
                f.write(f"| {k} | ERR | ERR | ERR | ERR | {s['error']} |\n")
            else:
                f.write(
                    f"| {k} | {s['accuracy']:.4f} | {s['ece']:.4f} | "
                    f"{s['mean_margin']:.4f} | {s['n']} | {s['eval_json']} |\n"
                )
    print("Wrote", md)
    print("Wrote", js)
PY

echo "[DONE] HH+PKU SB-Fresh/MAP-Retrain job complete."

