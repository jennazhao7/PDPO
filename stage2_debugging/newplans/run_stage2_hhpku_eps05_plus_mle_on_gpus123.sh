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
SKIP_EXISTING="${SKIP_EXISTING:-1}"
D2_FRACTION="${D2_FRACTION:-0.5}"
SUBSET_SEED="${SUBSET_SEED:-42}"

OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/stage1_followup_hhpku_seed${SEED}}"
LOG_DIR="${OUT_ROOT}/logs"
EVAL_DIR="${OUT_ROOT}/eval"
mkdir -p "${OUT_ROOT}" "${LOG_DIR}" "${EVAL_DIR}"

S1_HHRLHF="${S1_HHRLHF:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${S1_EPS}_seed${SEED}/hhrlhf_eps${S1_EPS}_s${SEED}}"
S1_PKU="${S1_PKU:-${ROOT_DIR}/stage2_debugging/stage1/results_eps${S1_EPS}_seed${SEED}/pku_eps${S1_EPS}_s${SEED}}"

D2_HHRLHF_E05="${D2_HHRLHF_E05:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_hhrlhf_eps0.5_seed${SEED}.jsonl}"
D2_PKU_E05="${D2_PKU_E05:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_pku_eps0.5_seed${SEED}.jsonl}"
D2_HHRLHF_E10="${D2_HHRLHF_E10:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_hhrlhf_eps1.0_seed${SEED}.jsonl}"
D2_PKU_E10="${D2_PKU_E10:-${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_pku_eps1.0_seed${SEED}.jsonl}"

TEST_HHRLHF="${TEST_HHRLHF:-${ROOT_DIR}/stage2_debugging/test_pref.jsonl}"
TEST_PKU="${TEST_PKU:-${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl}"

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
assert torch.cuda.device_count() >= 2, "Need at least 2 visible GPUs"
PY

nvidia-smi || true

prepare_d2_subset() {
  local ds="$1" eps="$2" full_d2="$3"
  local data_dir subset_d2
  data_dir="${OUT_ROOT}/data"
  mkdir -p "${data_dir}"
  subset_d2="${data_dir}/d2_rr_flipped_${ds}_eps${eps}_seed${SEED}_frac${D2_FRACTION}_subseed${SUBSET_SEED}.jsonl"

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

for f in \
  "${S1_HHRLHF}/adapter_model.safetensors" \
  "${S1_PKU}/adapter_model.safetensors" \
  "${D2_HHRLHF_E05}" "${D2_PKU_E05}" "${D2_HHRLHF_E10}" "${D2_PKU_E10}" \
  "${TEST_HHRLHF}" "${TEST_PKU}"
do
  [ -f "${f}" ] || {
    echo "[FATAL] Missing required file: ${f}"
    echo "        If missing eps=0.5 D2 for HH/PKU, generate those first, then rerun."
    exit 2
  }
done

SUB_D2_HHRLHF_E05="$(prepare_d2_subset hhrlhf 0.5 "${D2_HHRLHF_E05}")"
SUB_D2_PKU_E05="$(prepare_d2_subset pku 0.5 "${D2_PKU_E05}")"
SUB_D2_HHRLHF_E10="$(prepare_d2_subset hhrlhf 1.0 "${D2_HHRLHF_E10}")"
SUB_D2_PKU_E10="$(prepare_d2_subset pku 1.0 "${D2_PKU_E10}")"
echo "[INFO] Using subset D2 hhrlhf eps0.5: ${SUB_D2_HHRLHF_E05}"
echo "[INFO] Using subset D2 pku eps0.5: ${SUB_D2_PKU_E05}"
echo "[INFO] Using subset D2 hhrlhf eps1.0: ${SUB_D2_HHRLHF_E10}"
echo "[INFO] Using subset D2 pku eps1.0: ${SUB_D2_PKU_E10}"

train_and_eval() {
  local method="$1" ds="$2" eps="$3" gpu="$4"
  local s1 d2 test_jsonl out_dir manifest train_log eval_log eval_json

  if [[ "${ds}" == "hhrlhf" ]]; then
    s1="${S1_HHRLHF}"
    test_jsonl="${TEST_HHRLHF}"
    if [[ "${eps}" == "0.5" ]]; then d2="${SUB_D2_HHRLHF_E05}"; else d2="${SUB_D2_HHRLHF_E10}"; fi
  else
    s1="${S1_PKU}"
    test_jsonl="${TEST_PKU}"
    if [[ "${eps}" == "0.5" ]]; then d2="${SUB_D2_PKU_E05}"; else d2="${SUB_D2_PKU_E10}"; fi
  fi

  out_dir="${OUT_ROOT}/${method}_${ds}_eps${eps}_s${SEED}"
  manifest="${out_dir}/M2_manifest.json"
  train_log="${LOG_DIR}/train_${method}_${ds}_eps${eps}_s${SEED}.log"
  eval_log="${LOG_DIR}/eval_${method}_${ds}_eps${eps}_s${SEED}.log"
  eval_json="${EVAL_DIR}/${method}_${ds}_eps${eps}_s${SEED}_eval.json"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${manifest}" ]]; then
    echo "[SKIP] ${method}/${ds}/eps${eps}: ${manifest}" | tee -a "${train_log}"
    return 0
  fi

  echo "[RUN] gpu=${gpu} ${method}/${ds}/eps${eps}" | tee -a "${train_log}"
  mkdir -p "${out_dir}"

  if [[ "${method}" == "sb_fresh" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_sb_fresh.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
      --out "${out_dir}" \
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
      >> "${train_log}" 2>&1
  elif [[ "${method}" == "map_retrain" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_map_retrain.py \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
      --out "${out_dir}" \
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
      >> "${train_log}" 2>&1
  else
    CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/train_stage2_mle_fresh.py \
      --model "${BASE_MODEL}" \
      --data "${d2}" \
      --out "${out_dir}" \
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
      >> "${train_log}" 2>&1
  fi

  [ -f "${manifest}" ] || { echo "[FATAL] missing manifest: ${manifest}"; return 3; }

  CUDA_VISIBLE_DEVICES="${gpu}" python -u stage2_debugging/eval_preference_accuracy.py \
    --manifest "${manifest}" \
    --test_jsonl "${test_jsonl}" \
    --out_json "${eval_json}" \
    --max_len 512 \
    >> "${eval_log}" 2>&1

  echo "[DONE] ${method}/${ds}/eps${eps}"
}

# GPU 0 queue: HH-RLHF eps0.5 SB/MAP + MLE eps0.5
(
  train_and_eval sb_fresh hhrlhf 0.5 0
  train_and_eval map_retrain hhrlhf 0.5 0
  train_and_eval mle_fresh hhrlhf 0.5 0
) &
PID1=$!

# GPU 1 queue: PKU eps0.5 SB/MAP + MLE eps0.5
(
  train_and_eval sb_fresh pku 0.5 1
  train_and_eval map_retrain pku 0.5 1
  train_and_eval mle_fresh pku 0.5 1
) &
PID2=$!

# GPU 1 continuation queue: MLE eps1.0 for both datasets
(
  train_and_eval mle_fresh hhrlhf 1.0 1
  train_and_eval mle_fresh pku 1.0 1
) &
PID3=$!

FAILED=0
for p in "${PID1}" "${PID2}" "${PID3}"; do
  if ! wait "${p}"; then
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "[FATAL] one or more GPU queues failed"
  exit 1
fi

echo "[DONE] stage1 plan follow-up job complete: ${OUT_ROOT}"
