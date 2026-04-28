#!/bin/bash
#$ -S /bin/bash
set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-512}"
KEEP_FRACTION="${KEEP_FRACTION:-0.25}"
SCORE_ROOT="${SCORE_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/sweeps/score_cache_eps${EPS}_seed${SEED}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/sweeps/eps${EPS}_seed${SEED}_ablation_keep${KEEP_FRACTION//./p}}"

set +u
source ~/.bashrc
set -u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
cd "${ROOT_DIR}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/eval"

dataset_test() {
  case "$1" in
    truthy) echo "${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl" ;;
    hhrlhf) echo "${ROOT_DIR}/stage2_debugging/test_pref.jsonl" ;;
    pku) echo "${ROOT_DIR}/stage2_debugging/testsets/pku_secure/test_pref.jsonl" ;;
    *) return 1 ;;
  esac
}

train_eval() {
  local ds="$1" mode="$2" gpu="$3"
  local keep_arg frac_tag s1 d2 test cache out audit train_log eval_log eval_json
  frac_tag="${KEEP_FRACTION//./p}"
  s1="${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${ds}_eps${EPS}_s${SEED}"
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  test="$(dataset_test "${ds}")"
  cache="${SCORE_ROOT}/cache/${ds}_eps${EPS}_seed${SEED}_scores.pt"
  out="${OUT_ROOT}/${ds}/${mode}"
  audit="${out}/selection_audit.jsonl"
  train_log="${OUT_ROOT}/logs/train_${ds}_${mode}.log"
  eval_log="${OUT_ROOT}/logs/eval_${ds}_${mode}.log"
  eval_json="${OUT_ROOT}/eval/${ds}_${mode}_eval.json"

  for f in "${s1}/adapter_config.json" "${d2}" "${test}" "${cache}"; do
    [ -f "${f}" ] || { echo "[FATAL] missing file for ${ds}/${mode}: ${f}"; return 2; }
  done

  keep_arg=(--keep_fraction "${KEEP_FRACTION}")
  if [[ "${mode}" == "weighting_only" ]]; then
    keep_arg=()
  fi

  if [[ -f "${out}/M2_manifest.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    echo "[SKIP] ${ds}/${mode}: ${out}/M2_manifest.json"
  else
    mkdir -p "${out}"
    echo "[RUN] ds=${ds} mode=${mode} keep_fraction=${KEEP_FRACTION} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -u \
      "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py" \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
      --epsilon "${EPS}" \
      "${keep_arg[@]}" \
      --selection_mode "${mode}" \
      --score_cache_in "${cache}" \
      --save_selection_jsonl "${audit}" \
      --out "${out}" \
      --max_len "${MAX_LEN}" \
      --seed "${SEED}" \
      --epochs 3 \
      --bsz 1 \
      --ga 16 \
      --lr 2.5e-5 \
      --beta 0.5 \
      --bf16 \
      > "${train_log}" 2>&1
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" python -u "${ROOT_DIR}/stage2_debugging/eval_preference_accuracy.py" \
    --manifest "${out}/M2_manifest.json" \
    --test_jsonl "${test}" \
    --out_json "${eval_json}" \
    --max_len "${MAX_LEN}" \
    > "${eval_log}" 2>&1
}

queue_gpu0() {
  train_eval truthy both 0
  train_eval truthy selection_only 0
  train_eval truthy weighting_only 0
}
queue_gpu1() {
  train_eval hhrlhf both 1
  train_eval hhrlhf selection_only 1
}
queue_gpu2() {
  train_eval hhrlhf weighting_only 2
  train_eval pku both 2
}
queue_gpu3() {
  train_eval pku selection_only 3
  train_eval pku weighting_only 3
}

( queue_gpu0 ) > "${OUT_ROOT}/logs/gpu0.queue.log" 2>&1 & P0=$!
( queue_gpu1 ) > "${OUT_ROOT}/logs/gpu1.queue.log" 2>&1 & P1=$!
( queue_gpu2 ) > "${OUT_ROOT}/logs/gpu2.queue.log" 2>&1 & P2=$!
( queue_gpu3 ) > "${OUT_ROOT}/logs/gpu3.queue.log" 2>&1 & P3=$!
wait "${P0}" "${P1}" "${P2}" "${P3}"

python "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/aggregate_limo_sweeps.py" --root "${OUT_ROOT}"
echo "[DONE] ablation sweep complete: ${OUT_ROOT}"
