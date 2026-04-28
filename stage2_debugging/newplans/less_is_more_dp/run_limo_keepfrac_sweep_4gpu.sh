#!/bin/bash
#$ -S /bin/bash
set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-512}"
KEEP_FRACTIONS="${KEEP_FRACTIONS:-0.10 0.25 0.50 0.75 1.00}"
SCORE_ROOT="${SCORE_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/sweeps/score_cache_eps${EPS}_seed${SEED}}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/sweeps/eps${EPS}_seed${SEED}_keepfrac}"

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
  local ds="$1" frac="$2" gpu="$3"
  local frac_tag s1 d2 test cache out audit train_log eval_log eval_json
  frac_tag="${frac//./p}"
  s1="${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${ds}_eps${EPS}_s${SEED}"
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  test="$(dataset_test "${ds}")"
  cache="${SCORE_ROOT}/cache/${ds}_eps${EPS}_seed${SEED}_scores.pt"
  out="${OUT_ROOT}/${ds}/keep${frac_tag}"
  audit="${out}/selection_audit.jsonl"
  train_log="${OUT_ROOT}/logs/train_${ds}_keep${frac_tag}.log"
  eval_log="${OUT_ROOT}/logs/eval_${ds}_keep${frac_tag}.log"
  eval_json="${OUT_ROOT}/eval/${ds}_keep${frac_tag}_eval.json"

  for f in "${s1}/adapter_config.json" "${d2}" "${test}" "${cache}"; do
    [ -f "${f}" ] || { echo "[FATAL] missing file for ${ds}/keep${frac}: ${f}"; return 2; }
  done
  if [[ -f "${out}/M2_manifest.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    echo "[SKIP] ${ds} keep=${frac}: ${out}/M2_manifest.json"
  else
    mkdir -p "${out}"
    echo "[RUN] ds=${ds} keep_fraction=${frac} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -u \
      "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py" \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
      --epsilon "${EPS}" \
      --keep_fraction "${frac}" \
      --selection_mode both \
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
  for f in ${KEEP_FRACTIONS}; do train_eval truthy "$f" 0; done
}
queue_gpu1() {
  train_eval hhrlhf 0.10 1
  train_eval hhrlhf 0.25 1
  train_eval hhrlhf 0.50 1
}
queue_gpu2() {
  train_eval hhrlhf 0.75 2
  train_eval hhrlhf 1.00 2
  train_eval pku 0.10 2
}
queue_gpu3() {
  train_eval pku 0.25 3
  train_eval pku 0.50 3
  train_eval pku 0.75 3
  train_eval pku 1.00 3
}

( queue_gpu0 ) > "${OUT_ROOT}/logs/gpu0.queue.log" 2>&1 & P0=$!
( queue_gpu1 ) > "${OUT_ROOT}/logs/gpu1.queue.log" 2>&1 & P1=$!
( queue_gpu2 ) > "${OUT_ROOT}/logs/gpu2.queue.log" 2>&1 & P2=$!
( queue_gpu3 ) > "${OUT_ROOT}/logs/gpu3.queue.log" 2>&1 & P3=$!
wait "${P0}" "${P1}" "${P2}" "${P3}"

python "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/aggregate_limo_sweeps.py" --root "${OUT_ROOT}"
echo "[DONE] keep-fraction sweep complete: ${OUT_ROOT}"
