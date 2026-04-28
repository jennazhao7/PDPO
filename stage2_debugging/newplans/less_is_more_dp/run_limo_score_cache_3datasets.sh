#!/bin/bash
#$ -S /bin/bash
set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-512}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/sweeps/score_cache_eps${EPS}_seed${SEED}}"

set +u
source ~/.bashrc
set -u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
cd "${ROOT_DIR}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/cache"

score_one() {
  local ds="$1" gpu="$2"
  local s1 d2 cache log
  s1="${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${ds}_eps${EPS}_s${SEED}"
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  cache="${OUT_ROOT}/cache/${ds}_eps${EPS}_seed${SEED}_scores.pt"
  log="${OUT_ROOT}/logs/score_${ds}_eps${EPS}_seed${SEED}.log"

  for f in "${s1}/adapter_config.json" "${d2}"; do
    [ -f "${f}" ] || { echo "[FATAL] missing file for ${ds}: ${f}"; return 2; }
  done

  if [[ -f "${cache}" && "${FORCE_RESCORE:-0}" == "1" ]]; then
    rm -f "${cache}"
  fi
  if [[ -f "${cache}" ]]; then
    echo "[SKIP] ${ds} score cache exists: ${cache}" | tee -a "${log}"
    return 0
  fi

  echo "[RUN] score cache ds=${ds} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u \
    "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py" \
    --model "${BASE_MODEL}" \
    --stage1_adapter "${s1}" \
    --data "${d2}" \
    --epsilon "${EPS}" \
    --max_len "${MAX_LEN}" \
    --seed "${SEED}" \
    --bf16 \
    --score_cache_out "${cache}" \
    --score_only \
    > "${log}" 2>&1
}

( score_one truthy 0 ) & P1=$!
( score_one hhrlhf 1 ) & P2=$!
( score_one pku 2 ) & P3=$!
wait "${P1}" "${P2}" "${P3}"

echo "[DONE] score caches written under ${OUT_ROOT}/cache"
