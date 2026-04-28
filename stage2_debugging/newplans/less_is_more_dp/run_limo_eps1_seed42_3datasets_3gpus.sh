#!/bin/bash
#$ -S /bin/bash
set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
TAU_DROP="${TAU_DROP:-0.10}"
MODE="${MODE:-both}"  # both | weighting_only | selection_only
MAX_LEN="${MAX_LEN:-512}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/results_eps${EPS}_seed${SEED}}"

# Some cluster bashrc setups reference unset vars; source safely, then re-enable nounset.
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

run_one() {
  local ds="$1" gpu="$2"
  local s1 d2 out test disable_weight
  s1="${ROOT_DIR}/stage2_debugging/stage1/results_eps${EPS}_seed${SEED}/${ds}_eps${EPS}_s${SEED}"
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_${ds}_eps${EPS}_seed${SEED}.jsonl"
  test="$(dataset_test "${ds}")"
  out="${OUT_ROOT}/${ds}/${MODE}_tauDrop${TAU_DROP}"
  mkdir -p "${out}"

  disable_weight=""
  if [[ "${MODE}" == "selection_only" ]]; then
    disable_weight="--disable_weighting"
  fi

  local tau_drop_arg="${TAU_DROP}"
  if [[ "${MODE}" == "weighting_only" ]]; then
    tau_drop_arg="0.0"
  fi

  for f in "${s1}/adapter_config.json" "${d2}" "${test}"; do
    [ -f "${f}" ] || { echo "[FATAL] missing file: ${f}"; return 2; }
  done

  echo "[RUN] ds=${ds} gpu=${gpu} mode=${MODE} tau_drop=${tau_drop_arg}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u \
    "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py" \
    --model "${BASE_MODEL}" \
    --stage1_adapter "${s1}" \
    --data "${d2}" \
    --epsilon "${EPS}" \
    --tau_drop "${tau_drop_arg}" \
    --out "${out}" \
    --max_len "${MAX_LEN}" \
    --seed "${SEED}" \
    --epochs 3 \
    --bsz 1 \
    --ga 16 \
    --lr 2.5e-5 \
    --beta 0.5 \
    --bf16 \
    ${disable_weight} \
    > "${OUT_ROOT}/logs/train_${ds}_${MODE}_tauDrop${tau_drop_arg}.log" 2>&1

  CUDA_VISIBLE_DEVICES="${gpu}" python -u "${ROOT_DIR}/stage2_debugging/eval_preference_accuracy.py" \
    --manifest "${out}/M2_manifest.json" \
    --test_jsonl "${test}" \
    --out_json "${OUT_ROOT}/eval/${ds}_${MODE}_tauDrop${tau_drop_arg}_eval.json" \
    --max_len "${MAX_LEN}" \
    > "${OUT_ROOT}/logs/eval_${ds}_${MODE}_tauDrop${tau_drop_arg}.log" 2>&1
}

( run_one truthy 0 ) & P1=$!
( run_one hhrlhf 1 ) & P2=$!
( run_one pku 2 ) & P3=$!
wait "${P1}" "${P2}" "${P3}"

echo "[DONE] Less-is-More run complete at ${OUT_ROOT}"

