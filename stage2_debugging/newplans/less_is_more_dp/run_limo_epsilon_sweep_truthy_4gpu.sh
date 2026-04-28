#!/bin/bash
#$ -S /bin/bash
set -eo pipefail

ROOT_DIR="${ROOT_DIR:-/users/jzhao7/PDPO}"
CONDA_ENV="${CONDA_ENV:-pdpo}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
SEED="${SEED:-42}"
MAX_LEN="${MAX_LEN:-512}"
KEEP_FRACTION="${KEEP_FRACTION:-0.25}"
EPS_LIST="${EPS_LIST:-0.3 0.5 1.0 2.0}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/sweeps/epsilon_robustness/truthy_keep${KEEP_FRACTION//./p}_seed${SEED}}"

set +u
source ~/.bashrc
set -u
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
cd "${ROOT_DIR}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/eval" "${OUT_ROOT}/cache"

run_eps() {
  local eps="$1" gpu="$2"
  local eps_tag s1 d2 test cache out audit train_log score_log eval_log eval_json
  eps_tag="${eps//./p}"
  s1="${ROOT_DIR}/stage2_debugging/stage1/results_eps${eps}_seed${SEED}/truthy_eps${eps}_s${SEED}"
  d2="${ROOT_DIR}/stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps${eps}_seed${SEED}.jsonl"
  test="${ROOT_DIR}/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"
  cache="${OUT_ROOT}/cache/truthy_eps${eps}_seed${SEED}_scores.pt"
  out="${OUT_ROOT}/truthy/eps${eps_tag}_keep${KEEP_FRACTION//./p}"
  audit="${out}/selection_audit.jsonl"
  score_log="${OUT_ROOT}/logs/score_truthy_eps${eps}.log"
  train_log="${OUT_ROOT}/logs/train_truthy_eps${eps}.log"
  eval_log="${OUT_ROOT}/logs/eval_truthy_eps${eps}.log"
  eval_json="${OUT_ROOT}/eval/truthy_eps${eps_tag}_keep${KEEP_FRACTION//./p}_eval.json"

  for f in "${s1}/adapter_config.json" "${d2}" "${test}"; do
    [ -f "${f}" ] || { echo "[FATAL] missing file for truthy eps=${eps}: ${f}"; return 2; }
  done

  if [[ ! -f "${cache}" || "${FORCE_RESCORE:-0}" == "1" ]]; then
    echo "[RUN] score truthy eps=${eps} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -u \
      "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py" \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
      --epsilon "${eps}" \
      --max_len "${MAX_LEN}" \
      --seed "${SEED}" \
      --bf16 \
      --score_cache_out "${cache}" \
      --score_only \
      > "${score_log}" 2>&1
  fi

  if [[ -f "${out}/M2_manifest.json" && "${SKIP_EXISTING:-1}" == "1" ]]; then
    echo "[SKIP] truthy eps=${eps}: ${out}/M2_manifest.json"
  else
    mkdir -p "${out}"
    echo "[RUN] train truthy eps=${eps} keep_fraction=${KEEP_FRACTION} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -u \
      "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/train_stage2_sb_select_weight_fresh.py" \
      --model "${BASE_MODEL}" \
      --stage1_adapter "${s1}" \
      --data "${d2}" \
      --epsilon "${eps}" \
      --keep_fraction "${KEEP_FRACTION}" \
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

run_eps 0.3 0 > "${OUT_ROOT}/logs/gpu0_eps0.3.queue.log" 2>&1 & P0=$!
run_eps 0.5 1 > "${OUT_ROOT}/logs/gpu1_eps0.5.queue.log" 2>&1 & P1=$!
run_eps 1.0 2 > "${OUT_ROOT}/logs/gpu2_eps1.0.queue.log" 2>&1 & P2=$!
run_eps 2.0 3 > "${OUT_ROOT}/logs/gpu3_eps2.0.queue.log" 2>&1 & P3=$!
wait "${P0}" "${P1}" "${P2}" "${P3}"

python "${ROOT_DIR}/stage2_debugging/newplans/less_is_more_dp/aggregate_limo_sweeps.py" --root "${OUT_ROOT}"
echo "[DONE] truthy epsilon sweep complete: ${OUT_ROOT}"
