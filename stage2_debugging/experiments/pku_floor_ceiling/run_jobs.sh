#!/bin/bash
# Run all three jobs in parallel (GPUs 0, 1, 2), then evaluate.
# Run from: /users/jzhao7/PDPO/stage2_debugging/experiments/pku_floor_ceiling/
set -euo pipefail

# Activate conda environment (required for peft, transformers, etc.)
CONDA_ENV="${CONDA_ENV:-pdpo}"
set +u  # /etc/bashrc may use unbound vars
source ~/.bashrc 2>/dev/null || true
if command -v conda > /dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
set -u  # restore strict checking
export TOKENIZERS_PARALLELISM=false

ROOT="/users/jzhao7/PDPO/stage2_debugging"
EXP="${ROOT}/experiments/pku_floor_ceiling"
LOGS="${EXP}/logs"
MODELS="${EXP}/models"
RESULTS="${EXP}/results"
mkdir -p "${LOGS}" "${MODELS}" "${RESULTS}"

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
M1="${ROOT}/stage1/results_instruct_eps1.0_seed42/pku_eps1.0_s42"
DATA_NOISY="${ROOT}/preprocessing/d2_rr_flipped_pku_eps1.0_seed42.jsonl"
DATA_ORACLE="${EXP}/d2_pku_with_flipped_eps1.0_seed42.jsonl"
TEST="${ROOT}/testsets/pku_secure/test_pref.jsonl"

COMMON_ARGS="--epsilon 1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4"

# ── Step 0: Generate oracle labels (fast, CPU) ────────────────────────────
if [ ! -f "${DATA_ORACLE}" ]; then
  echo "[Setup] Generating oracle labels..."
  python "${EXP}/gen_oracle_labels.py" \
    --data    "${DATA_NOISY}" \
    --out     "${DATA_ORACLE}" \
    --epsilon 1.0 --seed 42
  echo "[Setup] Oracle labels written to ${DATA_ORACLE}"
else
  echo "[Setup] Oracle labels already exist at ${DATA_ORACLE}"
fi

# ── Step 1: Launch Job 1 (MLE-Fresh, GPU 0) ───────────────────────────────
echo "[Launch] Job 1 — MLE-Fresh (GPU 0)"
CUDA_VISIBLE_DEVICES=0 python "${ROOT}/train_stage2_mle_fresh.py" \
  --model    "${BASE_MODEL}" \
  --data     "${DATA_NOISY}" \
  --out      "${MODELS}/pku_mle_fresh" \
  --epsilon  1.0 --beta 0.5 --lr 2.5e-5 --epochs 3 --bsz 4 --ga 4 --bf16 \
  > "${LOGS}/job1.log" 2>&1 &
JOB1_PID=$!
echo "  PID=${JOB1_PID}  log=${LOGS}/job1.log"

# ── Step 2: Launch Job 2 (Oracle SB Base Ref, GPU 1) ─────────────────────
echo "[Launch] Job 2 — Oracle SB Base Ref (GPU 1)"
CUDA_VISIBLE_DEVICES=1 python "${EXP}/train_job2_oracle_sb_baseref.py" \
  --model   "${BASE_MODEL}" \
  --data    "${DATA_ORACLE}" \
  --out     "${MODELS}/pku_oracle_sb_baseref" \
  ${COMMON_ARGS} \
  > "${LOGS}/job2.log" 2>&1 &
JOB2_PID=$!
echo "  PID=${JOB2_PID}  log=${LOGS}/job2.log"

# ── Step 3: Launch Job 3 (Oracle SB Interp Ref, GPU 2) ───────────────────
echo "[Launch] Job 3 — Oracle SB Interpolated Ref (GPU 2)"
CUDA_VISIBLE_DEVICES=2 python "${EXP}/train_job3_oracle_sb_corrref.py" \
  --model   "${BASE_MODEL}" \
  --m1      "${M1}" \
  --data    "${DATA_ORACLE}" \
  --out     "${MODELS}/pku_oracle_sb_corrref" \
  ${COMMON_ARGS} \
  > "${LOGS}/job3.log" 2>&1 &
JOB3_PID=$!
echo "  PID=${JOB3_PID}  log=${LOGS}/job3.log"

echo ""
echo "All three jobs launched. Waiting for completion..."
wait ${JOB1_PID} && echo "[Job 1] DONE" || echo "[Job 1] FAILED (exit $?)"
wait ${JOB2_PID} && echo "[Job 2] DONE" || echo "[Job 2] FAILED (exit $?)"
wait ${JOB3_PID} && echo "[Job 3] DONE" || echo "[Job 3] FAILED (exit $?)"

# ── Evaluate ──────────────────────────────────────────────────────────────
echo ""
echo "=== Evaluating ==="
for JOB in 1 2 3; do
  case $JOB in
    1) MDIR="${MODELS}/pku_mle_fresh" ;;
    2) MDIR="${MODELS}/pku_oracle_sb_baseref" ;;
    3) MDIR="${MODELS}/pku_oracle_sb_corrref" ;;
  esac
  echo "[Eval] Job ${JOB} → ${MDIR}"
  python "${ROOT}/eval_preference_accuracy.py" \
    --manifest  "${MDIR}/M2_manifest.json" \
    --test_jsonl "${TEST}" \
    --out_json   "${RESULTS}/job${JOB}.json" \
    2>&1 | tee "${LOGS}/eval_job${JOB}.log"
done

# ── Report ────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "         FINAL RESULTS"
echo "============================================"
python3 - <<'PY'
import json, glob
labels = {
    "job1": "Job 1 (MLE-Fresh,        floor):",
    "job2": "Job 2 (Oracle SB baseref, ceil):",
    "job3": "Job 3 (Oracle SB corrref, meth):",
}
for key, label in labels.items():
    path = f"results/{key}.json"
    try:
        d = json.load(open(path))
        acc = d.get("accuracy", "N/A")
        print(f"  {label}  {acc:.4f}" if isinstance(acc, float) else f"  {label}  {acc}")
    except FileNotFoundError:
        print(f"  {label}  [result not found]")
PY
echo "============================================"
