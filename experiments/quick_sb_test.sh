#!/usr/bin/env bash
# quick_sb_test.sh — Minimal end-to-end verification that SB Stage 2 beats Stage 1.
#
# Runs on ONE GPU, sequentially:
#   1. Eval Stage 1 baseline accuracy
#   2. Train SB Stage 2 (50 steps, ~10 min)
#   3. Eval SB Stage 2 accuracy
#   4. Train MLE-DPO Stage 2 (50 steps, ~10 min)
#   5. Eval MLE-DPO Stage 2 accuracy
#   6. Print comparison table
#
# Usage:  GPU=0 bash experiments/quick_sb_test.sh
#
# SUCCESS if:  SB_acc > Stage1_acc  AND  SB_acc > MLE_acc
set -euo pipefail

GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=$GPU

MODEL="Qwen/Qwen2.5-3B"
SEED=42
EPSILON=1.0
MAX_STEPS=50          # just enough to see signal (full run = 300)
BSZ=1
GA=16
LR=1e-5
MAX_LEN=256
TEST_JSONL="data/pku_saferlhf_secure/test_pref.jsonl"

# Paths
OUT_ROOT="outputs_quick_test"
S1_ADAPTER="outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed${SEED}"
D2_DATA="outputs_new_models/preprocessing/d2_rr_flipped_eps${EPSILON}_seed${SEED}.jsonl"
SB_OUT="${OUT_ROOT}/sb_fixed_seed${SEED}"
MLE_OUT="${OUT_ROOT}/mle_dpo_seed${SEED}"

mkdir -p "${OUT_ROOT}/logs"

echo "=============================================="
echo " Quick SB Verification Test (GPU=${GPU})"
echo "=============================================="
echo " Model:  ${MODEL}"
echo " Stage1: ${S1_ADAPTER}"
echo " D2:     ${D2_DATA}"
echo " Steps:  ${MAX_STEPS}"
echo "=============================================="

# ---- Step 1: Evaluate Stage 1 baseline ----
echo ""
echo "[Step 1/6] Evaluating Stage 1 baseline..."
python -u eval/eval_stage1_accuracy.py \
  --base_model "${MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --test_jsonl "${TEST_JSONL}" \
  --max_len ${MAX_LEN} \
  --out_json "${OUT_ROOT}/eval_stage1.json" 2>&1 | tail -5

S1_ACC=$(python3 -c "import json; print(json.load(open('${OUT_ROOT}/eval_stage1.json'))['accuracy'])")
echo ">>> Stage 1 accuracy: ${S1_ACC}"

# ---- Step 2: Train SB Stage 2 (reference-model-fixed) ----
echo ""
echo "[Step 2/6] Training Soft Bayes Stage 2 (${MAX_STEPS} steps)..."
python -u lora/train_stage2_soft_bayes.py \
  --model "${MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${SB_OUT}" \
  --epsilon ${EPSILON} --tau 0.0 --beta 0.1 --delta_clamp 500.0 \
  --lr ${LR} --max_steps ${MAX_STEPS} --bsz ${BSZ} --ga ${GA} \
  --max_len ${MAX_LEN} --bf16 --seed ${SEED} 2>&1 | tail -20

# ---- Step 3: Evaluate SB Stage 2 ----
echo ""
echo "[Step 3/6] Evaluating Soft Bayes Stage 2..."
python -u eval/eval_preference_accuracy.py \
  --manifest "${SB_OUT}/M2_manifest.json" \
  --test_jsonl "${TEST_JSONL}" \
  --max_len ${MAX_LEN} \
  --out_json "${OUT_ROOT}/eval_sb.json" 2>&1 | tail -5

SB_ACC=$(python3 -c "import json; print(json.load(open('${OUT_ROOT}/eval_sb.json'))['accuracy'])")
echo ">>> SB Stage 2 accuracy: ${SB_ACC}"

# ---- Step 4: Train MLE-DPO Stage 2 ----
echo ""
echo "[Step 4/6] Training MLE-DPO Stage 2 (${MAX_STEPS} steps)..."
python -u lora/train_stage2_dpo.py \
  --model "${MODEL}" \
  --stage1_adapter "${S1_ADAPTER}" \
  --data "${D2_DATA}" \
  --out "${MLE_OUT}" \
  --beta 0.1 --lr ${LR} --max_steps ${MAX_STEPS} --bsz ${BSZ} --ga ${GA} \
  --bf16 --seed ${SEED} 2>&1 | tail -20

# ---- Step 5: Evaluate MLE-DPO Stage 2 ----
echo ""
echo "[Step 5/6] Evaluating MLE-DPO Stage 2..."
python -u eval/eval_preference_accuracy.py \
  --manifest "${MLE_OUT}/M2_manifest.json" \
  --test_jsonl "${TEST_JSONL}" \
  --max_len ${MAX_LEN} \
  --out_json "${OUT_ROOT}/eval_mle.json" 2>&1 | tail -5

MLE_ACC=$(python3 -c "import json; print(json.load(open('${OUT_ROOT}/eval_mle.json'))['accuracy'])")
echo ">>> MLE Stage 2 accuracy: ${MLE_ACC}"

# ---- Step 6: Print results ----
echo ""
echo "=============================================="
echo "           QUICK TEST RESULTS"
echo "=============================================="
printf "  %-25s %s\n" "Stage 1 (M1 on noisy D1):" "${S1_ACC}"
printf "  %-25s %s\n" "MLE-DPO (M2 on noisy D2):" "${MLE_ACC}"
printf "  %-25s %s\n" "Soft Bayes (M2 on D2):"    "${SB_ACC}"
echo "=============================================="
echo ""

python3 -c "
s1 = ${S1_ACC}
mle = ${MLE_ACC}
sb = ${SB_ACC}
print('VERDICT:')
if sb > s1 and sb > mle:
    print('  ✅ SUCCESS: SB > Stage1 AND SB > MLE')
    print('  Soft Bayes signal recovery is WORKING!')
elif sb > mle:
    print('  ⚠️  PARTIAL: SB > MLE but SB <= Stage1')
    print('  SB helps vs MLE but does not recover beyond Stage1.')
else:
    print('  ❌ FAIL: SB did not beat MLE or Stage1')
    print('  Something is still broken.')
"
