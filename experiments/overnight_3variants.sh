#!/usr/bin/env bash
# overnight_3variants.sh — Tests all 3 SB variants + baselines on 4 GPUs.
#
# GPU 0: Option A (PROPS MAP + DPOTrainer) → eval
# GPU 1: Option B (Soft Bayes v2, β=0.5, lr=2.5e-5) → eval
# GPU 2: Option C (MAP retrain, single fresh LoRA) → eval
# GPU 3: Baseline re-evals (S1, MLE, Clean DPO)
#
# Usage (from qrsh GPU session):
#   bash experiments/overnight_3variants.sh
#
# Monitor:
#   tail -f outputs_overnight/logs/*.txt
#   ls outputs_overnight/results/

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs_overnight"
LOGS="$OUT/logs"
RES="$OUT/results"
TEST="data/pku_saferlhf_secure/test_pref.jsonl"

# Existing adapters
S1_42="outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed42"
S1_43="outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed43"
MLE_42="outputs_new_models/stage2_mle_dpo/Qwen--Qwen2.5-3B_stage2_mle_dpo_eps1.0_seed42"
MLE_43="outputs_new_models/stage2_mle_dpo/Qwen--Qwen2.5-3B_stage2_mle_dpo_eps1.0_seed43"
CLEAN="outputs_quick_test/stage1_clean_dpo_qwen_seed42"
D2_42="outputs_new_models/preprocessing/d2_rr_flipped_eps1.0_seed42.jsonl"

mkdir -p "$LOGS" "$RES"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
run() {
    local gpu=$1 name=$2; shift 2
    local log="$LOGS/${name}.txt"
    echo "[$(ts)] GPU $gpu START: $name" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu "$@" >> "$log" 2>&1
    local rc=$?
    echo "[$(ts)] GPU $gpu $([ $rc -eq 0 ] && echo '✅' || echo '❌') $name (rc=$rc)" | tee -a "$log"
    return $rc
}

# ═══════════════════════════════════════════════
# GPU 0: Option A — Faithful PROPS MAP
# ═══════════════════════════════════════════════
gpu0() {
    echo "[$(ts)] GPU 0: Option A (PROPS MAP)"

    # Train
    run 0 "train_optA_qwen42" python -u lora/train_stage2_props_map.py \
        --model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_42" \
        --data "$D2_42" \
        --out "$OUT/optA_props_map_qwen_s42" \
        --epsilon 1.0 --beta 0.1 --lr 5e-5 --epochs 3 \
        --bsz 1 --ga 16 --max_len 512 --bf16 --seed 42

    # Eval
    if [ -f "$OUT/optA_props_map_qwen_s42/M2_manifest.json" ]; then
        run 0 "eval_optA_qwen42" python -u eval/eval_preference_accuracy.py \
            --manifest "$OUT/optA_props_map_qwen_s42/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RES/optA_props_map_qwen_s42.json"
    fi

    echo "[$(ts)] GPU 0 DONE"
}

# ═══════════════════════════════════════════════
# GPU 1: Option B — Calibrated Soft Bayes
# ═══════════════════════════════════════════════
gpu1() {
    echo "[$(ts)] GPU 1: Option B (Soft Bayes v2)"

    # Train with FIXED hyperparams: beta=0.5 (was 0.1), lr=2.5e-5 (was 1e-5)
    run 1 "train_optB_qwen42" python -u lora/train_stage2_soft_bayes.py \
        --model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_42" \
        --data "$D2_42" \
        --out "$OUT/optB_soft_bayes_v2_qwen_s42" \
        --epsilon 1.0 --tau 0.0 --beta 0.5 --delta_clamp 500.0 \
        --lr 2.5e-5 --epochs 3 --bsz 1 --ga 16 --max_len 512 --bf16 --seed 42

    # Eval
    if [ -f "$OUT/optB_soft_bayes_v2_qwen_s42/M2_manifest.json" ]; then
        run 1 "eval_optB_qwen42" python -u eval/eval_preference_accuracy.py \
            --manifest "$OUT/optB_soft_bayes_v2_qwen_s42/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RES/optB_soft_bayes_v2_qwen_s42.json"
    fi

    echo "[$(ts)] GPU 1 DONE"
}

# ═══════════════════════════════════════════════
# GPU 2: Option C — MAP Retrain (Single LoRA)
# ═══════════════════════════════════════════════
gpu2() {
    echo "[$(ts)] GPU 2: Option C (MAP Retrain)"

    # Train
    run 2 "train_optC_qwen42" python -u lora/train_stage2_map_retrain.py \
        --model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_42" \
        --data "$D2_42" \
        --out "$OUT/optC_map_retrain_qwen_s42" \
        --epsilon 1.0 --beta 0.1 --lr 5e-5 --epochs 3 \
        --bsz 1 --ga 16 --max_len 512 --bf16 --seed 42

    # Eval (uses eval_stage1_accuracy since it's a single-adapter model)
    if [ -d "$OUT/optC_map_retrain_qwen_s42/fresh_lora" ]; then
        run 2 "eval_optC_qwen42" python -u eval/eval_stage1_accuracy.py \
            --base_model Qwen/Qwen2.5-3B \
            --stage1_adapter "$OUT/optC_map_retrain_qwen_s42/fresh_lora" \
            --test_jsonl "$TEST" \
            --out_json "$RES/optC_map_retrain_qwen_s42.json"
    fi

    echo "[$(ts)] GPU 2 DONE"
}

# ═══════════════════════════════════════════════
# GPU 3: Baseline re-evals (all with new DPO reward metric)
# ═══════════════════════════════════════════════
gpu3() {
    echo "[$(ts)] GPU 3: Baseline re-evals"

    # Stage 1 Qwen s42
    run 3 "eval_s1_qwen42" python -u eval/eval_stage1_accuracy.py \
        --base_model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_42" \
        --test_jsonl "$TEST" \
        --out_json "$RES/baseline_s1_qwen_s42.json"

    # Stage 1 Qwen s43
    run 3 "eval_s1_qwen43" python -u eval/eval_stage1_accuracy.py \
        --base_model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_43" \
        --test_jsonl "$TEST" \
        --out_json "$RES/baseline_s1_qwen_s43.json"

    # MLE-DPO s42
    if [ -f "$MLE_42/M2_manifest.json" ]; then
        run 3 "eval_mle_qwen42" python -u eval/eval_preference_accuracy.py \
            --manifest "$MLE_42/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RES/baseline_mle_qwen_s42.json"
    fi

    # MLE-DPO s43
    if [ -f "$MLE_43/M2_manifest.json" ]; then
        run 3 "eval_mle_qwen43" python -u eval/eval_preference_accuracy.py \
            --manifest "$MLE_43/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RES/baseline_mle_qwen_s43.json"
    fi

    # Clean DPO upper bound
    if [ -d "$CLEAN" ]; then
        run 3 "eval_clean_dpo" python -u eval/eval_stage1_accuracy.py \
            --base_model Qwen/Qwen2.5-3B \
            --stage1_adapter "$CLEAN" \
            --test_jsonl "$TEST" \
            --out_json "$RES/baseline_clean_dpo_s42.json"
    fi

    echo "[$(ts)] GPU 3 DONE"
}

# ═══════════════════════════════════════════════
# LAUNCH
# ═══════════════════════════════════════════════
echo "═══════════════════════════════════════════════"
echo " 3-VARIANT OVERNIGHT EXPERIMENT"
echo " Started: $(ts)"
echo " Output:  $OUT/"
echo "═══════════════════════════════════════════════"
echo " GPU 0: Option A — PROPS MAP + DPOTrainer"
echo " GPU 1: Option B — Soft Bayes v2 (β=0.5, lr=2.5e-5)"
echo " GPU 2: Option C — MAP Retrain (single fresh LoRA)"
echo " GPU 3: Baseline re-evals (S1, MLE, Clean DPO)"
echo "═══════════════════════════════════════════════"
echo ""

gpu0 &
gpu1 &
gpu2 &
gpu3 &

echo "All 4 GPU workers launched."
echo "Monitor: tail -f $LOGS/*.txt"
echo "Results: ls $RES/"
echo ""

wait

echo ""
echo "═══════════════════════════════════════════════"
echo " ALL DONE: $(ts)"
echo "═══════════════════════════════════════════════"
echo ""
echo "RESULTS SUMMARY:"
for f in "$RES"/*.json; do
    [ -f "$f" ] || continue
    acc=$(python3 -c "import json; d=json.load(open('$f')); print(f'{d.get(\"accuracy\",\"N/A\"):.3f}')" 2>/dev/null || echo "ERR")
    printf "  %-42s accuracy=%s\n" "$(basename "$f" .json)" "$acc"
done
