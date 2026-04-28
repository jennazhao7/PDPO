#!/usr/bin/env bash
# overnight_jobs.sh — Fully utilizes 4 GPUs for ~9 hours to recover from all bugs.
#
# Architecture: 4 independent per-GPU worker scripts. Each chains tasks sequentially.
# No cross-GPU dependencies. Each task logs to its own file.
# If a task fails, the GPU moves to the next task (set +e per task).
#
# Usage (from qrsh GPU session):
#   bash experiments/overnight_jobs.sh
#
# To monitor:
#   tail -f outputs_overnight/logs/*.txt
#   ls -la outputs_overnight/results/
#
# Structure:
#   GPU 0: verify_sb → retrain SB Qwen s42 → eval SB Qwen s42
#   GPU 1: eval S1 Qwen s42+s43 → retrain SB Qwen s43 → eval SB Qwen s43
#   GPU 2: eval MLE Qwen s42+s43 → eval clean DPO → retrain SB Mistral s42 → eval
#   GPU 3: eval S1 Mistral s42+s43 → retrain SB Mistral s43 → eval

set -u  # fail on undefined vars, but NOT on command errors (we handle those per-task)

# ─── PATHS ──────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT="outputs_overnight"
LOGS="$OUT/logs"
RESULTS="$OUT/results"
SB_MODELS="$OUT/sb_v3"
TEST="data/pku_saferlhf_secure/test_pref.jsonl"

# Existing Stage 1 adapters
S1_QWEN_42="outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed42"
S1_QWEN_43="outputs_new_models/stage1/Qwen--Qwen2.5-3B_stage1_rr_eps1.0_seed43"
S1_MISTRAL_42="outputs_new_models/stage1/mistralai--Mistral-7B-v0.1_stage1_rr_eps1.0_seed42"
S1_MISTRAL_43="outputs_new_models/stage1/mistralai--Mistral-7B-v0.1_stage1_rr_eps1.0_seed43"

# Existing MLE-DPO adapters
MLE_QWEN_42="outputs_new_models/stage2_mle_dpo/Qwen--Qwen2.5-3B_stage2_mle_dpo_eps1.0_seed42"
MLE_QWEN_43="outputs_new_models/stage2_mle_dpo/Qwen--Qwen2.5-3B_stage2_mle_dpo_eps1.0_seed43"

# D2 datasets
D2_QWEN_42="outputs_new_models/preprocessing/d2_rr_flipped_eps1.0_seed42.jsonl"
D2_QWEN_43="outputs_new_models/preprocessing/d2_rr_flipped_eps1.0_seed43.jsonl"
D2_MISTRAL_42="$D2_QWEN_42"  # same data, different model
D2_MISTRAL_43="$D2_QWEN_43"

# Clean DPO (trained earlier tonight)
CLEAN_DPO_42="outputs_quick_test/stage1_clean_dpo_qwen_seed42"

mkdir -p "$LOGS" "$RESULTS" "$SB_MODELS"

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

run_task() {
    # Usage: run_task GPU_ID TASK_NAME COMMAND...
    local gpu=$1 name=$2
    shift 2
    local logfile="$LOGS/${name}.txt"
    echo "[$(timestamp)] GPU $gpu: Starting $name" | tee -a "$logfile"
    echo "  Command: $*" >> "$logfile"
    CUDA_VISIBLE_DEVICES=$gpu "$@" >> "$logfile" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(timestamp)] GPU $gpu: ✅ $name completed" | tee -a "$logfile"
    else
        echo "[$(timestamp)] GPU $gpu: ❌ $name FAILED (rc=$rc)" | tee -a "$logfile"
    fi
    return $rc
}

# ═══════════════════════════════════════════════════════
# GPU 0: verify → retrain SB Qwen s42 → eval SB Qwen s42
# ═══════════════════════════════════════════════════════
gpu0_worker() {
    echo "[$(timestamp)] GPU 0 worker started"

    # Task 0a: Verify SB fix (30 min)
    run_task 0 "verify_sb_fix" python -u experiments/verify_sb_fix.py
    cp outputs_quick_test/logs/verify_sb.txt "$RESULTS/verify_sb_result.txt" 2>/dev/null

    # Task 0b: Retrain SB Qwen s42 (2-3 hrs)
    local sb_out="$SB_MODELS/Qwen--Qwen2.5-3B_sb_v3_eps1.0_seed42"
    run_task 0 "train_sb_qwen_s42" python -u lora/train_stage2_soft_bayes.py \
        --model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_QWEN_42" \
        --data "$D2_QWEN_42" \
        --out "$sb_out" \
        --epsilon 1.0 --tau 0.0 --beta 0.1 --delta_clamp 500.0 \
        --lr 1e-5 --epochs 3 --bsz 1 --ga 16 --max_len 512 --bf16 --seed 42

    # Task 0c: Eval SB Qwen s42 (30 min)
    if [ -f "$sb_out/M2_manifest.json" ]; then
        run_task 0 "eval_sb_qwen_s42" python -u eval/eval_preference_accuracy.py \
            --manifest "$sb_out/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_sb_qwen_s42.json"
    fi

    # Task 0d: Re-eval full MLE Qwen s42 with new metric (30 min)
    if [ -f "$MLE_QWEN_42/M2_manifest.json" ]; then
        run_task 0 "eval_mle_qwen_s42_v3" python -u eval/eval_preference_accuracy.py \
            --manifest "$MLE_QWEN_42/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_mle_qwen_s42.json"
    fi

    echo "[$(timestamp)] GPU 0 worker DONE"
}

# ═══════════════════════════════════════════════════════
# GPU 1: eval S1 Qwen → retrain SB Qwen s43 → eval SB Qwen s43
# ═══════════════════════════════════════════════════════
gpu1_worker() {
    echo "[$(timestamp)] GPU 1 worker started"

    # Task 1a: Re-eval Stage 1 Qwen s42 with new DPO-reward metric (30 min)
    run_task 1 "eval_s1_qwen_s42_v3" python -u eval/eval_stage1_accuracy.py \
        --base_model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_QWEN_42" \
        --test_jsonl "$TEST" \
        --out_json "$RESULTS/eval_s1_qwen_s42.json"

    # Task 1b: Re-eval Stage 1 Qwen s43 (30 min)
    run_task 1 "eval_s1_qwen_s43_v3" python -u eval/eval_stage1_accuracy.py \
        --base_model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_QWEN_43" \
        --test_jsonl "$TEST" \
        --out_json "$RESULTS/eval_s1_qwen_s43.json"

    # Task 1c: Retrain SB Qwen s43 (2-3 hrs)
    local sb_out="$SB_MODELS/Qwen--Qwen2.5-3B_sb_v3_eps1.0_seed43"
    run_task 1 "train_sb_qwen_s43" python -u lora/train_stage2_soft_bayes.py \
        --model Qwen/Qwen2.5-3B \
        --stage1_adapter "$S1_QWEN_43" \
        --data "$D2_QWEN_43" \
        --out "$sb_out" \
        --epsilon 1.0 --tau 0.0 --beta 0.1 --delta_clamp 500.0 \
        --lr 1e-5 --epochs 3 --bsz 1 --ga 16 --max_len 512 --bf16 --seed 43

    # Task 1d: Eval SB Qwen s43 (30 min)
    if [ -f "$sb_out/M2_manifest.json" ]; then
        run_task 1 "eval_sb_qwen_s43" python -u eval/eval_preference_accuracy.py \
            --manifest "$sb_out/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_sb_qwen_s43.json"
    fi

    # Task 1e: Re-eval MLE Qwen s43 with new metric (30 min)
    if [ -f "$MLE_QWEN_43/M2_manifest.json" ]; then
        run_task 1 "eval_mle_qwen_s43_v3" python -u eval/eval_preference_accuracy.py \
            --manifest "$MLE_QWEN_43/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_mle_qwen_s43.json"
    fi

    echo "[$(timestamp)] GPU 1 worker DONE"
}

# ═══════════════════════════════════════════════════════
# GPU 2: eval clean DPO → retrain SB Mistral s42 → eval
# ═══════════════════════════════════════════════════════
gpu2_worker() {
    echo "[$(timestamp)] GPU 2 worker started"

    # Task 2a: Eval clean DPO upper bound (30 min)
    if [ -d "$CLEAN_DPO_42" ]; then
        run_task 2 "eval_clean_dpo_s42" python -u eval/eval_stage1_accuracy.py \
            --base_model Qwen/Qwen2.5-3B \
            --stage1_adapter "$CLEAN_DPO_42" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_clean_dpo_s42.json"
    fi

    # Task 2b: Eval Stage 1 Mistral s42 with new metric (45 min — bigger model)
    if [ -d "$S1_MISTRAL_42" ]; then
        run_task 2 "eval_s1_mistral_s42_v3" python -u eval/eval_stage1_accuracy.py \
            --base_model mistralai/Mistral-7B-v0.1 \
            --stage1_adapter "$S1_MISTRAL_42" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_s1_mistral_s42.json"
    fi

    # Task 2c: Retrain SB Mistral s42 (3-4 hrs — bigger model)
    local sb_out="$SB_MODELS/mistralai--Mistral-7B-v0.1_sb_v3_eps1.0_seed42"
    if [ -d "$S1_MISTRAL_42" ]; then
        run_task 2 "train_sb_mistral_s42" python -u lora/train_stage2_soft_bayes.py \
            --model mistralai/Mistral-7B-v0.1 \
            --stage1_adapter "$S1_MISTRAL_42" \
            --data "$D2_MISTRAL_42" \
            --out "$sb_out" \
            --epsilon 1.0 --tau 0.0 --beta 0.1 --delta_clamp 500.0 \
            --lr 1e-5 --epochs 3 --bsz 1 --ga 16 --max_len 512 --bf16 --seed 42
    fi

    # Task 2d: Eval SB Mistral s42
    if [ -f "$sb_out/M2_manifest.json" ]; then
        run_task 2 "eval_sb_mistral_s42" python -u eval/eval_preference_accuracy.py \
            --manifest "$sb_out/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_sb_mistral_s42.json"
    fi

    echo "[$(timestamp)] GPU 2 worker DONE"
}

# ═══════════════════════════════════════════════════════
# GPU 3: eval Mistral s43 → retrain SB Mistral s43 → eval
# ═══════════════════════════════════════════════════════
gpu3_worker() {
    echo "[$(timestamp)] GPU 3 worker started"

    # Task 3a: Eval Stage 1 Mistral s43 with new metric (45 min)
    if [ -d "$S1_MISTRAL_43" ]; then
        run_task 3 "eval_s1_mistral_s43_v3" python -u eval/eval_stage1_accuracy.py \
            --base_model mistralai/Mistral-7B-v0.1 \
            --stage1_adapter "$S1_MISTRAL_43" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_s1_mistral_s43.json"
    fi

    # Task 3b: Retrain SB Mistral s43 (3-4 hrs)
    local sb_out="$SB_MODELS/mistralai--Mistral-7B-v0.1_sb_v3_eps1.0_seed43"
    if [ -d "$S1_MISTRAL_43" ]; then
        run_task 3 "train_sb_mistral_s43" python -u lora/train_stage2_soft_bayes.py \
            --model mistralai/Mistral-7B-v0.1 \
            --stage1_adapter "$S1_MISTRAL_43" \
            --data "$D2_MISTRAL_43" \
            --out "$sb_out" \
            --epsilon 1.0 --tau 0.0 --beta 0.1 --delta_clamp 500.0 \
            --lr 1e-5 --epochs 3 --bsz 1 --ga 16 --max_len 512 --bf16 --seed 43
    fi

    # Task 3c: Eval SB Mistral s43
    if [ -f "$sb_out/M2_manifest.json" ]; then
        run_task 3 "eval_sb_mistral_s43" python -u eval/eval_preference_accuracy.py \
            --manifest "$sb_out/M2_manifest.json" \
            --test_jsonl "$TEST" \
            --out_json "$RESULTS/eval_sb_mistral_s43.json"
    fi

    echo "[$(timestamp)] GPU 3 worker DONE"
}

# ═══════════════════════════════════════════════════════
# LAUNCH ALL WORKERS
# ═══════════════════════════════════════════════════════
echo "═══════════════════════════════════════════════════"
echo " OVERNIGHT GPU JOB LAUNCHER"
echo " Started: $(timestamp)"
echo " Output:  $OUT/"
echo " Logs:    $LOGS/"
echo " Results: $RESULTS/"
echo "═══════════════════════════════════════════════════"
echo ""
echo " GPU 0: verify → SB Qwen s42 → eval"
echo " GPU 1: eval S1 Qwen → SB Qwen s43 → eval"
echo " GPU 2: eval clean DPO + Mistral S1 → SB Mistral s42 → eval"
echo " GPU 3: eval Mistral S1 s43 → SB Mistral s43 → eval"
echo ""
echo " Monitor: tail -f $LOGS/*.txt"
echo " Results: ls $RESULTS/"
echo "═══════════════════════════════════════════════════"

# Launch all 4 workers in background
gpu0_worker &
gpu1_worker &
gpu2_worker &
gpu3_worker &

echo ""
echo "All 4 GPU workers launched. PIDs:"
echo "  GPU 0: $(jobs -p | sed -n '1p')"
echo "  GPU 1: $(jobs -p | sed -n '2p')"
echo "  GPU 2: $(jobs -p | sed -n '3p')"
echo "  GPU 3: $(jobs -p | sed -n '4p')"
echo ""
echo "Use 'tail -f $LOGS/*.txt' to monitor progress."
echo "Use 'ls -la $RESULTS/' to check completed evals."
echo ""

# Wait for all workers
wait
echo ""
echo "═══════════════════════════════════════════════════"
echo " ALL WORKERS DONE: $(timestamp)"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Results:"
for f in "$RESULTS"/*.json; do
    if [ -f "$f" ]; then
        acc=$(python3 -c "import json; print(json.load(open('$f')).get('accuracy','N/A'))" 2>/dev/null)
        echo "  $(basename "$f"): accuracy=$acc"
    fi
done
