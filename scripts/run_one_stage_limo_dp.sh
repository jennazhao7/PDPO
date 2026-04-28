#!/bin/bash
#$ -N one_stage_limo
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4
#$ -pe smp 4
#$ -l h_rt=10:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "Job starting on $(hostname) with $NSLOTS allocated cores."
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

export BASE_MODEL="Qwen/Qwen2.5-3B"
export OUTPUT_ROOT="/users/jzhao7/PDPO/outputs/limo_dp_onestage"

echo "=== PHASE 1: SCORING ==="
CUDA_VISIBLE_DEVICES=0 python scripts/score_with_skywork.py \
    --input $OUTPUT_ROOT/hhrlhf_train_shard_0.jsonl \
    --output $OUTPUT_ROOT/skywork_shard_0.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

CUDA_VISIBLE_DEVICES=1 python scripts/score_with_skywork.py \
    --input $OUTPUT_ROOT/hhrlhf_train_shard_1.jsonl \
    --output $OUTPUT_ROOT/skywork_shard_1.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

CUDA_VISIBLE_DEVICES=2 python scripts/score_with_skywork.py \
    --input $OUTPUT_ROOT/hhrlhf_train_shard_2.jsonl \
    --output $OUTPUT_ROOT/skywork_shard_2.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

CUDA_VISIBLE_DEVICES=3 python scripts/score_with_skywork.py \
    --input $OUTPUT_ROOT/hhrlhf_train_shard_3.jsonl \
    --output $OUTPUT_ROOT/skywork_shard_3.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

wait

cat $OUTPUT_ROOT/skywork_shard_*.jsonl > $OUTPUT_ROOT/skywork_margins_hhrlhf_train.jsonl
echo "Phase 1 complete."

echo "=== PHASE 2: RR + BAYESIAN WEIGHTING ==="
python scripts/build_one_stage_dataset.py \
    --margins_file $OUTPUT_ROOT/skywork_margins_hhrlhf_train.jsonl \
    --original_data stage2_debugging/train_pref.jsonl \
    --output_prefix $OUTPUT_ROOT/d_train_eps1.0_seed42 \
    --epsilon 1.0 \
    --seed 42
echo "Phase 2 complete."

echo "=== PHASE 3: TRAINING ==="
TRAIN_SCRIPT="scripts/train_one_stage_limo_dp.py"

# GPU 0: keep=0.5 (headline)
CUDA_VISIBLE_DEVICES=0 python $TRAIN_SCRIPT \
    --train_data $OUTPUT_ROOT/d_train_eps1.0_seed42_keep0.5.jsonl \
    --eval_data stage2_debugging/test_pref.jsonl \
    --base_model $BASE_MODEL \
    --epsilon 1.0 \
    --seed 42 \
    --use_weights True \
    --output_dir $OUTPUT_ROOT/onestage_keep0.5_seed42 \
    --log_file $OUTPUT_ROOT/logs/onestage_keep0.5.json \
    > $OUTPUT_ROOT/logs/onestage_keep0.5.stdout 2>&1 &

# GPU 1: keep=1.0 (no selection, weighting only)
CUDA_VISIBLE_DEVICES=1 python $TRAIN_SCRIPT \
    --train_data $OUTPUT_ROOT/d_train_eps1.0_seed42_keep1.0.jsonl \
    --eval_data stage2_debugging/test_pref.jsonl \
    --base_model $BASE_MODEL \
    --epsilon 1.0 \
    --seed 42 \
    --use_weights True \
    --output_dir $OUTPUT_ROOT/onestage_keep1.0_seed42 \
    --log_file $OUTPUT_ROOT/logs/onestage_keep1.0.json \
    > $OUTPUT_ROOT/logs/onestage_keep1.0.stdout 2>&1 &

# GPU 2: keep=0.25 (aggressive selection)
CUDA_VISIBLE_DEVICES=2 python $TRAIN_SCRIPT \
    --train_data $OUTPUT_ROOT/d_train_eps1.0_seed42_keep0.25.jsonl \
    --eval_data stage2_debugging/test_pref.jsonl \
    --base_model $BASE_MODEL \
    --epsilon 1.0 \
    --seed 42 \
    --use_weights True \
    --output_dir $OUTPUT_ROOT/onestage_keep0.25_seed42 \
    --log_file $OUTPUT_ROOT/logs/onestage_keep0.25.json \
    > $OUTPUT_ROOT/logs/onestage_keep0.25.stdout 2>&1 &

# GPU 3: control — one-stage MLE-Fresh on RR'd labels (no Skywork at all)
CUDA_VISIBLE_DEVICES=3 python $TRAIN_SCRIPT \
    --train_data $OUTPUT_ROOT/d_train_eps1.0_seed42_rr_only.jsonl \
    --eval_data stage2_debugging/test_pref.jsonl \
    --base_model $BASE_MODEL \
    --epsilon 1.0 \
    --seed 42 \
    --use_weights False \
    --output_dir $OUTPUT_ROOT/onestage_rr_only_seed42 \
    --log_file $OUTPUT_ROOT/logs/onestage_rr_only.json \
    > $OUTPUT_ROOT/logs/onestage_rr_only.stdout 2>&1 &

wait
echo "Phase 3 complete."

echo "=== PHASE 4: EVALUATION ==="
# We will use the existing eval script for the final results
CUDA_VISIBLE_DEVICES=0 python stage2_debugging/eval_preference_accuracy.py \
    --manifest $OUTPUT_ROOT/onestage_keep0.5_seed42/M2_manifest.json \
    --test_jsonl stage2_debugging/test_pref.jsonl \
    --out_json $OUTPUT_ROOT/eval/hhrlhf_onestage_keep0.5_eval.json \
    --max_len 512 &

CUDA_VISIBLE_DEVICES=1 python stage2_debugging/eval_preference_accuracy.py \
    --manifest $OUTPUT_ROOT/onestage_keep1.0_seed42/M2_manifest.json \
    --test_jsonl stage2_debugging/test_pref.jsonl \
    --out_json $OUTPUT_ROOT/eval/hhrlhf_onestage_keep1.0_eval.json \
    --max_len 512 &

CUDA_VISIBLE_DEVICES=2 python stage2_debugging/eval_preference_accuracy.py \
    --manifest $OUTPUT_ROOT/onestage_keep0.25_seed42/M2_manifest.json \
    --test_jsonl stage2_debugging/test_pref.jsonl \
    --out_json $OUTPUT_ROOT/eval/hhrlhf_onestage_keep0.25_eval.json \
    --max_len 512 &

CUDA_VISIBLE_DEVICES=3 python stage2_debugging/eval_preference_accuracy.py \
    --manifest $OUTPUT_ROOT/onestage_rr_only_seed42/M2_manifest.json \
    --test_jsonl stage2_debugging/test_pref.jsonl \
    --out_json $OUTPUT_ROOT/eval/hhrlhf_onestage_rr_only_eval.json \
    --max_len 512 &

wait
echo "Phase 4 complete."

echo "All tasks finished successfully at $(date)."
