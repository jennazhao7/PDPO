#!/bin/bash
source ~/.bashrc
conda activate pdpo
export CUDA_VISIBLE_DEVICES=0
python scripts/train_one_stage_limo_dp.py \
    --train_data outputs/limo_dp_onestage/hhrlhf_train_shard_0.jsonl \
    --eval_data stage2_debugging/test_pref.jsonl \
    --base_model Qwen/Qwen2.5-3B \
    --output_dir outputs/limo_dp_onestage/test \
    --log_file outputs/limo_dp_onestage/logs/test.json \
    --max_steps 5 --bsz 1 --ga 1 --bf16
