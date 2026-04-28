#!/bin/bash
set -e

# Train over 3 datasets in parallel on 3 GPUs
# We use Qwen2.5 3B as the base model for speed and strong baseline capability.
BASE_MODEL="Qwen/Qwen2.5-3B"
BSZ=1
GA=32   # 1 * 32 = effective batch size 32
LR=5e-5
EPOCHS=1
MAX_LEN=1024

# 1. TruthyDPO
CUDA_VISIBLE_DEVICES=0 python -u lora/preprocessing/train_truthy_stage1_lora.py \
    --model $BASE_MODEL \
    --data outputs_new_models/preprocessing/d1_rr_flipped_eps1.0_seed42.jsonl \
    --out outputs_new_models/stage1/truthy_eps1.0_s42 \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > outputs_new_models/stage1/truthy_eps1.0_s42.log 2>&1 &

# 2. HH-RLHF (re-using the truthy script since data schema is identical)
CUDA_VISIBLE_DEVICES=1 python -u lora/preprocessing/train_truthy_stage1_lora.py \
    --model $BASE_MODEL \
    --data data/rr_flipped/hhrlhf_train_eps1.0.jsonl \
    --out outputs_new_models/stage1/hhrlhf_eps1.0_s42 \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > outputs_new_models/stage1/hhrlhf_eps1.0_s42.log 2>&1 &

# 3. PKU-SafeRLHF
CUDA_VISIBLE_DEVICES=2 python -u lora/preprocessing/train_truthy_stage1_lora.py \
    --model $BASE_MODEL \
    --data data/rr_flipped/pku_train_eps1.0.jsonl \
    --out outputs_new_models/stage1/pku_eps1.0_s42 \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > outputs_new_models/stage1/pku_eps1.0_s42.log 2>&1 &

wait
echo "All Stage 1 training jobs completed!"
