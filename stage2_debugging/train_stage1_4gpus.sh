#!/bin/bash
set -e

# Make sure you are in the correct conda environment (e.g. conda activate pdpo)
# Use the python executable from the current environment
PYTHON_CMD=$(which python)
BASE_MODEL="Qwen/Qwen2.5-3B"
BSZ=1
GA=32   # 1 * 32 = effective batch size 32
LR=5e-5
EPOCHS=1
MAX_LEN=1024

echo "Training TruthyDPO on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD -u lora/preprocessing/train_truthy_stage1_lora.py \
    --model $BASE_MODEL \
    --data outputs_new_models/preprocessing/d1_rr_flipped_eps1.0_seed42.jsonl \
    --out outputs_new_models/stage1/truthy_eps1.0_s42 \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max-len $MAX_LEN \
    > outputs_new_models/stage1/truthy_train.log 2>&1 &

echo "Training HH-RLHF on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD -u lora/preprocessing/train_truthy_stage1_lora.py \
    --model $BASE_MODEL \
    --data data/rr_flipped/hhrlhf_train_eps1.0.jsonl \
    --out outputs_new_models/stage1/hhrlhf_eps1.0_s42 \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max-len $MAX_LEN \
    > outputs_new_models/stage1/hhrlhf_train.log 2>&1 &

echo "Training PKU-SafeRLHF on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_CMD -u lora/preprocessing/train_truthy_stage1_lora.py \
    --model $BASE_MODEL \
    --data data/rr_flipped/pku_train_eps1.0.jsonl \
    --out outputs_new_models/stage1/pku_eps1.0_s42 \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max-len $MAX_LEN \
    > outputs_new_models/stage1/pku_train.log 2>&1 &

echo "Waiting for all 3 Stage 1 training jobs to finish..."
wait
echo "All Stage 1 training jobs completed!"
