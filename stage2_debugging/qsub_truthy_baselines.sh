#!/bin/bash
#$ -N pdpo_truthy_baseline
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4           # Request 4 GPUs concurrently
#$ -pe smp 4               # Request 4 CPU cores
#$ -l h_rt=24:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "Parallel Job starting on $(hostname)"
mkdir -p outputs_new_models/logs

# Shared parameters
MODEL="Qwen/Qwen2.5-3B"
BETA=0.1
LR=1e-5
EPOCHS=1.0

# -----------------
# Task 1: cDPO (eps 0.5) [GPU 0]
# -----------------
CUDA_VISIBLE_DEVICES=0 python stage2_debugging/train_stage2_cdpo.py \
  --model $MODEL \
  --stage1_adapter stage2_debugging/newplans/results_truthy_dpsgd_plain_seed42/plain_lora_truthy_eps0.5_s42 \
  --data stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed42.jsonl \
  --out outputs_new_models/stage2_cdpo/truthy_eps0.5_s42 \
  --rr_epsilon 0.5 \
  --beta $BETA --lr $LR --epochs $EPOCHS --bsz 1 --ga 16 --bf16 --seed 42 \
  > outputs_new_models/logs/cdpo_truthy_eps0.5_s42.log 2>&1 &

# -----------------
# Task 2: cDPO (eps 1.0) [GPU 1]
# -----------------
CUDA_VISIBLE_DEVICES=1 python stage2_debugging/train_stage2_cdpo.py \
  --model $MODEL \
  --stage1_adapter stage2_debugging/stage1/results_eps1.0_seed42/truthy_eps1.0_s42 \
  --data stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps1.0_seed42.jsonl \
  --out outputs_new_models/stage2_cdpo/truthy_eps1.0_s42 \
  --rr_epsilon 1.0 \
  --beta $BETA --lr $LR --epochs $EPOCHS --bsz 1 --ga 16 --bf16 --seed 42 \
  > outputs_new_models/logs/cdpo_truthy_eps1.0_s42.log 2>&1 &

# -----------------
# Task 3: rDPO (eps 0.5) [GPU 2]
# -----------------
CUDA_VISIBLE_DEVICES=2 python stage2_debugging/train_stage2_rdpo.py \
  --model $MODEL \
  --stage1_adapter stage2_debugging/newplans/results_truthy_dpsgd_plain_seed42/plain_lora_truthy_eps0.5_s42 \
  --data stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed42.jsonl \
  --out outputs_new_models/stage2_rdpo/truthy_eps0.5_s42 \
  --rr_epsilon 0.5 \
  --beta $BETA --lr $LR --epochs $EPOCHS --bsz 1 --ga 16 --bf16 --seed 42 \
  > outputs_new_models/logs/rdpo_truthy_eps0.5_s42.log 2>&1 &

# -----------------
# Task 4: rDPO (eps 1.0) [GPU 3]
# -----------------
CUDA_VISIBLE_DEVICES=3 python stage2_debugging/train_stage2_rdpo.py \
  --model $MODEL \
  --stage1_adapter stage2_debugging/stage1/results_eps1.0_seed42/truthy_eps1.0_s42 \
  --data stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps1.0_seed42.jsonl \
  --out outputs_new_models/stage2_rdpo/truthy_eps1.0_s42 \
  --rr_epsilon 1.0 \
  --beta $BETA --lr $LR --epochs $EPOCHS --bsz 1 --ga 16 --bf16 --seed 42 \
  > outputs_new_models/logs/rdpo_truthy_eps1.0_s42.log 2>&1 &

# Wait for all background tasks to finish
wait

echo "All tasks finished successfully at $(date)."
