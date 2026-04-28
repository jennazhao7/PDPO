#!/bin/bash
#$ -N sbfresh_resume
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=01:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

# Activate environment
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "[$(date)] Resuming SB-Fresh training on HH-RLHF eps=1.0..."
echo "Job ID: $JOB_ID"

OUT_DIR="models/sb_fresh_hhrlhf_eps1.0_seed42"

python -u stage2_debugging/train_stage2_sb_fresh.py \
    --model Qwen/Qwen2.5-3B \
    --stage1_adapter stage2_debugging/stage1/results_eps1.0_seed42/hhrlhf_eps1.0_s42 \
    --data stage2_debugging/preprocessing/d2_rr_flipped_hhrlhf_eps1.0_seed42.jsonl \
    --beta 0.5 \
    --lr 2.5e-5 \
    --epsilon 1.0 \
    --bsz 1 \
    --ga 16 \
    --epochs 1 \
    --max_len 1024 \
    --bf16 \
    --resume_from_checkpoint "$OUT_DIR/checkpoint-step-300" \
    --out "$OUT_DIR"

echo "[$(date)] Done training."
