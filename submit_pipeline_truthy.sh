#!/bin/bash
#$ -N pipeline_truthy_s2
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=06:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "[$(date)] Starting Job 3: TruthyDPO eps=1.0 (SB Fresh Only)"

# Truthy Stage 1 already exists at: stage2_debugging/stage1/results_eps1.0_seed42/truthy_eps1.0_s42
S1_OUT="stage2_debugging/stage1/results_eps1.0_seed42/truthy_eps1.0_s42"
SB_OUT="stage2_debugging/models/sb_fresh_truthy_eps1.0_seed42"

echo "[$(date)] Running Truthy Stage 2: SB Fresh..."
python -u stage2_debugging/train_stage2_sb_fresh.py \
    --model "Qwen/Qwen2.5-3B" \
    --stage1_adapter "$S1_OUT" \
    --data "stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps1.0_seed42.jsonl" \
    --beta 0.5 --lr 2.5e-5 --epsilon 1.0 --bsz 1 --ga 16 --epochs 1 --max_len 1024 --bf16 \
    --out "$SB_OUT"

echo "[$(date)] Done Job 3: TruthyDPO eps=1.0"
