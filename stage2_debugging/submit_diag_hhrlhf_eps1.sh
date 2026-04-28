#!/bin/bash
#$ -N sb_diag_hhrlhf
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=02:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

# Activate your environment
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO/stage2_debugging

echo "[$(date)] Starting SB diagnostic: HH-RLHF eps=1.0"
echo "Job ID: $JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python diagnose_sb_hhrlhf_eps1.py

echo "[$(date)] Done."
