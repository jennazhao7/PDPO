#!/bin/bash
#$ -N limo_score_cache
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=3
#$ -pe smp 12
#$ -l h_rt=12:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

set -eo pipefail
set +u
source ~/.bashrc
set -u
conda activate pdpo

cd /users/jzhao7/PDPO

echo "========================================================"
echo "[limo-cache] Job started: $(date)"
echo "[limo-cache] Host: $(hostname)"
echo "========================================================"

bash /users/jzhao7/PDPO/stage2_debugging/newplans/less_is_more_dp/run_limo_score_cache_3datasets.sh

echo "========================================================"
echo "[limo-cache] Job finished: $(date)"
echo "========================================================"
