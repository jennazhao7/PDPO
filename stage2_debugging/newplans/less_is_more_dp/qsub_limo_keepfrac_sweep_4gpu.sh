#!/bin/bash
#$ -N limo_keepfrac_4g
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4
#$ -pe smp 16
#$ -l h_rt=36:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

set -eo pipefail
set +u
source ~/.bashrc
set -u
conda activate pdpo

cd /users/jzhao7/PDPO

echo "========================================================"
echo "[limo-keepfrac] Job started: $(date)"
echo "[limo-keepfrac] Host: $(hostname)"
echo "========================================================"

bash /users/jzhao7/PDPO/stage2_debugging/newplans/less_is_more_dp/run_limo_keepfrac_sweep_4gpu.sh

echo "========================================================"
echo "[limo-keepfrac] Job finished: $(date)"
echo "========================================================"
