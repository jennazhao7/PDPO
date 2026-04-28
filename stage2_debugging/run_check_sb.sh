#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -N test_sb_theory
#$ -j y
#$ -o stage2_debugging/check_sb_theory.log

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pdpo
cd /users/jzhao7/PDPO
python stage2_debugging/check_sb_theory.py
