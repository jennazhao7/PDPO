#!/bin/bash
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -N eval_stage2_checkpoints
#$ -j y
#$ -o stage2_debugging/eval_stage2.log

source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate pdpo
ROOT_DIR="/users/jzhao7/PDPO"
cd $ROOT_DIR

python stage2_debugging/test_stage2_acc.py
