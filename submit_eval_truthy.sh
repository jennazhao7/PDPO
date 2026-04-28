#!/bin/bash
#$ -N eval_truthy_s2
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=01:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest stage2_debugging/models/sb_fresh_truthy_eps1.0_seed42/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl \
  --out_json stage2_debugging/models/sb_fresh_truthy_eps1.0_seed42/eval_accuracy.json
