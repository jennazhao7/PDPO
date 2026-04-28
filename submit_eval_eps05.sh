#!/bin/bash
#$ -N eval_all_eps05
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=04:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "[$(date)] Starting eps=0.5 evals for HH and PKU..."

# 1. HH-RLHF SB Fresh
python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/hhrlhf_test_pref_clean_v2.jsonl \
  --out_json stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42/eval_accuracy.json

# 2. HH-RLHF MAP Retrain
python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/hhrlhf_test_pref_clean_v2.jsonl \
  --out_json stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42/eval_accuracy.json

# 3. PKU SB Fresh
python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest stage2_debugging/models/sb_fresh_pku_eps0.5_seed42/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/pku_secure/test_pref.jsonl \
  --out_json stage2_debugging/models/sb_fresh_pku_eps0.5_seed42/eval_accuracy.json

# 4. PKU MAP Retrain
python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest stage2_debugging/models/map_retrain_pku_eps0.5_seed42/M2_manifest.json \
  --test_jsonl stage2_debugging/testsets/pku_secure/test_pref.jsonl \
  --out_json stage2_debugging/models/map_retrain_pku_eps0.5_seed42/eval_accuracy.json

echo "[$(date)] All evaluations complete! Check the eval_accuracy.json files in the respective model directories."
