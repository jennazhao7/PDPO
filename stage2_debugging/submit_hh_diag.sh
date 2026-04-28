#!/bin/bash
#$ -N diag_hh
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

HH_MEMBER="stage2_debugging/preprocessing/d2_rr_flipped_hhrlhf_eps0.5_seed42.jsonl"
HH_NONMEMBER="stage2_debugging/testsets/hhrlhf_test_pref_clean_v2.jsonl"

python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$HH_MEMBER" --nonmember_jsonl "$HH_NONMEMBER" \
  --out_json stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42/eval_privacy_auc_diag.json \
  --subset_size 500

python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$HH_MEMBER" --nonmember_jsonl "$HH_NONMEMBER" \
  --out_json stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42/eval_privacy_auc_diag.json \
  --subset_size 500
