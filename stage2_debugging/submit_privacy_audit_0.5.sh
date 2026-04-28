#!/bin/bash
#$ -N privacy_audit
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

echo "[$(date)] Starting Privacy Audits (AUC-ROC MIA)"

# ======================= HH-RLHF (eps=0.5) =======================
HH_MEMBER="stage2_debugging/preprocessing/d2_rr_flipped_hhrlhf_eps0.5_seed42.jsonl"
HH_NONMEMBER="stage2_debugging/testsets/hhrlhf_test_pref_clean_v2.jsonl"

echo "Evaluating HH MLE..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_hhrlhf_eps0.5_s42/M2_manifest.json \
  --member_jsonl "$HH_MEMBER" --nonmember_jsonl "$HH_NONMEMBER" \
  --out_json stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_hhrlhf_eps0.5_s42/eval_privacy_auc.json \
  --subset_size 500

echo "Evaluating HH SB Fresh..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$HH_MEMBER" --nonmember_jsonl "$HH_NONMEMBER" \
  --out_json stage2_debugging/models/sb_fresh_hhrlhf_eps0.5_seed42/eval_privacy_auc.json \
  --subset_size 500

echo "Evaluating HH MAP Retrain..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$HH_MEMBER" --nonmember_jsonl "$HH_NONMEMBER" \
  --out_json stage2_debugging/models/map_retrain_hhrlhf_eps0.5_seed42/eval_privacy_auc.json \
  --subset_size 500

# ======================= PKU (eps=0.5) =======================
PKU_MEMBER="stage2_debugging/preprocessing/d2_rr_flipped_pku_eps0.5_seed42.jsonl"
PKU_NONMEMBER="stage2_debugging/testsets/pku_secure/test_pref.jsonl"

echo "Evaluating PKU MLE..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_pku_eps0.5_s42/M2_manifest.json \
  --member_jsonl "$PKU_MEMBER" --nonmember_jsonl "$PKU_NONMEMBER" \
  --out_json stage2_debugging/newplans/stage1_followup_hhpku_seed42/mle_fresh_pku_eps0.5_s42/eval_privacy_auc.json \
  --subset_size 500

echo "Evaluating PKU SB Fresh..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/sb_fresh_pku_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$PKU_MEMBER" --nonmember_jsonl "$PKU_NONMEMBER" \
  --out_json stage2_debugging/models/sb_fresh_pku_eps0.5_seed42/eval_privacy_auc.json \
  --subset_size 500

echo "Evaluating PKU MAP Retrain..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/map_retrain_pku_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$PKU_MEMBER" --nonmember_jsonl "$PKU_NONMEMBER" \
  --out_json stage2_debugging/models/map_retrain_pku_eps0.5_seed42/eval_privacy_auc.json \
  --subset_size 500

# ======================= TRUTHY (eps=0.5) =======================
TRUTHY_MEMBER="stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed42.jsonl"
TRUTHY_NONMEMBER="stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"

echo "Evaluating TRUTHY MLE..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/stage2_truthy/eps_sweep_seed42/mle_fresh_truthy_eps0.5_s42/M2_manifest.json \
  --member_jsonl "$TRUTHY_MEMBER" --nonmember_jsonl "$TRUTHY_NONMEMBER" \
  --out_json stage2_debugging/stage2_truthy/eps_sweep_seed42/mle_fresh_truthy_eps0.5_s42/eval_privacy_auc.json \
  --subset_size 500

echo "Evaluating TRUTHY SB Fresh..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/sb_fresh_truthy_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$TRUTHY_MEMBER" --nonmember_jsonl "$TRUTHY_NONMEMBER" \
  --out_json stage2_debugging/models/sb_fresh_truthy_eps0.5_seed42/eval_privacy_auc.json \
  --subset_size 500

echo "Evaluating TRUTHY MAP Retrain..."
python -u stage2_debugging/eval_privacy_audit.py \
  --manifest stage2_debugging/models/map_retrain_truthy_eps0.5_seed42/M2_manifest.json \
  --member_jsonl "$TRUTHY_MEMBER" --nonmember_jsonl "$TRUTHY_NONMEMBER" \
  --out_json stage2_debugging/models/map_retrain_truthy_eps0.5_seed42/eval_privacy_auc.json \
  --subset_size 500

echo "[$(date)] Finished all 9 model privacy audits!"
