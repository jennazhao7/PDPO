#!/bin/bash
#$ -N pipeline_truthy_0.5
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=1
#$ -pe smp 4
#$ -l h_rt=08:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "[$(date)] Starting Job: Truthy eps=0.5 (Stage 1 -> SB Fresh -> MAP Retrain -> Eval)"

# -------------- STAGE 1 --------------
S1_OUT="stage2_debugging/stage1/results_eps0.5_seed42/truthy_eps0.5_s42"
echo "[$(date)] Running Truthy Stage 1..."
python -u lora/preprocessing/train_truthy_stage1_lora.py \
  --model "Qwen/Qwen2.5-3B" \
  --data "stage2_debugging/preprocessing/d1_rr_flipped_truthy_eps0.5_seed42.jsonl" \
  --out "$S1_OUT" \
  --max-steps 300 \
  --save-strategy no \
  --epochs 3.0 \
  --bsz 1 \
  --ga 32 \
  --lr 1e-4 \
  --warmup-ratio 0.03 \
  --max-prompt 256 \
  --max-target 256 \
  --max-len 384 \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --seed 42

# -------------- STAGE 2: SB FRESH --------------
SB_OUT="stage2_debugging/models/sb_fresh_truthy_eps0.5_seed42"
echo "[$(date)] Running Truthy Stage 2: SB Fresh..."
python -u stage2_debugging/train_stage2_sb_fresh.py \
    --model "Qwen/Qwen2.5-3B" \
    --stage1_adapter "$S1_OUT" \
    --data "stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed42.jsonl" \
    --beta 0.5 --lr 2.5e-5 --epsilon 0.5 --bsz 1 --ga 16 --epochs 1 --max_len 1024 --bf16 \
    --out "$SB_OUT"

# -------------- STAGE 2: MAP RETRAIN --------------
MAP_OUT="stage2_debugging/models/map_retrain_truthy_eps0.5_seed42"
echo "[$(date)] Running Truthy Stage 2: MAP Retrain..."
python -u stage2_debugging/train_stage2_map_retrain.py \
    --model "Qwen/Qwen2.5-3B" \
    --stage1_adapter "$S1_OUT" \
    --data "stage2_debugging/preprocessing/d2_rr_flipped_truthy_eps0.5_seed42.jsonl" \
    --lr 2.5e-5 --epochs 1 --bsz 1 --ga 16 --max_len 1024 --bf16 \
    --out "$MAP_OUT"

# -------------- EVALUATION --------------
echo "[$(date)] Evaluating Truthy eps=0.5 (SB Fresh)"
python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "$SB_OUT/M2_manifest.json" \
  --test_jsonl stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl \
  --out_json "$SB_OUT/eval_accuracy.json"

echo "[$(date)] Evaluating Truthy eps=0.5 (MAP Retrain)"
python -u stage2_debugging/eval_preference_accuracy.py \
  --manifest "$MAP_OUT/M2_manifest.json" \
  --test_jsonl stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl \
  --out_json "$MAP_OUT/eval_accuracy.json"

echo "[$(date)] Done Job: Truthy eps=0.5 Full Pipeline!"
