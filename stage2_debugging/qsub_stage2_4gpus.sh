#!/bin/bash
#$ -N pdpo_stage2          # Specify job name
#$ -j y                    # Join standard output and standard error
#$ -q gpu@@jung_gpu        # Run on the specified GPU host group
#$ -l gpu_card=4           # Request 4 GPU cards
#$ -pe smp 4               # Request 4 CPU cores (must match or exceed GPUs)
#$ -l h_rt=10:00:00        # Specify runtime limit (10 hours)
#$ -m abe                  # Send mail when job begins(b), ends(e), and aborts(a)
#$ -M jzhao7@nd.edu

# Activate your environment
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "Stage 2 Job starting on $(hostname)"

# ======================================================================
# DATASET is passed as an environment variable from qsub (-v DATASET=...)
# Default to "pku" if not provided.
# ======================================================================
DATASET=${DATASET:-"pku"}

BASE_MODEL="Qwen/Qwen2.5-3B"
BSZ=1
GA=16
EPOCHS=1
LR=5e-5
MAX_LEN=1024

if [ "$DATASET" == "pku" ]; then
    S1_ADAPTER="outputs_new_models/stage1/pku_eps1.0_s42"
    D2_DATA="data/rr_flipped/pku_d2_eps1.0.jsonl"
elif [ "$DATASET" == "hhrlhf" ]; then
    S1_ADAPTER="outputs_new_models/stage1/hhrlhf_eps1.0_s42"
    D2_DATA="data/rr_flipped/hhrlhf_d2_eps1.0.jsonl"
elif [ "$DATASET" == "truthy" ]; then
    S1_ADAPTER="outputs_new_models/stage1/truthy_eps1.0_s42"
    D2_DATA="outputs_new_models/preprocessing/d2_rr_flipped_eps1.0_seed42.jsonl"
else
    echo "Unknown dataset: $DATASET"
    exit 1
fi

echo "Running Stage 2 over 4 GPUs for dataset: $DATASET"
echo "Using perfectly disjoint D2 data: $D2_DATA"

# ======================================================================
# VARIANT A: Faithful PROPS MAP (GPU 0)
# ======================================================================
echo "Starting Option A (MAP) on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -u lora/train_stage2_props_map.py \
    --model $BASE_MODEL \
    --stage1_adapter $S1_ADAPTER \
    --data $D2_DATA \
    --out "outputs_new_models/stage2/${DATASET}_props_map" \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > "outputs_new_models/stage2/${DATASET}_props_map.log" 2>&1 &

# ======================================================================
# VARIANT B: Calibrated Soft Bayes (GPU 1)
# ======================================================================
echo "Starting Option B (Soft Bayes) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -u lora/train_stage2_soft_bayes.py \
    --model $BASE_MODEL \
    --stage1_adapter $S1_ADAPTER \
    --data $D2_DATA \
    --out "outputs_new_models/stage2/${DATASET}_soft_bayes" \
    --beta 0.5 --lr 2.5e-5 --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > "outputs_new_models/stage2/${DATASET}_soft_bayes.log" 2>&1 &

# ======================================================================
# VARIANT C: Single LoRA MAP Retrain (GPU 2)
# ======================================================================
echo "Starting Option C (MAP Retrain) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -u lora/train_stage2_map_retrain.py \
    --model $BASE_MODEL \
    --stage1_adapter $S1_ADAPTER \
    --data $D2_DATA \
    --out "outputs_new_models/stage2/${DATASET}_map_retrain" \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > "outputs_new_models/stage2/${DATASET}_map_retrain.log" 2>&1 &

# ======================================================================
# BASELINE: Standard MLE DPO on Noisy D2 (GPU 3)
# ======================================================================
echo "Starting Baseline (MLE DPO) on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python -u lora/train_stage2_dpo.py \
    --model $BASE_MODEL \
    --stage1_adapter $S1_ADAPTER \
    --data $D2_DATA \
    --out "outputs_new_models/stage2/${DATASET}_mle_dpo" \
    --lr $LR --epochs $EPOCHS --bsz $BSZ --ga $GA --max_len $MAX_LEN --bf16 \
    > "outputs_new_models/stage2/${DATASET}_mle_dpo.log" 2>&1 &


echo "Waiting for all 4 Stage 2 jobs to securely finish..."
wait
echo "All Stage 2 jobs completed!"
