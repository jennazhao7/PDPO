#!/bin/bash
# LoRA-based DPO training script for pythia-1b on Truthy-DPO
# Usage: bash train_pythia1b_lora.sh
# LoRA significantly reduces memory usage, allowing larger batch sizes

# === Setup environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# === Run LoRA fine-tuning ===
# Memory-optimized settings to avoid OOM:
# - batch_size: 1 (bsz) per GPU - reduced from 4 to save memory
# - gradient_accumulation_steps: 4 (ga) - increased to maintain effective batch size = 16 (1 × 4 GPUs × 4 GA)
# - max_length: 256 (max-len) - reduced from 512 to save memory
# - max_prompt_length: 128 (max-prompt) - reduced from 256
# - max_target_length: 128 (max-target) - reduced from 256
# - LoRA rank: 16 (lora-r) - good balance of performance and memory
# - LoRA alpha: 32 (lora-alpha) - typically 2x rank
echo "🚀 Starting LoRA-based DPO training for pythia-1b..."
echo "   Memory-optimized settings: bsz=1 per GPU, ga=4, max_len=256"
echo "   Effective batch size: 1 × 4 GPUs × 4 GA = 16 (matches PROPS-2025 eff_bsz)"
echo "   LoRA adapters will be saved (much smaller than full model)"

# Option 1: Single GPU training (simpler, LoRA is memory-efficient)
# Uncomment this if you want single GPU:
# python train_dpo_lora.py \
#   --model EleutherAI/pythia-1b \
#   --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
#   --out ./models/truthydpo/pythia-1b-dpo-lora-truthydpo \
#   --epochs 3 \
#   --bsz 2 \
#   --ga 8 \
#   --max-prompt 128 \
#   --max-target 128 \
#   --max-len 256 \
#   --lora-r 16 \
#   --lora-alpha 32 \
#   --lora-dropout 0.05

# Option 2: Multi-GPU training with torchrun (if you need distributed training)
# Set environment variables for NCCL
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_BACKEND=nccl
export NCCL_DEBUG=WARN

NUM_GPUS=4
echo "🚀 Starting multi-GPU LoRA training with torchrun on $NUM_GPUS GPUs..."

torchrun --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_dpo_lora.py \
    --model EleutherAI/pythia-1b \
    --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
    --out ./models/truthydpo/pythia-1b-dpo-lora-truthydpo \
    --epochs 3 \
    --bsz 1 \
    --ga 4 \
    --max-prompt 128 \
    --max-target 128 \
    --max-len 256 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05

