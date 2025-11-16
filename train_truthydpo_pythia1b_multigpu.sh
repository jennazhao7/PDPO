#!/bin/bash
# Multi-GPU training script for pythia-1b on Truthy-DPO using torchrun

# === Environment ===
source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

# Assume your job was launched with gpu=4
# (check with nvidia-smi; you should see 4 GPUs)
NUM_GPUS=4

# Make sure all 4 GPUs are visible
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional: debugging info from torch distributed if things fail
export TORCH_DISTRIBUTED_DEBUG=INFO

# Optional: allocator tweak
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🚀 Starting multi-GPU training with $NUM_GPUS GPUs..."
nvidia-smi
echo ""

# Clear caches (not strictly necessary but fine)
python - << 'EOF'
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
print("✅ GPU cache cleared")
EOF

# === Launch torchrun ===
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPUS \
  train_dpo_stage1.py \
  --model EleutherAI/pythia-1b \
  --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
  --out ./models/truthydpo/pythia-1b-dpo-truthydpo \
  --epochs 3 \
  --bsz 1 \
  --ga 1 \
  --max-prompt 128 \
  --max-target 128 \
  --max-len 256
