#!/bin/bash
# Clear GPU memory on all GPUs
# Usage: bash clear_gpu_memory.sh

source ~/.bashrc
conda activate pdpo

echo "🧹 Clearing GPU memory on all GPUs..."
python clear_gpu_memory.py

