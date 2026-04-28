#!/bin/bash
#$ -N pdpo_job             # Specify job name
#$ -j y                    # Join standard output and standard error
#$ -q gpu@@jung_gpu        # Run on the specified GPU host group
#$ -l gpu_card=4           # Request 4 GPU cards
#$ -pe smp 4               # Request 4 CPU cores (must match or exceed GPUs)
#$ -l h_rt=10:00:00        # Specify runtime limit (10 hours)
#$ -m abe                  # Send mail when job begins(b), ends(e), and aborts(a)
#$ -M jzhao7@nd.edu

# Load necessary modules if not in bash_profile
# module load conda

# Activate your environment
source ~/.bashrc
conda activate pdpo

# Navigate to your working directory
cd /users/jzhao7/PDPO

echo "Job starting on $(hostname) with $NSLOTS allocated cores."
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

# ----------------------------------------------------------------------
# PUT YOUR COMMANDS BELOW THIS LINE
# ----------------------------------------------------------------------

# Example: Run the multi-GPU evaluation script
# chmod +x experiments/core_result_tonight/eval_stage1_4gpus.sh
# ./experiments/core_result_tonight/eval_stage1_4gpus.sh

echo "Job finished at $(date)"
