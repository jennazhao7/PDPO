#!/bin/bash
set -euo pipefail
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate "pdpo"
cd "/users/jzhao7/PDPO"

export TOKENIZERS_PARALLELISM=false
export HF_HOME="/users/jzhao7/.cache/huggingface"
export HF_DATASETS_CACHE="/users/jzhao7/.cache/huggingface/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

CUDA_VISIBLE_DEVICES=0 python -u lora/preprocessing/train_truthy_stage1_lora.py   --model "Qwen/Qwen2.5-3B"   --data "/users/jzhao7/PDPO/stage2_debugging/preprocessing/d1_rr_flipped_truthy_eps1.0_seed43.jsonl"   --out "/users/jzhao7/PDPO/stage2_debugging/stage1/results_eps1.0_seed43_truthy/truthy_eps1.0_s43"   --max-steps "300"   --save-strategy steps   --save-steps 50   --save-total-limit 2   --epochs 3.0   --bsz 1   --ga 32   --lr 1e-4   --warmup-ratio 0.03   --max-prompt 256   --max-target 256   --max-len "384"   --lora-r 16   --lora-alpha 32   --lora-dropout 0.05   --target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"   --seed "43"   > "/users/jzhao7/PDPO/stage2_debugging/stage1/results_eps1.0_seed43_truthy/logs/train_truthy_s43.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python -u eval/eval_stage1_accuracy.py   --base_model "Qwen/Qwen2.5-3B"   --stage1_adapter "/users/jzhao7/PDPO/stage2_debugging/stage1/results_eps1.0_seed43_truthy/truthy_eps1.0_s43"   --test_jsonl "/users/jzhao7/PDPO/stage2_debugging/testsets/truthy_test_pref_random100_seed123.jsonl"   --out_json "/users/jzhao7/PDPO/stage2_debugging/stage1/results_eps1.0_seed43_truthy/eval/truthy_eps1.0_s43_eval.json"   --max_len "512"   > "/users/jzhao7/PDPO/stage2_debugging/stage1/results_eps1.0_seed43_truthy/logs/eval_truthy_s43.log" 2>&1
