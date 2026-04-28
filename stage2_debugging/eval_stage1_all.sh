#!/bin/bash
set -e

BASE_MODEL="Qwen/Qwen2.5-3B"

# 1. TruthyDPO
echo "Evaluating TruthyDPO..."
CUDA_VISIBLE_DEVICES=0 python -u eval/eval_stage1_accuracy.py \
    --base_model $BASE_MODEL \
    --stage1_adapter outputs_new_models/stage1/truthy_eps1.0_s42 \
    --test_jsonl data/truthydpo/test_pref.jsonl \
    --out_json outputs_new_models/stage1/truthy_eps1.0_s42_eval.json \
    --max_len 1024 &

# 2. HH-RLHF
echo "Evaluating HH-RLHF..."
CUDA_VISIBLE_DEVICES=1 python -u eval/eval_stage1_accuracy.py \
    --base_model $BASE_MODEL \
    --stage1_adapter outputs_new_models/stage1/hhrlhf_eps1.0_s42 \
    --test_jsonl data/hhrlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage1/hhrlhf_eps1.0_s42_eval.json \
    --max_len 1024 &

# 3. PKU-SafeRLHF
echo "Evaluating PKU-SafeRLHF..."
CUDA_VISIBLE_DEVICES=2 python -u eval/eval_stage1_accuracy.py \
    --base_model $BASE_MODEL \
    --stage1_adapter outputs_new_models/stage1/pku_eps1.0_s42 \
    --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage1/pku_eps1.0_s42_eval.json \
    --max_len 1024 &

wait
echo "All eval jobs completed!"
