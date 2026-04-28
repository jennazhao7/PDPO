#!/bin/bash
set -e

# Make sure to activate the correct physical environment:
source ~/.bashrc
conda activate pdpo

# Use the python executable from the current environment
PYTHON_CMD=$(which python)

BASE_MODEL="Qwen/Qwen2.5-3B"

echo "Evaluating TruthyDPO on GPU 0..."
CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD -u eval/eval_stage1_accuracy.py \
    --base_model $BASE_MODEL \
    --stage1_adapter outputs_new_models/stage1/truthy_eps1.0_s42 \
    --test_jsonl preprocessing/truthydpo/truthy_dpo_subset.jsonl \
    --out_json outputs_new_models/stage1/truthy_eval.json \
    --max_len 1024 > outputs_new_models/stage1/truthy_eval.log 2>&1 &

echo "Evaluating HH-RLHF on GPU 1..."
CUDA_VISIBLE_DEVICES=1 $PYTHON_CMD -u eval/eval_stage1_accuracy.py \
    --base_model $BASE_MODEL \
    --stage1_adapter outputs_new_models/stage1/hhrlhf_eps1.0_s42 \
    --test_jsonl data/hhrlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage1/hhrlhf_eval.json \
    --max_len 1024 > outputs_new_models/stage1/hhrlhf_eval.log 2>&1 &

echo "Evaluating PKU-SafeRLHF on GPU 2..."
CUDA_VISIBLE_DEVICES=2 $PYTHON_CMD -u eval/eval_stage1_accuracy.py \
    --base_model $BASE_MODEL \
    --stage1_adapter outputs_new_models/stage1/pku_eps1.0_s42 \
    --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage1/pku_eval.json \
    --max_len 1024 > outputs_new_models/stage1/pku_eval.log 2>&1 &

echo "Waiting for all 3 evaluations to finish..."
wait

echo "=========================================="
echo "EVALUATION RESULTS (Expected ~55%+ accuracy)"
echo "=========================================="
for f in outputs_new_models/stage1/*eval.json; do
    echo "--- $f ---"
    cat $f | grep -E '"dataset_or_model"|"accuracy"|"ece"|"mean_margin"'
    echo ""
done
