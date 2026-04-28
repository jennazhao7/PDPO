#!/bin/bash
# Eval script for PKU Stage 2 models

cd /users/jzhao7/PDPO

echo "Evaluating PKU PROPS MAP on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -u eval/eval_preference_accuracy.py \
    --manifest outputs_new_models/stage2/pku_props_map/M2_manifest.json \
    --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage2/pku_props_map/eval.json \
    > outputs_new_models/stage2/pku_props_map/eval.log 2>&1 &

echo "Evaluating PKU Soft Bayes on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python -u eval/eval_preference_accuracy.py \
    --manifest outputs_new_models/stage2/pku_soft_bayes/M2_manifest.json \
    --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage2/pku_soft_bayes/eval.json \
    > outputs_new_models/stage2/pku_soft_bayes/eval.log 2>&1 &

echo "Evaluating PKU MAP Retrain on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python -u eval/eval_preference_accuracy.py \
    --manifest outputs_new_models/stage2/pku_map_retrain/M2_manifest.json \
    --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage2/pku_map_retrain/eval.json \
    > outputs_new_models/stage2/pku_map_retrain/eval.log 2>&1 &

echo "Evaluating PKU MLE DPO on GPU 3..."
CUDA_VISIBLE_DEVICES=3 python -u eval/eval_preference_accuracy.py \
    --manifest outputs_new_models/stage2/pku_mle_dpo/M2_manifest.json \
    --test_jsonl data/pku_saferlhf_secure/test_pref.jsonl \
    --out_json outputs_new_models/stage2/pku_mle_dpo/eval.json \
    > outputs_new_models/stage2/pku_mle_dpo/eval.log 2>&1 &

echo "Waiting for all evaluations to finish..."
wait
echo "All done! Results saved to eval.json in each output directory."
