#!/bin/bash
#$ -N limo_score_cache
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=4
#$ -pe smp 4
#$ -l h_rt=10:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

source ~/.bashrc
conda activate pdpo
cd /users/jzhao7/PDPO

echo "Job starting on $(hostname) with $NSLOTS allocated cores."
echo "CUDA_VISIBLE_DEVICES = $CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=0 python scripts/score_with_skywork.py \
    --input outputs/limo_dp_onestage/hhrlhf_train_shard_0.jsonl \
    --output outputs/limo_dp_onestage/skywork_shard_0.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

CUDA_VISIBLE_DEVICES=1 python scripts/score_with_skywork.py \
    --input outputs/limo_dp_onestage/hhrlhf_train_shard_1.jsonl \
    --output outputs/limo_dp_onestage/skywork_shard_1.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

CUDA_VISIBLE_DEVICES=2 python scripts/score_with_skywork.py \
    --input outputs/limo_dp_onestage/hhrlhf_train_shard_2.jsonl \
    --output outputs/limo_dp_onestage/skywork_shard_2.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

CUDA_VISIBLE_DEVICES=3 python scripts/score_with_skywork.py \
    --input outputs/limo_dp_onestage/hhrlhf_train_shard_3.jsonl \
    --output outputs/limo_dp_onestage/skywork_shard_3.jsonl \
    --model_name Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    --batch_size 8 --max_length 4096 &

wait

cat outputs/limo_dp_onestage/skywork_shard_*.jsonl > outputs/limo_dp_onestage/skywork_margins_hhrlhf_train.jsonl
echo "Job finished at $(date)"
