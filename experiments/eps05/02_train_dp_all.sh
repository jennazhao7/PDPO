#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

DATASET_PATH="${DATASET_PATH:-preprocessing/truthydpo/truthy_dpo_subset.jsonl}"
STEPS="${STEPS:-300}"
EPSILON="${EPSILON:-0.5}"
SEED="${SEED:-42}"

COMMON_ARGS=(
  --dataset_path "$DATASET_PATH"
  --seed "$SEED"
  --max_steps "$STEPS"
  --learning_rate 5e-5
  --warmup_ratio 0.03
  --lr_scheduler_type cosine
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 16
  --max_seq_length 512
  --lora_r 16
  --lora_alpha 32
  --lora_dropout 0.05
  --dpo_beta 0.1
  --dp
  --epsilon "$EPSILON"
  --max_grad_norm 1.0
  --accountant rdp
  --logging_steps 10
  --max_eval_samples 200
  --gen_max_new_tokens 128
)

echo "[eps05] DP-LoRA gpt2-medium"
python lora/dp-lora/train_dp_lora.py \
  --model_id gpt2-medium \
  --target_modules c_attn,c_proj,c_fc \
  --output_dir outputs_eps05/dp_lora_truthy_gpt2M \
  "${COMMON_ARGS[@]}"

echo "[eps05] DP-LoRA gpt2-large"
python lora/dp-lora/train_dp_lora.py \
  --model_id gpt2-large \
  --target_modules c_attn,c_proj,c_fc \
  --output_dir outputs_eps05/dp_lora_truthy_gpt2L \
  "${COMMON_ARGS[@]}"

echo "[eps05] DP-LoRA EleutherAI/pythia-1b"
python lora/dp-lora/train_dp_lora.py \
  --model_id EleutherAI/pythia-1b \
  --target_modules query_key_value,dense_h_to_4h,dense_4h_to_h \
  --output_dir outputs_eps05/dp_lora_truthy_pythia1b \
  "${COMMON_ARGS[@]}"

echo "[eps05] DP-LoRA done for all models."
