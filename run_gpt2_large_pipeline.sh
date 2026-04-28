#!/bin/bash
# Full gpt2-large pipeline: stage1 (from HF) -> stage2 MLE -> stage2 Soft-Bayes
# Also plain-lora and eval.
#
# Prerequisites:
#   - gpt2-large stage1 on HF (Jennazhao7/pdpo-lora, gpt2-large/truthydpo/stage1/eps_1.0)
#   - Download stage1: huggingface-cli download Jennazhao7/pdpo-lora \\
#       gpt2-large/truthydpo/stage1/eps_1.0 --local-dir models/
#     This creates models/gpt2-large/truthydpo/stage1/eps_1.0/
#
# Usage:
#   bash run_gpt2_large_pipeline.sh [stage1|plain|mle|softbayes|eval]
#   (no args = run all)

set -e
cd "$(dirname "$0")"

STAGE1_DIR="models/gpt2-large/truthydpo/stage1/eps_1.0"
[[ -d "$STAGE1_DIR" ]] || STAGE1_DIR="models/gpt2-large-truthy-stage1-rr-eps1"

run_stage1_download() {
  echo "=== Downloading gpt2-large stage1 from HF ==="
  huggingface-cli download Jennazhao7/pdpo-lora \
    gpt2-large/truthydpo/stage1/eps_1.0 \
    --local-dir models/ \
    --local-dir-use-symlinks False
  echo "Stage1 at: models/gpt2-large/truthydpo/stage1/eps_1.0"
}

run_plain() {
  echo "=== Plain-LoRA gpt2-large ==="
  python lora/plain-lora/train_plain_lora.py \
    --model_id gpt2-large \
    --dataset_path preprocessing/truthydpo/truthy_dpo_subset.jsonl \
    --output_dir outputs/plain_lora_truthy_gpt2L \
    --target_modules c_attn,c_proj,c_fc \
    --max_steps 100 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --seed 42
}

run_mle() {
  echo "=== Stage2 MLE gpt2-large ==="
  [[ -d "$STAGE1_DIR" ]] || { echo "Need stage1 at $STAGE1_DIR. Run stage1 download first."; exit 1; }
  python lora/train_stage2_lora.py \
    --model gpt2-large \
    --stage1_adapter "$STAGE1_DIR" \
    --data lora/preprocessing/d2_rr_flipped.jsonl \
    --out outputs/gpt2-large-stage2-mle \
    --target_modules c_attn,c_proj,c_fc \
    --max_steps 300
}

run_softbayes() {
  echo "=== Stage2 Soft-Bayes gpt2-large ==="
  [[ -d "$STAGE1_DIR" ]] || { echo "Need stage1 at $STAGE1_DIR. Run stage1 download first."; exit 1; }
  python lora/train_stage2_soft_bayes.py \
    --model gpt2-large \
    --stage1_adapter "$STAGE1_DIR" \
    --data lora/preprocessing/d2_rr_flipped.jsonl \
    --out outputs/gpt2-large-stage2-softbayes \
    --target_modules c_attn,c_proj,c_fc \
    --max_steps 300
}

run_eval() {
  echo "=== Eval: plain-LoRA vs rr+Soft-Bayes gpt2-large ==="
  python eval/eval_compare.py \
    --model_a_type lora \
    --model_a_path outputs/plain_lora_truthy_gpt2L \
    --model_a_base gpt2-large \
    --model_a_label "plain-lora-gpt2L" \
    --model_b_type stage2 \
    --model_b_manifest outputs/gpt2-large-stage2-softbayes/M2_manifest.json \
    --model_b_label "rr+SoftBayes-stage2-gpt2L" \
    --prompts_jsonl preprocessing/truthydpo/truthy_dpo_subset.jsonl \
    --n_prompts 100 \
    --out_dir lora/eval/results/plain_vs_stage2_softbayes_gpt2L
}

case "${1:-}" in
  stage1) run_stage1_download ;;
  plain) run_plain ;;
  mle) run_mle ;;
  softbayes) run_softbayes ;;
  eval) run_eval ;;
  "")
    run_stage1_download
    run_plain
    run_mle
    run_softbayes
    run_eval
    ;;
  *) echo "Usage: $0 [stage1|plain|mle|softbayes|eval]"; exit 1 ;;
esac
