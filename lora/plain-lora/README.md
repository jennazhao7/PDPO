# Plain LoRA Fine-Tuning Baseline

Non-DP LoRA fine-tuning on preference (DPO) or instruction (SFT) datasets.
Produces a reproducible baseline directly comparable to a DP-LoRA run using
the same train/val splits, tokenisation, and evaluation metrics.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Smoke Test (DPO, ~30 seconds)

```bash
python train_plain_lora.py \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --output_dir outputs/smoke_test \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --max_seq_length 256 \
  --target_modules c_attn,c_proj,c_fc \
  --seed 42
```

## Full DPO Run (GPT-2 Medium, Truthy-DPO D1)

```bash
python train_plain_lora.py \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --output_dir outputs/gpt2m_dpo_baseline \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --max_seq_length 512 \
  --lora_r 16 \
  --lora_alpha 32 \
  --target_modules c_attn,c_proj,c_fc \
  --fp16 \
  --gradient_checkpointing \
  --seed 42
```

## Using a Config File

```bash
python train_plain_lora.py --config configs/gpt2_medium_dpo.json
```

CLI arguments override values from the config file.

## Outputs

| File | Description |
|------|-------------|
| `adapter_model.safetensors` | LoRA weights |
| `adapter_config.json` | PEFT config |
| `config.json` | Full run config + git hash |
| `split_indices.json` | Train/val indices (for DP fair comparison) |
| `metrics.jsonl` | Per-step training metrics |
| `summary.json` | Final metrics (pairwise accuracy, loss, etc.) |
| `generations.jsonl` | Fixed-prompt generation samples |

## Fair Comparison with DP-LoRA

To ensure an apples-to-apples comparison:

1. **Same splits**: Load `split_indices.json` from the non-DP run and apply
   the same train/val partition in your DP script.
2. **Same tokenisation**: Use the same `--model_id`, `--max_seq_length`, and
   `--target_modules`.
3. **Same eval**: The `summary.json` uses identical metric names
   (`pairwise_accuracy`, `chosen_logp_mean`, `rejected_logp_mean`,
   `train_loss_final`) across both runs.
4. **Same eval prompts**: The default 8-prompt generation set is baked in;
   override with `--eval_prompts_json` if needed.

## Dataset Format

**DPO (preference pairs)**:
```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

**SFT (instruction/response)**:
```json
{"prompt": "...", "response": "..."}
```

The script auto-detects the format from column names.
