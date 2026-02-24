# DP-LoRA Fine-Tuning

Differentially private LoRA fine-tuning on preference (DPO) or instruction (SFT)
datasets. Produces results directly comparable to the plain-LoRA baseline using
identical dataset splits, tokenisation, and evaluation metrics.

## Setup

```bash
pip install -r requirements.txt
```

## Quick Smoke Test (single seed, few steps)

```bash
python train_dp_lora.py \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --output_dir outputs/dp_smoke_test \
  --dp \
  --noise_multiplier 1.0 \
  --max_grad_norm 1.0 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --eval_every_steps 3 \
  --target_modules c_attn,c_proj,c_fc \
  --seed 42
```

This produces in `outputs/dp_smoke_test/`:
- `run_manifest.json` — git hash, CLI args, model, dataset, DP/LoRA/trainer params
- `metrics.jsonl` — per-step train + eval records
- `summary.json` — final metrics
- `convergence_summary.json` — best/final metric, steps-to-95%, AUC
- `eval_metric_vs_steps.png` — convergence plot

## Running 5 seeds (multi-seed + aggregation)

```bash
python run_multiseed.py \
  --seeds 0 1 2 3 4 \
  --base_output_dir outputs/dp_multiseed \
  -- \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --dp --noise_multiplier 1.0 --max_grad_norm 1.0 \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --logging_steps 1 \
  --eval_every_steps 3 \
  --target_modules c_attn,c_proj,c_fc
```

This produces in `outputs/dp_multiseed/`:
- `seed_0/`, `seed_1/`, ..., `seed_4/` — individual run directories (each contains the full output set)
- `aggregate_summary.json` — mean/std of best_eval_metric, auc, steps_to_95%, final metric
- `aggregate_plot.png` — mean eval curve with ±1 std shaded band

To aggregate existing runs without re-training:

```bash
python run_multiseed.py \
  --seeds 0 1 2 3 4 \
  --base_output_dir outputs/dp_multiseed \
  --skip_training \
  --
```

## Full DP Run (target epsilon)

```bash
# Step 1: Run plain-LoRA baseline first to get split_indices.json
python ../plain-lora/train_plain_lora.py \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --output_dir outputs/plain_baseline \
  --target_modules c_attn,c_proj,c_fc \
  --seed 42

# Step 2: Run DP-LoRA with the SAME split
python train_dp_lora.py \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --output_dir outputs/dp_eps8 \
  --dp \
  --epsilon 8.0 \
  --max_grad_norm 1.0 \
  --split_indices_json outputs/plain_baseline/split_indices.json \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 5e-5 \
  --eval_every_steps 50 \
  --target_modules c_attn,c_proj,c_fc \
  --seed 42
```

## Full DP Run (explicit noise_multiplier)

```bash
python train_dp_lora.py \
  --model_id gpt2-medium \
  --dataset_path ../../lora/preprocessing/d1_rr_flipped.jsonl \
  --output_dir outputs/dp_sigma1 \
  --dp \
  --noise_multiplier 1.0 \
  --max_grad_norm 1.0 \
  --split_indices_json outputs/plain_baseline/split_indices.json \
  --eval_every_steps 50 \
  --target_modules c_attn,c_proj,c_fc \
  --seed 42
```

## Using Config Files

```bash
python train_dp_lora.py --config configs/gpt2_medium_dp_eps8.json
```

## Choosing (epsilon, delta) vs (noise_multiplier, max_grad_norm)

### Mode 1: Target epsilon (recommended for beginners)

Pass `--epsilon <value>` and `--max_grad_norm <value>`. The script auto-computes
the required `noise_multiplier` using Opacus's RDP accountant. Delta defaults to
`1/N` if not provided.

```bash
--dp --epsilon 8.0 --max_grad_norm 1.0
```

### Mode 2: Explicit noise_multiplier (for experts)

Pass `--noise_multiplier <sigma>` and `--max_grad_norm <C>` directly. The
resulting epsilon is computed and logged after training.

```bash
--dp --noise_multiplier 1.0 --max_grad_norm 1.0
```

### Tuning guidance

| Parameter | Effect |
|-----------|--------|
| `max_grad_norm` (C) | Higher = less clipping but more noise needed. Start with 1.0. |
| `noise_multiplier` (sigma) | Higher = more noise = stronger privacy = worse utility. |
| `batch_size` (L) | Larger batches improve the signal/noise ratio per step. |
| `epsilon` | Lower = stronger privacy. Typical research range: 1-10. |
| `delta` | Should be << 1/N. Default 1/N is standard. |

## Privacy Accounting Assumptions

- **Sampling**: Fixed-size shuffled batches treated under Poisson sub-sampling
  assumption at rate `q = batch_size / N`. This is standard practice and
  provides an upper bound on the true privacy cost.
- **Composition**: RDP (Renyi Differential Privacy) accountant by default.
  PRV accountant available via `--accountant prv`.
- **Noise**: Gaussian mechanism with `std = sigma * C` added to the sum of
  clipped gradients before averaging.
- **LoRA dropout**: Automatically disabled under DP (dropout + DP noise
  can interfere with accounting).
- **Precision**: Training runs in fp32 under DP to avoid numerical issues
  with gradient clipping. The script auto-falls back from fp16/bf16 with a
  warning.

## Output Directory Layout

### Single run (`train_dp_lora.py`)

```
outputs/dp_run/
├── adapter_model.safetensors   # LoRA weights
├── adapter_config.json         # PEFT config
├── run_manifest.json           # Full run config, git hash, data stats, DP/LoRA/trainer params
├── split_indices.json          # Train/val indices (load via --split_indices_json for fair comparison)
├── trainable_params.json       # Trainable parameter names (for DP/non-DP comparison)
├── metrics.jsonl               # Per-step train logs + periodic eval records
├── summary.json                # Final metrics + DP fields
├── convergence_summary.json    # best/final metric, steps_to_95%, AUC
├── eval_metric_vs_steps.png    # Convergence plot (metric + train loss)
└── generations.jsonl           # Fixed-prompt generation samples
```

### Multi-seed (`run_multiseed.py`)

```
outputs/dp_multiseed/
├── seed_0/                     # Full single-run output for seed 0
├── seed_1/
├── ...
├── aggregate_summary.json      # Mean/std of best_eval_metric, auc, steps_to_95%, final metric
└── aggregate_plot.png          # Mean eval curve ± 1 std across seeds
```

### Key fields in `metrics.jsonl`

Each line is a JSON object. Training-step records contain:
- `step`, `epoch`, `loss`, `grad_norm`, `lr`, `examples_seen`, `effective_batch_size`
- DP-only: `epsilon_spent`, `noise_multiplier`, `clipping_fraction`

Eval records (logged at `--eval_every_steps` and at end) contain:
- `step`, `epoch`, `is_eval: true`
- `eval_pairwise_accuracy`, `eval_chosen_logp_mean`, `eval_rejected_logp_mean`, `eval_n`
- `eval_loss`

### Key fields in `aggregate_summary.json`

- `best_eval_metric`: `{mean, std, n}`
- `final_eval_metric`: `{mean, std, n}`
- `auc_eval_metric_vs_steps`: `{mean, std, n}`
- `steps_to_95pct_best`: `{mean, std, n}`

## Verifying DP vs Non-DP Comparison

The `trainable_params.json` files from both runs should have identical parameter
names. Use `split_indices.json` from the plain run to guarantee identical data.
Compare `summary.json` fields: `pairwise_accuracy`, `chosen_logp_mean`,
`rejected_logp_mean` are available in both.
