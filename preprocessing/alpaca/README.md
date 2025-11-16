# Alpaca Dataset Preprocessing

This folder contains scripts for preprocessing Alpaca datasets with margin-aware label flipping.

## Workflow

The preprocessing pipeline consists of 4 main steps:

### Step 0: Convert Alpaca Eval to DPO Format (if needed)

If you're using `alpaca_eval` or similar evaluation datasets that don't have chosen/rejected pairs, first convert them:

**Usage:**
```bash
python convert_alpaca_eval_to_dpo.py --dataset tatsu-lab/alpaca_eval --split eval --output_file alpaca_eval_dpo.jsonl
```

**Arguments:**
- `--dataset`: Alpaca eval dataset name (default: `tatsu-lab/alpaca_eval`)
- `--split`: Dataset split (default: `eval`)
- `--output_file`: Output JSONL file (default: `alpaca_eval_dpo.jsonl`)
- `--reference_generator`: Reference generator to use as baseline (default: `gpt4`)
- `--min_pairs`: Minimum outputs per instruction (default: 2)

### Step 1: Score with Reward Model (`score_alpaca_rm.py`)
- Loads an Alpaca DPO dataset (or pre-converted DPO file)
- Scores responses using `OpenAssistant/reward-model-deberta-v3-large-v2`
- Computes margins (chosen_score - rejected_score)
- Saves scored dataset to `./alpaca_with_margins/`

**Usage:**
```bash
# Using a DPO-formatted dataset
python score_alpaca_rm.py --dataset <dpo_dataset> --split train --batch_size 32

# Using a pre-converted DPO file
python score_alpaca_rm.py --dpo_file alpaca_eval_dpo.jsonl --batch_size 32
```

**Arguments:**
- `--dataset`: Alpaca dataset name or path (default: `tatsu-lab/alpaca_eval`)
- `--split`: Dataset split to use (default: `train`)
- `--batch_size`: Batch size for scoring (default: 32)
- `--output_dir`: Output directory (default: `./alpaca_with_margins`)
- `--dpo_file`: Optional pre-converted DPO JSONL file

### Step 2: Process Margins (`process_alpaca_margins.py`)
- Loads the scored dataset
- Clips margins to [-6, 6] (configurable)
- Normalizes margins (zero mean, unit variance)
- Saves processed dataset to `./alpaca_with_processed_margins/`

**Usage:**
```bash
python process_alpaca_margins.py --input_dir ./alpaca_with_margins --output_dir ./alpaca_with_processed_margins
```

**Arguments:**
- `--input_dir`: Input directory with scored dataset (default: `./alpaca_with_margins`)
- `--output_dir`: Output directory (default: `./alpaca_with_processed_margins`)
- `--clip_min`: Minimum value for clipping (default: -6.0)
- `--clip_max`: Maximum value for clipping (default: 6.0)

### Step 3: Privatize Labels (`privatize_alpaca_labels.py`)
- Loads the processed dataset
- Performs margin-aware label flipping using sigmoid: `p_keep = sigmoid(eps_labels * margin)`
- Creates DPO-ready dataset with flipped labels
- Saves to `alpaca_privatized_dataset.jsonl` and summary to `alpaca_privatized_dataset_summary.json`

**Usage:**
```bash
python privatize_alpaca_labels.py --input_dir ./alpaca_with_processed_margins --eps_labels 1.0
```

**Arguments:**
- `--input_dir`: Input directory with processed margins (default: `./alpaca_with_processed_margins`)
- `--output_file`: Output JSONL file (default: `alpaca_privatized_dataset.jsonl`)
- `--eps_labels`: Privacy parameter for label flipping (default: 1.0)
- `--seed`: Random seed for reproducibility (default: 42)

## Quick Start: Run All Steps

Use the master script to run all steps in sequence:

```bash
# For alpaca_eval (needs conversion first)
bash run_alpaca_privatization.sh [dataset_name] [split] [eps_labels]

# Or manually:
# Step 0: Convert to DPO format
python convert_alpaca_eval_to_dpo.py --dataset tatsu-lab/alpaca_eval --split eval

# Step 1: Score with reward model
python score_alpaca_rm.py --dpo_file alpaca_eval_dpo.jsonl

# Step 2: Process margins
python process_alpaca_margins.py

# Step 3: Privatize labels
python privatize_alpaca_labels.py --eps_labels 1.0
```

## Configuration

- **Reward Model**: `OpenAssistant/reward-model-deberta-v3-large-v2`
- **Batch Size**: 32 (adjust based on GPU memory)
- **Epsilon (eps_labels)**: 1.0 (default, configurable)
- **Margin Clipping**: [-6, 6] (configurable)

## Output Files

- `alpaca_eval_dpo.jsonl` - Converted DPO dataset (if using conversion)
- `./alpaca_with_margins/` - Scored dataset with raw margins
- `./alpaca_with_processed_margins/` - Dataset with processed margins
- `alpaca_privatized_dataset.jsonl` - Privatized DPO dataset
- `alpaca_privatized_dataset_summary.json` - Summary statistics

## Supported Datasets

### DPO-formatted datasets (can use directly):
- Any Hugging Face dataset with `prompt`, `chosen`, `rejected` columns

### Evaluation datasets (need conversion):
- `tatsu-lab/alpaca_eval` - Alpaca evaluation dataset (has `instruction`, `output`, `generator`)
- Other evaluation datasets with multiple model outputs per instruction

## Notes

- For `alpaca_eval`, you must first convert it to DPO format using `convert_alpaca_eval_to_dpo.py`
- The conversion script pairs outputs from different generators (using GPT-4 as reference by default)
- The reward model scoring step may take time depending on dataset size
- Adjust batch size if you encounter GPU memory issues
- Make sure you have sufficient disk space for intermediate datasets
