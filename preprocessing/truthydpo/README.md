# Truthy-DPO Dataset Preprocessing

This folder contains scripts for preprocessing the Truthy-DPO dataset with margin-aware label flipping.

## Workflow

The preprocessing pipeline consists of 3 main steps:

### Step 1: Score with Reward Model (`min_rm.py`)
- Loads the Truthy-DPO dataset
- Scores responses using `OpenAssistant/reward-model-deberta-v3-large-v2`
- Computes margins (chosen_score - rejected_score)
- Saves scored dataset to `./props_with_margins_truthy/`

**Usage:**
```bash
python min_rm.py
```

### Step 2: Process Margins (`process_margins.py`)
- Loads the scored dataset
- Clips margins to [-6, 6]
- Normalizes margins (zero mean, unit variance)
- Saves processed dataset to `./props_with_processed_margins/`

**Usage:**
```bash
python process_margins.py
```

### Step 3: Privatize Labels (`privatize_labels.py`)
- Loads the processed dataset
- Performs margin-aware label flipping using sigmoid: `p_keep = sigmoid(eps_labels * margin)`
- Creates DPO-ready dataset with flipped labels
- Saves to `dpo_privatized_dataset.jsonl` and summary to `privatization_summary.json`

**Usage:**
```bash
python privatize_labels.py
```

### Step 4: Prepare DPO Training Data (`prepare_dpo_data.py`)
- Filters the privatized dataset to keep only required fields (prompt, chosen, rejected)
- Saves clean DPO training dataset to `dpo_train_ready.jsonl`

**Usage:**
```bash
python prepare_dpo_data.py
```

## Testing

Use `test_rm.py` to test the reward model scoring on a small subset:

```bash
python test_rm.py
```

## Configuration

- **Reward Model**: `OpenAssistant/reward-model-deberta-v3-large-v2`
- **Batch Size**: 32 (adjust based on GPU memory)
- **Epsilon (eps_labels)**: 1.0 (default, can be modified in `privatize_labels.py`)
- **Margin Clipping**: [-6, 6]

## Output Files

- `./props_with_margins_truthy/` - Scored dataset with raw margins
- `./props_with_processed_margins/` - Dataset with processed margins
- `dpo_privatized_dataset.jsonl` - Privatized DPO dataset
- `dpo_train_ready.jsonl` - Clean DPO training dataset
- `privatization_summary.json` - Summary statistics

## Notes

- All scripts assume the dataset is in DPO format with `prompt`, `chosen`, and `rejected` columns
- The reward model scoring step may take time depending on dataset size
- Adjust batch size in `min_rm.py` if you encounter GPU memory issues

