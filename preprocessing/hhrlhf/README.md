# HH-RLHF Dataset Preprocessing

This folder contains scripts for preprocessing the Anthropic HH-RLHF dataset with reward model scoring.

## Workflow

The preprocessing pipeline consists of multiple steps:

### Step 1: Score with Reward Model (`score_rm.py`)
- Loads the HH-RLHF dataset from HuggingFace (`Anthropic/hh-rlhf`)
- Extracts prompts and responses from the dataset format
- Scores responses using `OpenAssistant/reward-model-deberta-v3-large-v2`
- Computes margins (chosen_score - rejected_score)
- Saves scored dataset to `./props_with_margins_hhrlhf/`

**Usage:**
```bash
python score_rm.py
```

### Step 2: Process Margins (`process_margins.py`)
- Loads the scored dataset
- Clips margins based on percentiles (default: 5th and 95th percentiles)
- Normalizes margins for DP sensitivity (scales to [-DP_SENSITIVITY, DP_SENSITIVITY] range)
- Saves processed dataset to `./props_with_processed_margins_hhrlhf/`
- Saves clipping parameters to `clipping_info.json`

**Configuration:**
- `LOWER_PERCENTILE`: Lower percentile for clipping (default: 5.0)
- `UPPER_PERCENTILE`: Upper percentile for clipping (default: 95.0)
- `DP_SENSITIVITY`: DP sensitivity parameter for normalization (default: 1.0)

**Usage:**
```bash
python process_margins.py
```

### Step 3: Privatize Labels (`privatize_labels.py`)
- Loads the processed dataset
- Performs DP label flipping using **Binary Exponential Mechanism** (equivalent to Randomized Response with prior)
- Probability of keeping original label: `P(keep | margin) = sigmoid(ε * margin / Δ)`
  - Where ε is the privacy parameter (eps_labels)
  - Δ is the sensitivity (from normalization step)
- Creates DPO-ready dataset with flipped labels
- Saves to `dpo_privatized_dataset.jsonl` and summary to `privatization_summary.json`

**Configuration:**
- `eps_labels`: Privacy parameter ε (default: 1.0)
- `sensitivity`: Sensitivity parameter Δ (automatically read from dataset, default: 1.0)

**Usage:**
```bash
python privatize_labels.py
```

**Note:** The Binary Exponential Mechanism provides ε-differential privacy for label flipping, where the probability of keeping the original label depends on the margin (reward difference).

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

## Dataset Format

HH-RLHF dataset has the following structure:
- `chosen`: Full conversation ending with the chosen response
- `rejected`: Full conversation ending with the rejected response

The script extracts the prompt by finding the last occurrence of `\n\nAssistant:` in the chosen response, and extracts the response text that follows.

## Configuration

- **Reward Model**: `OpenAssistant/reward-model-deberta-v3-large-v2`
- **Batch Size**: 32 (adjust based on GPU memory)
- **Epsilon (eps_labels)**: 1.0 (default, can be modified in `privatize_labels.py`)
- **Margin Clipping**: Percentile-based (default: 5th-95th percentile)
- **DP Sensitivity**: 1.0 (default, controls normalization range)

## Output Files

- `./props_with_margins_hhrlhf/` - Scored dataset with raw margins
- `./props_with_processed_margins_hhrlhf/` - Dataset with processed margins (if process_margins.py is run)
- `clipping_info.json` - Clipping parameters and statistics
- `dpo_privatized_dataset.jsonl` - Privatized DPO dataset (if privatize_labels.py is run)
- `dpo_train_ready.jsonl` - Clean DPO training dataset (if prepare_dpo_data.py is run)
- `privatization_summary.json` - Summary statistics (if privatize_labels.py is run)

## Notes

- The reward model scoring step may take time depending on dataset size
- Adjust batch size in `score_rm.py` if you encounter GPU memory issues
- The HH-RLHF dataset format requires prompt extraction from the full conversation text

