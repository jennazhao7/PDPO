# Evaluation Scripts for GPT-2 Medium DPO Models

## Setup

### 1. Set OpenAI API Key

You need to set your OpenAI API key to use the judge model:

```bash
# Option 1: Export in current session
export OPENAI_API_KEY='your-api-key-here'

# Option 2: Add to ~/.bashrc (permanent)
echo 'export OPENAI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# Option 3: Verify it's set
echo $OPENAI_API_KEY
```

### 2. Activate Environment

```bash
conda activate pdpo
cd /users/jzhao7/PDPO
```

## Usage

### Quick Start (Using Bash Script)

```bash
# Make sure OPENAI_API_KEY is set first!
export OPENAI_API_KEY='your-api-key-here'

# Run the evaluation
bash eval/run_gpt2M_eval.sh
```

### Manual Run (Python Directly)

```bash
# Set API key
export OPENAI_API_KEY='your-api-key-here'

# Run with local model
python eval/truthy_gpt2M_eval.py \
  --model_a ./models/M1_medium \
  --model_b Setpember/Jon_GPT2M_DPO_props_epi_point5 \
  --n 100 \
  --batch_size 8 \
  --judge_model gpt-4o-mini \
  --out_csv eval/props_win_tie_results_gpt2M.csv

# Or with Hugging Face model
python eval/truthy_gpt2M_eval.py \
  --model_a Jennazhao7/gpt2-medium-dpo-m1 \
  --model_b Setpember/Jon_GPT2M_DPO_props_epi_point5 \
  --n 100 \
  --batch_size 8 \
  --out_csv eval/props_win_tie_results_gpt2M.csv
```

## Arguments

- `--model_a`: Your DPO model (local path or HF model ID)
- `--model_b`: Baseline model to compare against
- `--n`: Number of prompts to evaluate (default: 100)
- `--batch_size`: Batch size for generation (default: 8, reduce if OOM)
- `--judge_model`: Judge model to use (default: gpt-4o-mini)
- `--out_csv`: Output CSV file path
- `--n_votes`: Number of votes per judgment (default: 1, use 3 or 5 for majority vote)

## Output

The script generates two files:
1. `props_win_tie_results_gpt2M.csv` - Detailed results with all prompts and judgments
2. `props_win_tie_results_gpt2M_summary.csv` - Summary statistics with win/tie/loss rates

## Troubleshooting

### OpenAI API Key Error
```
openai.OpenAIError: The api_key client option must be set
```
**Solution**: Set `OPENAI_API_KEY` environment variable (see Setup section)

### Out of Memory (OOM)
**Solution**: Reduce `--batch_size` to 4 or 2

### Slow Generation
**Solution**: Increase `--batch_size` to 16 (if you have enough GPU memory)

### Model Loading Errors
**Solution**: Make sure model paths are correct, or models are available on Hugging Face


