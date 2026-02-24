# Core Result Tonight: Data Secure Step

This folder contains the first step from your plan:

1. Secure and snapshot `PKU-Alignment/PKU-SafeRLHF`

## Run

```bash
cd /users/jzhao7/PDPO
conda activate pdpo

python experiments/core_result_tonight/01_secure_pku_saferlhf.py \
  --dataset_id PKU-Alignment/PKU-SafeRLHF \
  --dataset_config default \
  --output_dir data/pku_saferlhf_secure \
  --train_rows 10000 \
  --test_rows 1000
```

## Outputs

- `data/pku_saferlhf_secure/raw_hf_dataset/`
- `data/pku_saferlhf_secure/train_pref.jsonl`
- `data/pku_saferlhf_secure/test_pref.jsonl`
- `data/pku_saferlhf_secure/manifest.json`

## Why this is "secure"

- Keeps a raw immutable snapshot (`save_to_disk`) for auditability.
- Creates normalized preference files for training/eval.
- Stores provenance and SHA-256 hashes in `manifest.json`.

