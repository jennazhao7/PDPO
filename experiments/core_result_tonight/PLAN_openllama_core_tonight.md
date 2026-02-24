# Plan Mode: Tonight Core Experiment (OpenLLaMA 3B + 7B)

## Objective (single-night core result)

Produce a strong minimum result answering:

**Does RR-SB outperform RR-MLE on PKU-SafeRLHF under the same RR budget?**

Use:
- `openlm-research/open_llama_3b_v2`
- `openlm-research/open_llama_7b_v2`

## Constraints and current storage

- Current `/users` free space observed: **~12G free / 100G total**.
- Existing experiment outputs already use ~1.2G (`outputs` + `outputs_eps05`).
- We must keep run artifacts compact and avoid duplicated checkpoints.

## Storage policy for tonight

- Root: `outputs_openllama_tonight/`
- Save adapter-only outputs (no full-model saves).
- Keep max 1 checkpoint per run or disable intermediate checkpoints where safe.
- Use 10K train / 1K test subset from PKU-SafeRLHF.
- Use low generation length for eval.

## Phase 0 - Data secure snapshot (already prepared)

Script:
- `experiments/core_result_tonight/01_secure_pku_saferlhf.py`

Run:
```bash
python experiments/core_result_tonight/01_secure_pku_saferlhf.py \
  --dataset_id PKU-Alignment/PKU-SafeRLHF \
  --dataset_config default \
  --output_dir data/pku_saferlhf_secure \
  --train_rows 10000 \
  --test_rows 1000
```

Outputs:
- `data/pku_saferlhf_secure/train_pref.jsonl`
- `data/pku_saferlhf_secure/test_pref.jsonl`
- `data/pku_saferlhf_secure/manifest.json`

## Phase 1 - OpenLLaMA-3B core trio (must finish tonight)

Runs (seed 42):
1. Plain LoRA (upper bound)
2. RR-MLE (stage1 RR -> stage2 MLE)
3. RR-SB (stage1 RR -> stage2 Soft-Bayes)

Notes:
- OpenLLaMA tokenizer safety is enabled in training scripts (auto `use_fast=False`).
- This 3B trio is the primary decision signal.

## Phase 2 - OpenLLaMA-7B fast confirmation

Runs (seed 42):
1. RR-MLE
2. RR-SB

Optional:
- Plain LoRA 7B if time/GPU budget remains.

## Phase 3 - Evaluation (same prompt list, paired comparison)

Compare at minimum:
- 3B: RR-SB vs RR-MLE
- 7B: RR-SB vs RR-MLE

Preferred add-ons:
- 3B: RR-SB vs Plain
- 3B: RR-MLE vs Plain

Use cost controls:
- high-signal prompt mode
- `max_new_tokens <= 96`
- single judge vote

## Phase 4 - One-page result artifact

Produce:
- `results_openllama_tonight_summary.json`
- table: win/tie/loss + reward means
- short recommendation: proceed / debug / scale

## Resume strategy (disconnect-safe)

- Launch each phase with `nohup`.
- Keep logs under `outputs_openllama_tonight/logs/`.
- Phase scripts should skip if target artifact already exists.

## Stop conditions

- If free space `< 4G`, pause new runs and prune old checkpoints first.
- If 7B OOM, reduce batch and increase gradient accumulation before retry.
