# Parallel GPU Runbook (Independent Tasks)

This replaces the sequential overnight runner with independent task scripts.
Each task can run on a different GPU / shell.

## 0) One-time data prep task

```bash
cd /users/jzhao7/PDPO
bash experiments/core_result_tonight/run_data_prep_task.sh
```

## 1) Checkpoint status

```bash
bash experiments/core_result_tonight/checkpoint_status.sh
```

## 2) Run independent tasks in parallel

Examples below assume two GPUs (`0`, `1`) and detached logging.

### A. Resume failed OpenLLaMA-3B Soft-Bayes only

```bash
cd /users/jzhao7/PDPO
mkdir -p outputs_openllama_tonight/logs
GPU_ID=0 nohup bash experiments/core_result_tonight/run_stage2_sb_task.sh \
  openlm-research/open_llama_3b_v2 \
  > outputs_openllama_tonight/logs/openllama3b_stage2_sb.log 2>&1 &
```

### B. Start OpenLLaMA-7B Stage1 on another GPU

```bash
cd /users/jzhao7/PDPO
GPU_ID=1 nohup bash experiments/core_result_tonight/run_stage1_task.sh \
  openlm-research/open_llama_7b_v2 \
  > outputs_openllama_tonight/logs/openllama7b_stage1.log 2>&1 &
```

### C. Then launch 7B Stage2 MLE / SB when Stage1 completes

```bash
cd /users/jzhao7/PDPO
GPU_ID=1 nohup bash experiments/core_result_tonight/run_stage2_mle_task.sh \
  openlm-research/open_llama_7b_v2 \
  > outputs_openllama_tonight/logs/openllama7b_stage2_mle.log 2>&1 &

GPU_ID=0 nohup bash experiments/core_result_tonight/run_stage2_sb_task.sh \
  openlm-research/open_llama_7b_v2 \
  > outputs_openllama_tonight/logs/openllama7b_stage2_sb.log 2>&1 &
```

## 3) Monitor

```bash
tail -f outputs_openllama_tonight/logs/openllama3b_stage2_sb.log
tail -f outputs_openllama_tonight/logs/openllama7b_stage1.log
bash experiments/core_result_tonight/checkpoint_status.sh
```

## Notes

- Scripts auto-skip if artifact already exists.
- For OpenLLaMA, stage2 scripts use lower max length defaults for RTX 6000 stability.
- Override via env vars if needed:
  - `MAX_STEPS=300`
  - `EPSILON=1.0`
  - `SEED=42`
  - `MAX_LEN=320` (OpenLLaMA SB override)
