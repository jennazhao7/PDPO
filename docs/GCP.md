# Run PDPO on Google Cloud GPUs

This guide uses a **Compute Engine GPU VM + SSH** — the same workflow as the Notre Dame CRC cluster (`qsub` → bash scripts), but on GCP.

## Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) — on macOS with Python 3.9, run `bash scripts/gcp/install_gcloud.sh` first
2. A GCP project with billing enabled and GPU quota (T4 or L4 in `us-central1`)
3. Authenticated CLI:
   ```bash
   gcloud auth login
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

## Quick start (from repo root)

```bash
# 1. Configure (once)
cp scripts/gcp/gcp.env.example scripts/gcp/gcp.env
# Edit scripts/gcp/gcp.env with your project ID

# 2. Verify GPU quota
bash scripts/gcp/verify_quota.sh

# 3. Create VM (1× T4, 200 GB disk, Deep Learning image)
bash scripts/gcp/create_vm.sh

# 4. SSH setup
bash scripts/gcp/setup_ssh.sh
gcloud compute ssh pdpo-gpu --zone=us-central1-a

# 5. Bootstrap PDPO on VM (from Mac, or run bootstrap_vm.sh on VM)
bash scripts/gcp/run_remote_bootstrap.sh

# 6. Smoke test on GPU (fast GPU check; add FULL_TEST=1 for full quick_sb_test)
bash scripts/gcp/run_smoke_test.sh
# FULL_TEST=1 bash scripts/gcp/run_smoke_test.sh   # slow: downloads Qwen2.5-3B + trains stage1

# 7. Stop VM when idle
bash scripts/gcp/stop_vm.sh
```

Or run the full local orchestrator:

```bash
bash scripts/gcp/setup_gcp.sh
```

## Connect from Cursor / VS Code

```bash
gcloud compute config-ssh
```

Then Remote SSH to: `us-central1-a.pdpo-gpu.YOUR_PROJECT_ID` and open `~/PDPO`.

## Run training jobs

After bootstrap, SSH into the VM:

```bash
conda activate pdpo
cd ~/PDPO

# Smoke test
GPU=0 bash experiments/quick_sb_test.sh

# Parallel tasks (see experiments/core_result_tonight/PARALLEL_RUNBOOK.md)
GPU_ID=0 nohup bash experiments/core_result_tonight/run_stage2_sb_task.sh \
  openlm-research/open_llama_3b_v2 \
  > outputs_openllama_tonight/logs/openllama3b_stage2_sb.log 2>&1 &

# Keep jobs alive after disconnect
tmux new -s train
```

Top-level scripts use `$PDPO_ROOT` (see `scripts/pdpo_env.sh`) instead of hardcoded CRC paths.

## VM sizing

| GPU | Machine type | Use case |
|-----|--------------|----------|
| 1× T4 | `n1-standard-8` | GPT-2, Pythia-1B, smoke tests |
| 1× L4 | `g2-standard-8` | Same, faster |
| 4× A100 | `a2-highgpu-4g` | OpenLLaMA 3B/7B, 4-GPU sweeps |

Edit `scripts/gcp/gcp.env` to change `GPU_TYPE`, `GPU_COUNT`, and `MACHINE_TYPE`.

## Data and checkpoints

| Asset | Location |
|-------|----------|
| Code | `git clone` → `~/PDPO` |
| HF models | `~/PDPO/.cache/huggingface` |
| Checkpoints | VM boot disk (lost if VM deleted) |
| Backup | `bash scripts/gcp/backup_outputs.sh` → GCS |

## Cost control

```bash
bash scripts/gcp/stop_vm.sh          # stop compute, keep disk
bash scripts/gcp/start_vm.sh         # resume
bash scripts/gcp/setup_budget_alert.sh   # billing alerts
```

Set budget alerts in [GCP Console → Billing → Budgets](https://console.cloud.google.com/billing/budgets).

Rough costs: 1× T4 ≈ $0.35–0.50/hr; 4× A100 ≈ $12–15/hr.

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/gcp/verify_quota.sh` | Check T4/L4/A100 quota |
| `scripts/gcp/create_vm.sh` | Create `pdpo-gpu` VM |
| `scripts/gcp/setup_ssh.sh` | Configure local SSH |
| `scripts/gcp/bootstrap_vm.sh` | Run on VM: conda env + pip install |
| `scripts/gcp/run_remote_bootstrap.sh` | Bootstrap from Mac via SSH |
| `scripts/gcp/run_smoke_test.sh` | GPU verify on VM; `FULL_TEST=1` for full quick_sb_test |
| `scripts/gcp/verify_gpu.sh` | Fast GPU sanity check (nvidia-smi + torch matmul) |
| `scripts/gcp/prepare_smoke_data.sh` | Prepare data for full quick_sb_test |
| `scripts/gcp/install_gcloud.sh` | Install gcloud on macOS (Python 3.9 workaround) |
| `scripts/gcp/preflight.sh` | Check auth + project before setup |
| `scripts/gcp/setup_gcp.sh` | End-to-end orchestrator |
| `scripts/gcp/stop_vm.sh` / `start_vm.sh` | Cost control |
| `scripts/gcp/backup_outputs.sh` | Sync outputs to GCS |
| `scripts/gcp/setup_budget_alert.sh` | Billing budget helper |
