# PDPO Cost Playbook

This repo is configured for single-GPU, credit-limited GCP runs. The operating principle is simple: use the cheapest viable Spot GPU, do a mandatory smoke test, run idempotent cells, sync results, and delete the VM.

## Configuration

Fill `config.py` via constants or environment variables. The existing `scripts/gcp/gcp.env` values are read by shell scripts through `scripts/gcp/common.sh`; Python tools read the same values from the environment.

Required values:

- `GCP_PROJECT` / `PROJECT_ID`
- `GCP_ZONE` / `ZONE`
- `GCS_BUCKET` / `BUCKET`
- `CREDIT_CEILING_USD`
- `PREFERRED_CARD`, default `L4`

Card mapping in `config.py`:

- `L4`: `g2-standard-8`, `nvidia-l4`, default choice.
- `T4`: `n1-standard-8`, `nvidia-tesla-t4`, cheapest viable card if ref logps are cached.
- `A100`: `a2-highgpu-1g`, `nvidia-tesla-a100`, never default.

Rough rates are planning estimates only. Check current regional pricing before spend.

## Verified GCP Flags

Verified against current Google Cloud `gcloud compute instances create` and GPU VM docs:

- `--provisioning-model=SPOT`
- `--instance-termination-action=DELETE`
- `--accelerator=type=...,count=...`
- `--maintenance-policy=TERMINATE`

GCP GPU docs note that Deep Learning VM images are not supported on G2 VMs, so `IMAGE_FAMILY` / `IMAGE_PROJECT` remain configurable.

## Precompute Reference Logprobs

Run this once per `(dataset, base_model)` before the grid:

```bash
python3 precompute_ref_logps.py
```

It writes local caches under:

```text
experiments/ref_logps/
```

and uploads them to:

```text
${GCS_BUCKET}/ref_logps/
```

`experiments/run_queue.py` detects local caches and exports `PDPO_REF_LOGPS_CACHE` for training/eval commands. Custom method commands can also use `{ref_logps_cache}` as a template variable. Existing trainers that do not yet consume cached ref logps will still run normally, but the cache is available for memory-optimized trainer variants.

## Mandatory Preflight

Before launch:

```bash
python3 preflight_cost.py
```

It counts unfinished manifest cells, estimates:

```text
remaining_cells * hours_per_cell * PREFERRED_CARD spot_rate
```

then adds spend-to-date from `cost_ledger.csv`. It refuses to continue if projected total exceeds `CREDIT_CEILING_USD`, and it requires typed `yes`.

## Launch

Use:

```bash
./launch.sh
```

The launcher:

1. Runs `preflight_cost.py`.
2. Creates a one-GPU Spot VM with `--instance-termination-action=DELETE`.
3. Uses a small boot disk.
4. Pulls cached data/results from the bucket.
5. Starts `idle_watchdog.py`.
6. Runs `python3 experiments/smoke.py`.
7. Runs `python3 experiments/run_queue.py`.
8. Pushes results and `cost_ledger.csv` to the bucket.
9. Deletes the VM.

For non-interactive launches after reviewing the estimate:

```bash
./launch.sh --yes
```

A100 is blocked unless:

```bash
./launch.sh --force-a100
```

and the confirmation phrase is typed. Do not use A100 for the default PDPO grid.

## Idle Watchdog

`idle_watchdog.py` polls `nvidia-smi`. If GPU utilization remains below 5% for 10 minutes, it shuts the VM down.

Manual run:

```bash
python3 idle_watchdog.py --log idle_watchdog.log
```

This is the main defense against idle billing.

## Idempotent Resume

`experiments/run_queue.py` skips any valid existing result JSON unless `--force` is used. Spot preemption is handled by rerunning the queue: completed cells are skipped, failed/incomplete cells resume.

Each completed cell appends one row to:

```text
cost_ledger.csv
```

Fields:

- `cell`
- `method`
- `dataset`
- `eps`
- `seed`
- `card`
- `start`
- `end`
- `gpu_hours`
- `est_cost`
- `result_json`

## Storage Hygiene

Keep:

- LoRA adapter weights
- result JSONs
- cost ledger
- ref-logp caches

Do not keep:

- merged 3B model weights
- repeated full-model downloads
- intermediate checkpoints after successful cells

`run_queue.py` deletes `checkpoint-*` directories after a successful cell and refuses full-model artifacts like `pytorch_model*.bin` or `model*.safetensors` in the cell output directory.

## Reports

Regenerate experiment summary:

```bash
python3 experiments/aggregate.py
```

Regenerate decision report:

```bash
python3 experiments/decision_report.py
```

Both reports include cumulative estimated spend from `cost_ledger.csv` and remaining credit against `CREDIT_CEILING_USD`.

Do not use a GPU VM for aggregation, editing, or reading results. Those are CPU/local tasks.

## Smoke Gate

`launch.sh` always runs:

```bash
python3 experiments/smoke.py
```

before the real queue. This checks every method on `n=8`, one tiny training step, schema-valid result output, aggregation, and decision reporting. If smoke fails, the batch aborts before full-grid spend.
