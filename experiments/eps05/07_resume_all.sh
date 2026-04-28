#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

echo "[eps05] Resume pipeline from current state"
bash experiments/eps05/03_train_stage1_all.sh
bash experiments/eps05/04_train_stage2_all.sh
bash experiments/eps05/05_eval_all.sh
python experiments/eps05/06_report.py --results_root lora/eval/results_eps05
echo "[eps05] Resume pipeline complete"
