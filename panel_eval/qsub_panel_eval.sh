#!/usr/bin/env bash
# qsub_panel_eval.sh
# ==================
# SGE/PBS qsub wrapper for the panel eval pipeline.
# Requests one GPU node; Con-J loads on GPU, API judges are CPU-only.
#
# Usage:
#   qsub qsub_panel_eval.sh
#   or for smoke test (50 prompts, con_j only):
#   qsub -v SMOKE_TEST=1 qsub_panel_eval.sh

#$ -N panel_eval
#$ -l gpu=1
#$ -l h_rt=6:00:00
#$ -l h_vmem=48G
#$ -j y
#$ -cwd

set -euo pipefail

echo "[qsub] Job started: $(date)"
echo "[qsub] Host: $(hostname)"

# Activate your conda/venv environment if needed:
# source /path/to/your/env/bin/activate
# conda activate pdpo

SCRIPT_DIR="/users/jzhao7/PDPO/panel_eval"

# Smoke-test override: 50 prompts, con_j only
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    echo "[qsub] SMOKE TEST MODE: 50 prompts, con_j only"
    # Temporarily patch run_panel_eval to use N=50
    export PANEL_N_PROMPTS_OVERRIDE=50
    export PANEL_JUDGES_OVERRIDE="con_j"
fi

bash "${SCRIPT_DIR}/run_panel_eval.sh"

echo "[qsub] Job finished: $(date)"
