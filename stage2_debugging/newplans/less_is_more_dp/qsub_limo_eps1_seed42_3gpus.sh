#!/bin/bash
#$ -N limo_eps1_s42_3g
#$ -j y
#$ -q gpu@@jung_gpu
#$ -l gpu_card=3
#$ -pe smp 12
#$ -l h_rt=24:00:00
#$ -m abe
#$ -M jzhao7@nd.edu

set -eo pipefail
# Some cluster bashrc setups reference unset vars; source safely, then re-enable nounset.
set +u
source ~/.bashrc
set -u
conda activate pdpo

cd /users/jzhao7/PDPO

# Optional overrides at submission:
#   qsub -v MODE=selection_only,TAU_DROP=0.10 qsub_limo_eps1_seed42_3gpus.sh
MODE="${MODE:-both}"                # both | weighting_only | selection_only
TAU_DROP="${TAU_DROP:-0.10}"
EPS="${EPS:-1.0}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B}"
OUT_ROOT="${OUT_ROOT:-/users/jzhao7/PDPO/stage2_debugging/newplans/less_is_more_dp/results_eps${EPS}_seed${SEED}}"

echo "========================================================"
echo "[limo] Job started: $(date)"
echo "[limo] Host: $(hostname)"
echo "[limo] MODE=${MODE} TAU_DROP=${TAU_DROP} EPS=${EPS} SEED=${SEED}"
echo "========================================================"

bash /users/jzhao7/PDPO/stage2_debugging/newplans/less_is_more_dp/run_limo_eps1_seed42_3datasets_3gpus.sh

echo "========================================================"
echo "[limo] Job finished: $(date)"
echo "========================================================"

