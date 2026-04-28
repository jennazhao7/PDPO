#!/bin/bash
# Safe helper script to submit all 3 Stage 2 jobs without file race conditions

echo "Submitting PKU-SafeRLHF..."
qsub -v DATASET="pku" experiments/core_result_tonight/qsub_stage2_4gpus.sh

echo "Submitting HH-RLHF..."
qsub -v DATASET="hhrlhf" experiments/core_result_tonight/qsub_stage2_4gpus.sh

echo "Submitting TruthyDPO..."
qsub -v DATASET="truthy" experiments/core_result_tonight/qsub_stage2_4gpus.sh

echo "All 3 datasets safely submitted to the queue parameterized via -v!"
