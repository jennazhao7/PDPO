#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs_eps05}"
export RESULTS_ROOT="${RESULTS_ROOT:-lora/eval/results_eps05}"
export JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
export PROMPT_MODE="${PROMPT_MODE:-high_signal}"
export N_PROMPTS="${N_PROMPTS:-100}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
export TARGET_MODELS="${TARGET_MODELS:-gpt2M,gpt2L,pythia1b}"

# Plain models remain in outputs/ unless overridden.
export PLAIN_GPT2M="${PLAIN_GPT2M:-outputs/plain_lora_truthy_subset}"
export PLAIN_GPT2L="${PLAIN_GPT2L:-outputs/plain_lora_truthy_gpt2L}"
export PLAIN_PYTHIA1B="${PLAIN_PYTHIA1B:-outputs/plain_lora_truthy_pythia1b}"

# Ensure smoke test uses eps05 paths.
export SMOKE_STAGE2_MANIFEST="${SMOKE_STAGE2_MANIFEST:-${OUTPUT_ROOT}/gpt2-medium-stage2-softbayes/M2_manifest.json}"
export SMOKE_PLAIN_PATH="${SMOKE_PLAIN_PATH:-${PLAIN_GPT2M}}"

mkdir -p "$RESULTS_ROOT"

echo "[eps05] Running eval suite with OUTPUT_ROOT=$OUTPUT_ROOT RESULTS_ROOT=$RESULTS_ROOT"
bash eval/run_all_models_eval.sh
