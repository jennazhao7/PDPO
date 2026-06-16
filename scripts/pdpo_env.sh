#!/usr/bin/env bash
# Source from run scripts: source "$(dirname "$0")/scripts/pdpo_env.sh" or similar.
# Sets PDPO_ROOT to repo root; falls back from CRC path for portability.
if [[ -z "${PDPO_ROOT:-}" ]]; then
  if [[ -d "${HOME}/PDPO" ]]; then
    export PDPO_ROOT="${HOME}/PDPO"
  elif [[ -d "/users/jzhao7/PDPO" ]]; then
    export PDPO_ROOT="/users/jzhao7/PDPO"
  else
    _pdpo_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    export PDPO_ROOT="$_pdpo_env_dir"
  fi
fi
export HF_HOME="${HF_HOME:-${PDPO_ROOT}/.cache/huggingface}"
