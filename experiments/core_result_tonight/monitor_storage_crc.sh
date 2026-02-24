#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/users/jzhao7/PDPO}"
LOG="${2:-$ROOT/outputs_openllama_tonight/storage_monitor.log}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"

mkdir -p "$(dirname "$LOG")"

echo "[monitor] root=$ROOT interval=${INTERVAL_SEC}s log=$LOG"
while true; do
  {
    echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
    df -h "$ROOT"
    du -sh "$ROOT/outputs_openllama_tonight" 2>/dev/null || true
    du -sh "$ROOT/data/pku_saferlhf_secure" 2>/dev/null || true
    du -sh "$ROOT/outputs" "$ROOT/outputs_eps05" 2>/dev/null || true
    echo
  } | tee -a "$LOG"
  sleep "$INTERVAL_SEC"
done
