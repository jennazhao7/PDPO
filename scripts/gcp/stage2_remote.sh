#!/usr/bin/env bash
# On-VM bootstrap for Stage 2 RR arms. Mirrors stage1_remote.sh.
set -euo pipefail
ROOT="${HOME}/PDPO"; OUT_DIR="${ROOT}/experiments/stage2_rr"
BUCKET_URI="${GCS_BUCKET}"; [[ "${BUCKET_URI}" == gs://* ]] || BUCKET_URI="gs://${BUCKET_URI}"
GCS_OUT="${BUCKET_URI}/experiments/stage2_rr"
mkdir -p "${ROOT}/experiments" "${OUT_DIR}"; cd "${ROOT}"
log() { echo "[$(date -Is)] $*" | tee -a "${OUT_DIR}/remote.log"; }

# --- periodic log sync -------------------------------------------------------------------
# Previously remote.log and startup.log reached the bucket only at teardown or on error, so
# during a live run both were empty in GCS and a wedged process was indistinguishable from
# slow training. These now go up on a fixed interval, independent of commit points, together
# with a heartbeat that records how long the log has been quiet and what the GPU is doing.
LOG_SYNC_SECONDS="${LOG_SYNC_SECONDS:-120}"
LOG_SYNC_PID=""

sync_logs() {
  local f
  for f in remote.log startup.log status.json idle_watchdog.log idle_watchdog.stdout heartbeat.json; do
    [[ -s "${OUT_DIR}/${f}" ]] || continue
    gcloud storage cp "${OUT_DIR}/${f}" "${GCS_OUT}/${f}" >/dev/null 2>&1 || true
  done
}

write_heartbeat() {
  # JSON is assembled by python from argv, not by a shell heredoc: the values include a raw log
  # line, which needs real escaping, and a heredoc closing brace at column 0 also defeats naive
  # function extraction in tests.
  local now log_mtime quiet remote_mtime remote_quiet util mem last
  now="$(date -u +%s)"
  # startup.log, NOT remote.log. remote.log receives only phase-boundary messages from log(), so
  # once run_stage2_rr.py takes over its stdout goes to startup.log and remote.log stops moving.
  # Statting remote.log therefore reported hours of "quiet" during entirely healthy training,
  # which made this field useless as the wedge detector it exists to be.
  log_mtime="$(stat -c %Y "${OUT_DIR}/startup.log" 2>/dev/null || echo "${now}")"
  quiet=$(( now - log_mtime ))
  remote_mtime="$(stat -c %Y "${OUT_DIR}/remote.log" 2>/dev/null || echo "${now}")"
  remote_quiet=$(( now - remote_mtime ))
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)"
  last="$(tail -1 "${OUT_DIR}/remote.log" 2>/dev/null || true)"
  python3 - "${OUT_DIR}/heartbeat.json" "${VM_NAME:-unknown}" "${GCP_ZONE:-unknown}" \
    "${quiet}" "${LOG_SYNC_SECONDS}" "${util:-unknown}" "${mem:-unknown}" "${last}" \
    "${remote_quiet}" <<'HEARTBEAT_PY'
import json, sys
from datetime import datetime, timezone
path, vm, zone, quiet, interval, util, mem, last, remote_quiet = sys.argv[1:10]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "vm": vm,
        "zone": zone,
        # Seconds since startup.log -- the runner's live stdout -- was last written. A large
        # value with the GPU idle is a wedged process; a large value with the GPU busy is slow
        # training. That distinction is the whole point of syncing on an interval rather than
        # only at commit points.
        "log_quiet_seconds": int(quiet),
        "log_quiet_source": "startup.log",
        # Kept for reference only: expected to be large and harmless during training.
        "remote_log_quiet_seconds": int(remote_quiet),
        "sync_interval_seconds": int(interval),
        "gpu_utilization_pct": util,
        "gpu_memory_used_mib": mem,
        "last_remote_log_line": last,
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
HEARTBEAT_PY
}

start_log_syncer() {
  # The subshell inherits set -e, so both calls are guarded: a single transient failure must not
  # silently kill the syncer and take the run's only live visibility with it.
  ( while true; do
      sleep "${LOG_SYNC_SECONDS}"
      write_heartbeat || true
      sync_logs || true
    done ) >/dev/null 2>&1 &
  LOG_SYNC_PID=$!
  log "log syncer started pid=${LOG_SYNC_PID} interval=${LOG_SYNC_SECONDS}s -> ${GCS_OUT}"
}

stop_log_syncer() {
  [[ -n "${LOG_SYNC_PID}" ]] || return 0
  kill "${LOG_SYNC_PID}" 2>/dev/null || true
  wait "${LOG_SYNC_PID}" 2>/dev/null || true
  LOG_SYNC_PID=""
}

write_status() {
  local status="$1" signature="${2:-}" completed=0
  [[ -f "${OUT_DIR}/ledger.jsonl" ]] && completed="$(grep -c . "${OUT_DIR}/ledger.jsonl" | tr -d ' ')"
  python3 - "${OUT_DIR}/status.json" "${status}" "${signature}" "${completed}" <<'PY'
import json,sys
from datetime import datetime,timezone
from pathlib import Path
p,s,sig,c=sys.argv[1:]
Path(p).write_text(json.dumps({"status":s,"failure_signature":sig,"completed_cells":int(c),
 "progress":int(c),"expected_cells":60,
 "timestamp":datetime.now(timezone.utc).isoformat(timespec="seconds")},indent=2,sort_keys=True)+"\n")
PY
  gcloud storage cp "${OUT_DIR}/status.json" "${GCS_OUT}/status.json" >/dev/null 2>&1 || true
}
on_error() {
  local rc=$?; trap - ERR
  stop_log_syncer
  local sig; sig="$(tail -80 "${OUT_DIR}/remote.log" 2>/dev/null | sha256sum | awk '{print $1}')"
  log "FAILED rc=${rc} signature=${sig}"; write_status failed "${sig}"
  gcloud storage cp -r "${OUT_DIR}" "${BUCKET_URI}/experiments/" >/dev/null 2>&1 || true
  exit "${rc}"
}
trap on_error ERR

start_log_syncer
log "Installing dependencies and asserting L4 CUDA"
nvidia-smi
if ! python3 -m pip --version >/dev/null 2>&1; then
  log "image python3 lacks pip; installing python3-pip"
  apt-get update; DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
fi
python3 -m pip install -r requirements.txt
python3 - <<'PY'
import torch
assert torch.cuda.is_available(), "Refusing to run on CPU"
n=torch.cuda.get_device_name(0); assert "L4" in n, f"Refusing non-L4 GPU: {n}"
print(n)
PY
log "Running correctness guarantees"
python3 -m unittest discover -s tests -p 'test_ref_logprob_parity.py'
python3 -m unittest discover -s tests -p 'test_checkpoint_boundaries.py'
python3 -m unittest discover -s tests -p 'test_heldout_guard.py'

log "Restoring resumable progress (never logs, so a failure keeps its own traceback)"
gcloud storage cp "${GCS_OUT}/ledger.jsonl" "${OUT_DIR}/ledger.jsonl" 2>/dev/null || true
gcloud storage cp "${GCS_OUT}/train_acc_subsample_true.jsonl" "${OUT_DIR}/" 2>/dev/null || true
gcloud storage cp -r "${GCS_OUT}/rr_eps"* "${OUT_DIR}/" 2>/dev/null || true
gcloud storage cp -r "${BUCKET_URI}/experiments/ref_logps_stage2" "${ROOT}/experiments/" 2>/dev/null || true

log "Starting deletion-capable GPU idle watchdog"
nohup python3 idle_watchdog.py --idle-minutes 45 --poll-seconds 30 \
  --log "${OUT_DIR}/idle_watchdog.log" --delete-vm >"${OUT_DIR}/idle_watchdog.stdout" 2>&1 &

write_status running
log "Handing off to the Stage 2 runner (canary_cleared=${STAGE2_CANARY_CLEARED:-unset})"
python3 experiments/run_stage2_rr.py --bucket "${BUCKET_URI}" ${STAGE2_CANARY_CLEARED:+--canary-cleared}
rc=$?
log "Runner exited rc=${rc}"
stop_log_syncer
# Guarded: a failure in the final flush must not change the runner's exit status.
write_heartbeat || true
sync_logs || true
exit "${rc}"
