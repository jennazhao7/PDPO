#!/usr/bin/env bash
# Supervisor for the Stage 2 randomized-response arms.
#
# Mirrors scripts/gcp/supervise_token_concentration.sh: delete-before-recreate, multi-zone Spot
# L4 relaunch, and a three-strike breaker that trips only on an identical failure signature at an
# UNCHANGED progress count. Because the runner advances progress once per evaluated checkpoint
# (12 in the Gate 1 population, plus 4 HH-RLHF evaluations outside it), a preemption that costs one
# checkpoint still counts as progress and is retried, while a genuinely stuck cell trips the
# breaker.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts/gcp"
# shellcheck source=scripts/gcp/common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

BUCKET_URI="${GCS_BUCKET}"
[[ "${BUCKET_URI}" == gs://* ]] || BUCKET_URI="gs://${BUCKET_URI}"
VM_NAME="${STAGE2_VM_NAME:-pdpo-stage2-rr-l4}"
ZONES="${STAGE2_ZONES:-asia-east1-a asia-east1-b asia-east1-c us-central1-a us-central1-b us-central1-c}"
POLL_SECONDS="${POLL_SECONDS:-300}"
MAX_RELAUNCHES="${MAX_RELAUNCHES:-60}"
BREAKER_N=3
REMOTE_PREFIX="experiments/stage2_rr"
LOG_DIR="${ROOT_DIR}/${REMOTE_PREFIX}"
LOG_PATH="${LOG_DIR}/supervisor.log"
mkdir -p "${LOG_DIR}"

log() {
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG_PATH}"
  gcloud storage cp "${LOG_PATH}" "${BUCKET_URI}/${REMOTE_PREFIX}/supervisor.log" >/dev/null 2>&1 || true
}

snapshot() {
  gcloud storage cat "${BUCKET_URI}/${REMOTE_PREFIX}/status.json" 2>/dev/null \
    | python3 -c 'import json,sys; d=json.load(sys.stdin); print("|".join(str(d.get(k,"")) for k in ("status","failure_signature","progress")))' \
    || true
}

complete() {
  local value
  value="$(snapshot)"
  local state="${value%%|*}"
  if [[ "${state}" == "complete" ]]; then
    return 0
  fi
  # canary_gate is terminal ONLY while the canary is unreviewed: the runner deliberately stopped
  # for confirmation, and relaunching would march past the gate one checkpoint at a time. Once
  # STAGE2_CANARY_CLEARED is set the gate has been reviewed and passed, and the stale
  # canary_gate status.json that run left behind must not be read as completion -- otherwise the
  # supervisor exits on its first poll without ever launching a VM.
  if [[ -z "${STAGE2_CANARY_CLEARED:-}" && "${state}" == "canary_gate" ]]; then
    return 0
  fi
  return 1
}

instance_rows() {
  gcloud compute instances list --project="${GCP_PROJECT}" --filter="name=${VM_NAME}" --format='value(name,zone.basename(),status)'
}

consecutive_launch_failures=0

# Every halt path must free the GPU before recording the verdict, or a "STOPPED" file sits in
# the bucket while an L4 keeps billing.
halt_with() {
  local code="$1" label="$2"
  log "HALT ${label}: deleting owned GPU VM before writing verdict"
  while read -r name zone _; do
    [[ -n "${name:-}" ]] || continue
    gcloud compute instances delete "${name}" --project="${GCP_PROJECT}" --zone="${zone}" --quiet || true
  done <<<"$(instance_rows)"
  # Re-run the health writer so HEALTH.json reflects the post-delete state and the verdict.
  timeout 150 python3 experiments/stage2_health.py --project "${GCP_PROJECT}" --bucket "${BUCKET_URI}" \
    --vm-name "${VM_NAME}" --out-dir "${LOG_DIR}" \
    --consecutive-launch-failures "${consecutive_launch_failures}" \
    --relaunches-used "${relaunches}" --max-relaunches "${MAX_RELAUNCHES}" || true
  log "HALT ${label}: supervisor exiting; no further relaunches"
  exit "${code}"
}

write_health() {
  set +e
  timeout 150 python3 experiments/stage2_health.py --project "${GCP_PROJECT}" --bucket "${BUCKET_URI}" \
    --vm-name "${VM_NAME}" --out-dir "${LOG_DIR}" \
    --consecutive-launch-failures "${consecutive_launch_failures}" \
    --relaunches-used "${relaunches}" --max-relaunches "${MAX_RELAUNCHES}"
  local rc=$?
  set -e
  case "${rc}" in
    10) halt_with 10 "STALLED" ;;
    11) halt_with 11 "STOPPED_SPEND_CEILING" ;;
    12) halt_with 12 "FAILED_LAUNCH_CHURN" ;;
  esac
}

relaunches=0
last_signature=""
last_progress=""
same_failures=0
write_health
log "Stage-2 supervisor start project=${GCP_PROJECT} vm=${VM_NAME} breaker=${BREAKER_N}"
while ! complete; do
  rows="$(instance_rows)"
  if printf '%s\n' "${rows}" | awk '$3 == "RUNNING" {found=1} END {exit found ? 0 : 1}'; then
    status="$(snapshot)"
    log "VM running snapshot=${status:-none}; waiting ${POLL_SECONDS}s"
    consecutive_launch_failures=0
    write_health
    sleep "${POLL_SECONDS}"
    continue
  fi
  while read -r name zone status; do
    [[ -n "${name:-}" ]] || continue
    log "Deleting non-running owned VM name=${name} zone=${zone} status=${status}"
    gcloud compute instances delete "${name}" --project="${GCP_PROJECT}" --zone="${zone}" --quiet || true
  done <<<"${rows}"

  state="$(snapshot)"
  IFS='|' read -r run_status signature progress <<<"${state:-||}"
  if [[ "${run_status}" == "failed" && -n "${signature}" ]]; then
    if [[ "${signature}" == "${last_signature}" && "${progress}" == "${last_progress}" ]]; then
      same_failures=$((same_failures + 1))
    else
      same_failures=1
    fi
    last_signature="${signature}"
    last_progress="${progress}"
    log "Failure signature=${signature} progress=${progress} consecutive=${same_failures}/${BREAKER_N}"
    if [[ "${same_failures}" -ge "${BREAKER_N}" ]]; then
      log "CIRCUIT BREAKER: identical failure repeated ${BREAKER_N} times without progress"
      exit 3
    fi
  fi
  if [[ "${relaunches}" -ge "${MAX_RELAUNCHES}" ]]; then
    log "ERROR: max relaunches reached (${MAX_RELAUNCHES})"
    exit 1
  fi
  launched=0
  for zone in ${ZONES}; do
    log "Launching resume attempt=$((relaunches + 1)) zone=${zone} progress=${progress:-0}/60"
    # launch_stage2_rr.sh bakes STAGE2_CANARY_CLEARED into the VM startup script, so a
    # relaunch that dropped it would silently re-arm the gate and stop after one cell.
    if (cd "${ROOT_DIR}" && STAGE2_ZONE="${zone}" \
          STAGE2_CANARY_CLEARED="${STAGE2_CANARY_CLEARED:-}" \
          bash scripts/gcp/launch_stage2_rr.sh); then
      launched=1
      relaunches=$((relaunches + 1))
      break
    fi
  done
  if [[ "${launched}" == "0" ]]; then
    consecutive_launch_failures=$((consecutive_launch_failures + 1))
    log "No Spot L4 launch succeeded (consecutive=${consecutive_launch_failures}); retrying after ${POLL_SECONDS}s"
  else
    consecutive_launch_failures=0
  fi
  write_health
  sleep "${POLL_SECONDS}"
done

log "Stage-2 sweep complete relaunches=${relaunches}; waiting for owned VM deletion"
while [[ -n "$(instance_rows)" ]]; do
  sleep 30
done
gcloud storage rsync -r "${BUCKET_URI}/${REMOTE_PREFIX}" "${LOG_DIR}"
# Report from the artifacts Stage 2 actually writes. The previous version read
# GATE1_STATUS.json -- a Stage 1 artifact Stage 2 never produces -- so this block always
# raised FileNotFoundError and the supervisor died on its exit path instead of reporting.
python3 - <<'REPORT_PY' "${LOG_DIR}"
import collections, json, pathlib, sys
log_dir = pathlib.Path(sys.argv[1])
status_path = log_dir / 'status.json'
status = json.loads(status_path.read_text()) if status_path.is_file() else {}
print('STAGE 2 status:', status.get('status', 'unknown'),
      'progress:', status.get('progress'), '/', status.get('expected_cells'))
ledger = log_dir / 'ledger.jsonl'
if ledger.is_file():
    rows = [json.loads(l) for l in ledger.read_text().splitlines() if l.strip()]
    per_arm = collections.Counter(r.get('eps') for r in rows)
    order = sorted(per_arm.items(), key=lambda kv: (kv[0] is None, kv[0]))
    print('cells per eps:', dict(order))
else:
    print('no ledger at', ledger)
REPORT_PY
log "Stage-2 report synced to ${LOG_DIR}"
