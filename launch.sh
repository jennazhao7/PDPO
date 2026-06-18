#!/usr/bin/env bash
# Cost-controlled single-GPU launcher for PDPO.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts/gcp"
# shellcheck source=scripts/gcp/common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

FORCE_A100=0
YES=0
for arg in "$@"; do
  case "${arg}" in
    --force-a100) FORCE_A100=1 ;;
    --yes) YES=1 ;;
    *) echo "Unknown arg: ${arg}" >&2; exit 2 ;;
  esac
done

PREFERRED_CARD="${PREFERRED_CARD:-L4}"
CARD="${PREFERRED_CARD^^}"
case "${CARD}" in
  L4)
    MACHINE_TYPE="g2-standard-8"
    GPU_TYPE="nvidia-l4"
    ;;
  T4)
    MACHINE_TYPE="n1-standard-8"
    GPU_TYPE="nvidia-tesla-t4"
    ;;
  A100)
    if [[ "${FORCE_A100}" != "1" ]]; then
      echo "REFUSING A100 without --force-a100." >&2
      exit 2
    fi
    echo "A100 is expensive and should not be used by default."
    read -r -p 'Type "A100 on demand approved" to continue: ' confirm
    [[ "${confirm}" == "A100 on demand approved" ]] || { echo "Cancelled."; exit 2; }
    MACHINE_TYPE="a2-highgpu-1g"
    GPU_TYPE="nvidia-tesla-a100"
    ;;
  *)
    echo "Unknown PREFERRED_CARD=${PREFERRED_CARD}; expected L4, T4, or A100." >&2
    exit 2
    ;;
esac

GPU_COUNT=1
BOOT_DISK_SIZE="${COST_BOOT_DISK_SIZE:-50GB}"
VM_NAME="${VM_NAME:-pdpo-spot-runner}"
BUCKET_URI="${GCS_BUCKET}"
if [[ "${BUCKET_URI}" != gs://* ]]; then
  BUCKET_URI="gs://${BUCKET_URI}"
fi

export GCP_PROJECT GCP_ZONE GCS_BUCKET="${BUCKET_URI}" PREFERRED_CARD="${CARD}"

echo "=== Cost preflight ==="
if [[ "${YES}" == "1" ]]; then
  python3 "${ROOT_DIR}/preflight_cost.py" --card "${CARD}" --yes
else
  python3 "${ROOT_DIR}/preflight_cost.py" --card "${CARD}"
fi

STARTUP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/pdpo-startup.XXXXXX.sh")"
cat > "${STARTUP_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export GCP_PROJECT="${GCP_PROJECT}"
export GCP_ZONE="${GCP_ZONE}"
export GCS_BUCKET="${BUCKET_URI}"
export PREFERRED_CARD="${CARD}"
export VM_NAME="${VM_NAME}"
export PDPO_ROOT="\${HOME}/PDPO"
export HF_HOME="\${PDPO_ROOT}/.cache/huggingface"

log() { echo "[\$(date -Is)] \$*"; }

cleanup_and_delete() {
  rc=\$?
  set +e
  cd "\${PDPO_ROOT}" 2>/dev/null || true
  log "Pushing results and ledgers to ${BUCKET_URI}"
  gcloud storage cp -r experiments/run_queue "${BUCKET_URI}/experiments/" || true
  gcloud storage cp -r experiments/smoke_results "${BUCKET_URI}/experiments/" || true
  gcloud storage cp cost_ledger.csv "${BUCKET_URI}/cost_ledger.csv" || true
  log "Deleting VM ${VM_NAME}"
  gcloud compute instances delete "${VM_NAME}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --quiet || sudo shutdown -h now
  exit \$rc
}
trap cleanup_and_delete EXIT

if [[ -f "\${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "\${HOME}/miniconda3/etc/profile.d/conda.sh"
fi

if [[ ! -d "\${PDPO_ROOT}/.git" ]]; then
  git clone "\${PDPO_REPO_URL:-https://github.com/jennazhao7/PDPO.git}" "\${PDPO_ROOT}"
else
  git -C "\${PDPO_ROOT}" pull --ff-only || true
fi

cd "\${PDPO_ROOT}"
mkdir -p "\${HF_HOME}" experiments
if command -v conda >/dev/null 2>&1; then
  conda activate "\${CONDA_ENV:-pdpo}" || true
fi
python3 -m pip install -r requirements.txt || true

log "Pulling cached data/results from ${BUCKET_URI}"
gcloud storage cp -r "${BUCKET_URI}/experiments/run_queue" experiments/ || true
gcloud storage cp -r "${BUCKET_URI}/ref_logps" experiments/ || true
gcloud storage cp -r "${BUCKET_URI}/data" . || true
mkdir -p stage2_debugging
gcloud storage cp -r "${BUCKET_URI}/stage2_debugging/preprocessing" stage2_debugging/ || true
gcloud storage cp -r "${BUCKET_URI}/stage2_debugging/testsets" stage2_debugging/ || true
gcloud storage cp "${BUCKET_URI}/stage2_debugging/test_pref.jsonl" stage2_debugging/test_pref.jsonl || true
gcloud storage cp "${BUCKET_URI}/cost_ledger.csv" cost_ledger.csv || true

log "Starting idle watchdog"
nohup python3 idle_watchdog.py --log idle_watchdog.log >/tmp/pdpo_idle_watchdog.out 2>&1 &

export CUDA_VISIBLE_DEVICES=0
log "Running mandatory smoke gate"
python3 experiments/smoke.py

log "Running queue"
python3 experiments/run_queue.py --card "${CARD}" --cost-ledger cost_ledger.csv
log "Queue complete"
EOF

echo "Creating Spot VM ${VM_NAME} (${CARD}: ${MACHINE_TYPE}, ${GPU_TYPE}) in ${GCP_ZONE}"
gcloud compute instances create "${VM_NAME}" \
  --project="${GCP_PROJECT}" \
  --zone="${GCP_ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --boot-disk-type=pd-balanced \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=startup-script="${STARTUP_SCRIPT}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform

echo "Launched ${VM_NAME}. It will run smoke.py, then run_queue.py, sync results, and delete itself."
