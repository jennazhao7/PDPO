#!/usr/bin/env bash
# Launch one isolated Spot L4 for the Stage 1 operating-point pilot.
# Refuses if an owned VM already exists (delete-before-recreate is the supervisor's job).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts/gcp"
# shellcheck source=scripts/gcp/common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

BUCKET_URI="${GCS_BUCKET}"
[[ "${BUCKET_URI}" == gs://* ]] || BUCKET_URI="gs://${BUCKET_URI}"
VM_NAME="${STAGE2_VM_NAME:-pdpo-stage2-rr-l4}"
GCP_ZONE="${STAGE2_ZONE:-${GCP_ZONE}}"
MACHINE_TYPE="${STAGE2_MACHINE_TYPE:-g2-standard-8}"
BOOT_DISK_SIZE="${STAGE2_BOOT_DISK_SIZE:-200GB}"
export VM_NAME GCP_ZONE

existing="$(gcloud compute instances list --project="${GCP_PROJECT}" --filter="name=${VM_NAME}" --format='value(name,zone.basename(),status)')"
if [[ -n "${existing}" ]]; then
  echo "REFUSING: owned VM already exists: ${existing}" >&2
  exit 2
fi

BUNDLE="$(mktemp "${TMPDIR:-/tmp}/pdpo-stage2-source.XXXXXX")"
BUNDLE_OBJECT="session-control/stage2-source-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
COPYFILE_DISABLE=1 tar -C "${ROOT_DIR}" -czf "${BUNDLE}" \
  requirements.txt \
  idle_watchdog.py \
  precompute_ref_logps.py \
  config.py \
  experiments/run_stage2_rr.py \
  experiments/precompute_ref_logps_v3.py \
  experiments/build_eval_ref_cache.py \
  experiments/heldout_guard.py \
  experiments/aggregate.py \
  experiments/manifest.yaml \
  stage2_debugging/train_matched_cached.py \
  stage2_debugging/eval_preference_accuracy.py \
  stage2_debugging/eval_privacy_audit.py \
  stage2_debugging/matched_core.py \
  stage2_debugging/metric_primitives.py \
  stage2_debugging/ref_logprob_core.py \
  data/pku_saferlhf_secure_v3 \
  data/hhrlhf_secure_v3 \
  data/stage2_rr \
  experiments/stage2_rr/PREREGISTRATION.md \
  tests/test_ref_logprob_parity.py \
  tests/test_checkpoint_boundaries.py \
  tests/test_heldout_guard.py \
  scripts/gcp/stage2_remote.sh
gcloud storage cp "${BUNDLE}" "${BUCKET_URI}/${BUNDLE_OBJECT}"
BUNDLE_SHA="$(sha256sum "${BUNDLE}" 2>/dev/null | awk '{print $1}' || shasum -a 256 "${BUNDLE}" | awk '{print $1}')"

STARTUP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/pdpo-stage1-startup.XXXXXX")"
cat >"${STARTUP_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export GCP_PROJECT="${GCP_PROJECT}"
export GCP_ZONE="${GCP_ZONE}"
export GCS_BUCKET="${BUCKET_URI}"
export PREFERRED_CARD="L4"
export VM_NAME="${VM_NAME}"
# Durable identifier of the exact code this run executes. No git commit can serve: the bundle is
# built from an uncommitted tree, so the immutable tarball is the only stable code identity.
export STAGE2_CANARY_CLEARED="${STAGE2_CANARY_CLEARED:-}"
export PDPO_SOURCE_BUNDLE="${BUNDLE_OBJECT}"
export PDPO_SOURCE_BUNDLE_SHA256="${BUNDLE_SHA}"
export HOME="/root"
export HF_HOME="/root/PDPO/.cache/huggingface"
ROOT="/root/PDPO"
OUT_DIR="\${ROOT}/experiments/stage2_rr"
mkdir -p "\${ROOT}" "\${OUT_DIR}"
exec > >(tee -a "\${OUT_DIR}/startup.log") 2>&1

# Boot-phase log sync. stage2_remote.sh installs a richer syncer once it starts, but the driver
# install and bundle extraction happen before that, and a VM that wedges there previously left
# nothing at all in the bucket. This loop covers that window; it costs one small object per minute.
( while true; do
    sleep 60
    gcloud storage cp "\${OUT_DIR}/startup.log" "${BUCKET_URI}/experiments/stage2_rr/startup.log" \
      >/dev/null 2>&1 || true
  done ) >/dev/null 2>&1 &
BOOT_SYNC_PID=\$!

cleanup_and_delete() {
  rc=\$?
  set +e
  kill "\${BOOT_SYNC_PID}" 2>/dev/null || true
  # Final flush. Per-checkpoint syncs already committed progress, so this only catches the tail.
  gcloud storage cp -r "\${OUT_DIR}" "${BUCKET_URI}/experiments/" || true
  gcloud compute instances delete "${VM_NAME}" --project="${GCP_PROJECT}" --zone="${GCP_ZONE}" --quiet \
    || sudo shutdown -h now
  exit \${rc}
}
trap cleanup_and_delete EXIT

gcloud storage cp "${BUCKET_URI}/${BUNDLE_OBJECT}" /tmp/pdpo-stage2-source.tar.gz
tar -xzf /tmp/pdpo-stage2-source.tar.gz -C "\${ROOT}"
cd "\${ROOT}"
bash scripts/gcp/stage2_remote.sh
EOF

echo "Creating isolated Spot L4 VM ${VM_NAME} in ${GCP_ZONE}"
gcloud compute instances create "${VM_NAME}" \
  --project="${GCP_PROJECT}" \
  --zone="${GCP_ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=nvidia-l4,count=1" \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --boot-disk-size="${BOOT_DISK_SIZE}" \
  --boot-disk-type=pd-standard \
  --image-family="${IMAGE_FAMILY}" \
  --image-project="${IMAGE_PROJECT}" \
  --metadata=install-nvidia-driver=True \
  --metadata-from-file=startup-script="${STARTUP_SCRIPT}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
echo "Launched ${VM_NAME}; per-checkpoint progress persists under ${BUCKET_URI}/experiments/stage2_rr."
