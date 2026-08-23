#!/usr/bin/env bash
# Host the Stage 2 supervisor on a small always-on CPU VM instead of a laptop.
#
# The remaining Stage 2 sweep is ~186 GPU-hr / ~8 days. A laptop-hosted supervisor cannot carry
# that: the process does not survive the local session ending or the machine sleeping, and it
# depends on a user credential that expires every few hours. Three supervisors have now died this
# way, the most recent being the Stage 2 canary run's own supervisor, which exited after the
# canary gate. A VM with an attached service account has neither failure mode -- credentials come
# from the metadata server and refresh automatically, so no key material is created, downloaded,
# or handled.
#
# GO GATE. The supervisor idles until a GO marker appears in GCS and launches no GPU work before
# then. That makes it safe to create this VM and verify it survives while spending nothing on L4s:
#
#   bash scripts/gcp/launch_stage2_supervisor_vm.sh          # ~$0.014/hr, no GPU spend
#   gcloud storage cat gs://<bucket>/experiments/stage2_rr/supervisor_vm.log   # confirm alive
#   echo GO > /tmp/pdpo-go                                                    # releases the GPU spend
#   gcloud storage cp /tmp/pdpo-go gs://<bucket>/experiments/stage2_rr/GO
#
# Use a real file, NOT /dev/null: gcloud storage treats a character device as a non-regular
# file and skips it, so `cp /dev/null` leaves no object and the supervisor waits forever.
#
# Deleting the GO marker does not stop an in-flight sweep; delete the GPU VM and this VM for that.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${ROOT_DIR}/scripts/gcp"
# shellcheck source=scripts/gcp/common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

BUCKET_URI="${GCS_BUCKET}"
[[ "${BUCKET_URI}" == gs://* ]] || BUCKET_URI="gs://${BUCKET_URI}"
VM_NAME="${STAGE2_SUPERVISOR_VM_NAME:-pdpo-stage2-supervisor-cpu}"
GCP_ZONE="${STAGE2_SUPERVISOR_ZONE:-${GCP_ZONE}}"
MACHINE_TYPE="${STAGE2_SUPERVISOR_MACHINE_TYPE:-e2-small}"
SERVICE_ACCOUNT="${STAGE2_SUPERVISOR_SA:-}"
REMOTE_PREFIX="experiments/stage2_rr"
# Passed through to the supervisor, which passes it to every relaunch. Without it a relaunch
# re-arms the eps=0 canary gate and the sweep stops again after one checkpoint.
CANARY_CLEARED="${STAGE2_CANARY_CLEARED:-}"

if [[ -z "${SERVICE_ACCOUNT}" ]]; then
  project_number="$(gcloud projects describe "${GCP_PROJECT}" --format='value(projectNumber)')"
  SERVICE_ACCOUNT="${project_number}-compute@developer.gserviceaccount.com"
fi

existing="$(gcloud compute instances list --project="${GCP_PROJECT}" --filter="name=${VM_NAME}" --format='value(name,zone.basename(),status)')"
if [[ -n "${existing}" ]]; then
  echo "REFUSING: supervisor VM already exists: ${existing}" >&2
  exit 2
fi

# The supervisor runs launch_stage2_rr.sh, which builds the GPU VM's source bundle from this tree.
# So the supervisor VM must carry everything that bundle needs, not just the supervisor scripts.
BUNDLE="$(mktemp "${TMPDIR:-/tmp}/pdpo-stage2-supervisor-source.XXXXXX")"
BUNDLE_OBJECT="session-control/stage2-supervisor-source-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
COPYFILE_DISABLE=1 tar -C "${ROOT_DIR}" -czf "${BUNDLE}" \
  scripts/gcp/common.sh \
  scripts/gcp/gcp.env \
  scripts/gcp/supervise_stage2_rr.sh \
  scripts/gcp/launch_stage2_rr.sh \
  scripts/gcp/stage2_remote.sh \
  requirements.txt \
  idle_watchdog.py \
  precompute_ref_logps.py \
  config.py \
  experiments/run_stage2_rr.py \
  experiments/precompute_ref_logps_v3.py \
  experiments/build_eval_ref_cache.py \
  experiments/stage2_health.py \
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
  tests/test_heldout_guard.py
gcloud storage cp "${BUNDLE}" "${BUCKET_URI}/${BUNDLE_OBJECT}"
BUNDLE_SHA="$(shasum -a 256 "${BUNDLE}" 2>/dev/null | awk '{print $1}' || sha256sum "${BUNDLE}" | awk '{print $1}')"

STARTUP_SCRIPT="$(mktemp "${TMPDIR:-/tmp}/pdpo-stage2-supervisor-startup.XXXXXX")"
cat >"${STARTUP_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export GCP_PROJECT="${GCP_PROJECT}"
export GCP_ZONE="${GCP_ZONE}"
export GCS_BUCKET="${BUCKET_URI}"
export STAGE2_CANARY_CLEARED="${CANARY_CLEARED}"
export PDPO_SUPERVISOR_BUNDLE="${BUNDLE_OBJECT}"
export PDPO_SUPERVISOR_BUNDLE_SHA256="${BUNDLE_SHA}"
export HOME="/root"
ROOT="/root/PDPO"
LOG_DIR="\${ROOT}/${REMOTE_PREFIX}"
mkdir -p "\${ROOT}" "\${LOG_DIR}"
exec > >(tee -a "\${LOG_DIR}/supervisor_vm.log") 2>&1

echo "[\$(date -Is)] supervisor VM boot bundle=${BUNDLE_OBJECT} canary_cleared='${CANARY_CLEARED}'"
gcloud storage cp "${BUCKET_URI}/${BUNDLE_OBJECT}" /tmp/pdpo-stage2-supervisor-source.tar.gz
tar -xzf /tmp/pdpo-stage2-supervisor-source.tar.gz -C "\${ROOT}"
cd "\${ROOT}"

# Deliberately does NOT delete itself on exit. If the supervisor stops, the VM stays up with its
# log intact -- the opposite of the laptop failure mode where the process vanished silently.
publish() {
  gcloud storage cp "\${LOG_DIR}/supervisor_vm.log" \\
    "${BUCKET_URI}/${REMOTE_PREFIX}/supervisor_vm.log" >/dev/null 2>&1 || true
}

echo "[\$(date -Is)] waiting for GO marker at ${BUCKET_URI}/${REMOTE_PREFIX}/GO"
publish
while ! gcloud storage ls "${BUCKET_URI}/${REMOTE_PREFIX}/GO" >/dev/null 2>&1; do
  sleep 60
  echo "[\$(date -Is)] alive, waiting for GO (no GPU spend)"
  publish
done

echo "[\$(date -Is)] GO observed; starting Stage 2 supervisor"
publish
# Keep publishing while the supervisor runs. supervise_stage2_rr.sh pushes its own supervisor.log
# on every log line, but this VM's boot/exit context lives only in supervisor_vm.log.
( while true; do sleep 300; publish; done ) >/dev/null 2>&1 &
bash scripts/gcp/supervise_stage2_rr.sh || rc=\$?
echo "[\$(date -Is)] supervisor exited rc=\${rc:-0}"
publish
# Stay up so the exit reason remains inspectable.
sleep infinity
EOF

echo "Creating always-on CPU supervisor VM ${VM_NAME} in ${GCP_ZONE} (SA ${SERVICE_ACCOUNT})"
gcloud compute instances create "${VM_NAME}" \
  --project="${GCP_PROJECT}" \
  --zone="${GCP_ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-standard \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --service-account="${SERVICE_ACCOUNT}" \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata-from-file=startup-script="${STARTUP_SCRIPT}"
cat <<MSG
Supervisor VM ${VM_NAME} created. It idles until the GO marker is written and spends no GPU money
until then. Verify it is alive, then release the sweep:

  gcloud storage cat ${BUCKET_URI}/${REMOTE_PREFIX}/supervisor_vm.log
  gcloud storage cp /dev/null ${BUCKET_URI}/${REMOTE_PREFIX}/GO
MSG
