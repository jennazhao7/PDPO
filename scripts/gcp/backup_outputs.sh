#!/usr/bin/env bash
# Sync PDPO outputs from VM to GCS (run from Mac or on VM with gsutil).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

OUTPUT_DIR="${1:-outputs_openllama_tonight}"
REMOTE="${2:-gs://${GCS_BUCKET}/pdpo-outputs/}"

if ! gsutil ls "gs://${GCS_BUCKET}" >/dev/null 2>&1; then
  echo "Creating bucket gs://${GCS_BUCKET}..."
  gsutil mb -l "${GCP_REGION}" "gs://${GCS_BUCKET}"
fi

echo "Syncing ${VM_NAME}:${PDPO_ROOT}/${OUTPUT_DIR} -> ${REMOTE}"
gcloud compute ssh "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --command="gsutil -m rsync -r \${HOME}/PDPO/${OUTPUT_DIR} ${REMOTE}${OUTPUT_DIR}/"

echo "Backup complete: ${REMOTE}${OUTPUT_DIR}/"
