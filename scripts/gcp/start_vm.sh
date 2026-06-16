#!/usr/bin/env bash
# Start a stopped PDPO GPU VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

gcloud compute instances start "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}"

EXTERNAL_IP="$(gcloud compute instances describe "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"

echo "Started ${VM_NAME}. External IP: ${EXTERNAL_IP}"
echo "SSH: gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE}"
