#!/usr/bin/env bash
# Configure local SSH for gcloud and print connect instructions.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

gcloud compute config-ssh --project="${GCP_PROJECT}" --quiet

EXTERNAL_IP="$(gcloud compute instances describe "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null || echo "")"

echo ""
echo "SSH connect:"
echo "  gcloud compute ssh ${VM_NAME} --zone=${GCP_ZONE} --project=${GCP_PROJECT}"
if [[ -n "${EXTERNAL_IP}" ]]; then
  echo ""
  echo "External IP: ${EXTERNAL_IP}"
  echo "Cursor/VS Code Remote SSH host: ${GCP_ZONE}.${VM_NAME}.${GCP_PROJECT}"
fi
