#!/usr/bin/env bash
# Stop VM to save GPU/compute costs (disk charges continue).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

gcloud compute instances stop "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}"

echo "Stopped ${VM_NAME}. Start again with: bash scripts/gcp/start_vm.sh"
