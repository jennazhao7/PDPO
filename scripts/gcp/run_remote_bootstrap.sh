#!/usr/bin/env bash
# Bootstrap PDPO on the remote VM from your Mac.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

REPO_URL="${PDPO_REPO_URL:-$(git -C "${PDPO_ROOT}" remote get-url origin 2>/dev/null || echo "")}"

gcloud compute scp "${SCRIPT_DIR}/bootstrap_vm.sh" \
  "${VM_NAME}:~/bootstrap_vm.sh" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}"

ENV_ARGS="PDPO_ROOT=\${HOME}/PDPO"
if [[ -n "${REPO_URL}" ]]; then
  ENV_ARGS="${ENV_ARGS} PDPO_REPO_URL=${REPO_URL}"
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  ENV_ARGS="${ENV_ARGS} HF_TOKEN=${HF_TOKEN}"
fi

gcloud compute ssh "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --command="chmod +x ~/bootstrap_vm.sh && ${ENV_ARGS} bash ~/bootstrap_vm.sh"
