#!/usr/bin/env bash
# Create a GPU VM for PDPO training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

try_zones() {
  local zone
  for zone in "$@"; do
    echo ""
    echo "Creating VM ${VM_NAME} (${GPU_COUNT}x ${GPU_TYPE}) in ${zone}..."
    if gcloud compute instances create "${VM_NAME}" \
      --project="${GCP_PROJECT}" \
      --zone="${zone}" \
      --machine-type="${MACHINE_TYPE}" \
      --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
      --maintenance-policy=TERMINATE \
      --boot-disk-size="${BOOT_DISK_SIZE}" \
      --boot-disk-type=pd-balanced \
      --image-family="${IMAGE_FAMILY}" \
      --image-project="${IMAGE_PROJECT}" \
      --metadata=install-nvidia-driver=True \
      --scopes=https://www.googleapis.com/auth/cloud-platform 2>&1; then
      GCP_ZONE="${zone}"
      return 0
    fi
    echo "Zone ${zone} unavailable, trying next..."
  done
  return 1
}

echo "Enabling APIs..."
gcloud services enable compute.googleapis.com storage.googleapis.com \
  --project="${GCP_PROJECT}"

echo ""
echo "Running quota check..."
bash "${SCRIPT_DIR}/verify_quota.sh"

for z in "${GCP_REGION}-a" "${GCP_REGION}-b" "${GCP_REGION}-c" "${GCP_REGION}-f"; do
  if gcloud compute instances describe "${VM_NAME}" \
    --zone="${z}" \
    --project="${GCP_PROJECT}" >/dev/null 2>&1; then
    GCP_ZONE="${z}"
    echo "VM ${VM_NAME} already exists in ${GCP_ZONE}."
    gcloud compute instances describe "${VM_NAME}" \
      --zone="${GCP_ZONE}" \
      --project="${GCP_PROJECT}" \
      --format="table(name,status,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)"
    exit 0
  fi
done

# Prefer configured zone, then try siblings in the same region.
fallback_zones=("${GCP_ZONE}")
for z in "${GCP_REGION}-a" "${GCP_REGION}-b" "${GCP_REGION}-c" "${GCP_REGION}-f"; do
  [[ " ${fallback_zones[*]} " == *" ${z} "* ]] || fallback_zones+=("${z}")
done

if ! try_zones "${fallback_zones[@]}"; then
  echo ""
  echo "ERROR: No GPU capacity for ${GPU_TYPE} in ${GCP_REGION}."
  echo "Try another zone in scripts/gcp/gcp.env, e.g.:"
  echo "  GCP_ZONE=us-central1-b"
  echo "Or switch to L4:"
  echo "  GPU_TYPE=nvidia-l4"
  echo "  MACHINE_TYPE=g2-standard-8"
  exit 1
fi

echo ""
echo "VM created in ${GCP_ZONE}. External IP:"
gcloud compute instances describe "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

echo ""
echo "Update scripts/gcp/gcp.env if needed: GCP_ZONE=${GCP_ZONE}"
echo "Next: bash scripts/gcp/setup_ssh.sh"
