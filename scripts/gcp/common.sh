#!/usr/bin/env bash
# Shared defaults for GCP scripts. Override via environment or gcp.env.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PDPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${SCRIPT_DIR}/gcp.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/gcp.env"
fi

: "${GCP_PROJECT:=$(gcloud config get-value project 2>/dev/null || true)}"
: "${GCP_ZONE:=us-central1-a}"
: "${GCP_REGION:=us-central1}"
: "${VM_NAME:=pdpo-gpu}"
: "${MACHINE_TYPE:=n1-standard-8}"
: "${GPU_TYPE:=nvidia-tesla-t4}"
: "${GPU_COUNT:=1}"
: "${BOOT_DISK_SIZE:=200GB}"
: "${GCS_BUCKET:=${GCP_PROJECT}-pdpo}"
: "${IMAGE_FAMILY:=common-cu129-ubuntu-2204-nvidia-580}"
: "${IMAGE_PROJECT:=deeplearning-platform-release}"

require_gcloud() {
  if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud not found. Run: bash scripts/gcp/install_gcloud.sh" >&2
    exit 1
  fi
  if ! gcloud --version >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
      export CLOUDSDK_PYTHON="$(uv python find 3.12 2>/dev/null || true)"
    fi
    if ! gcloud --version >/dev/null 2>&1; then
      echo "gcloud needs Python 3.10+. Run: bash scripts/gcp/install_gcloud.sh" >&2
      exit 1
    fi
  fi
}

require_project() {
  require_gcloud
  if [[ -z "${GCP_PROJECT}" || "${GCP_PROJECT}" == "(unset)" ]]; then
    echo "ERROR: Set GCP project: export GCP_PROJECT=your-project-id" >&2
    echo "  or: gcloud config set project YOUR_PROJECT_ID" >&2
    exit 1
  fi
}
