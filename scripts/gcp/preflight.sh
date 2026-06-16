#!/usr/bin/env bash
# Check gcloud auth and project before running GCP setup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ -f "${SCRIPT_DIR}/gcp.env" ]]; then
  # shellcheck source=/dev/null
  source "${SCRIPT_DIR}/gcp.env"
fi

require_gcloud

if ! gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null | grep -q .; then
  echo "ERROR: Not logged in to gcloud."
  echo "Run:"
  echo "  gcloud auth login"
  echo "  gcloud auth application-default login"
  exit 1
fi

if [[ -z "${GCP_PROJECT:-}" || "${GCP_PROJECT}" == "your-project-id" ]]; then
  echo "ERROR: Set GCP_PROJECT in scripts/gcp/gcp.env"
  echo "  cp scripts/gcp/gcp.env.example scripts/gcp/gcp.env"
  echo "  # edit GCP_PROJECT=your-actual-project-id"
  exit 1
fi

gcloud config set project "${GCP_PROJECT}" --quiet
echo "Preflight OK: project=${GCP_PROJECT}, account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' | head -1)"
