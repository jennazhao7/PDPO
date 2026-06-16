#!/usr/bin/env bash
# End-to-end GCP setup from your Mac. Requires gcloud auth and scripts/gcp/gcp.env.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

# Install gcloud if missing (macOS Python 3.9 workaround)
if ! command -v gcloud >/dev/null 2>&1 || ! gcloud --version >/dev/null 2>&1; then
  bash scripts/gcp/install_gcloud.sh
  export PATH="${HOME}/google-cloud-sdk/bin:${HOME}/.local/bin:${PATH}"
  export CLOUDSDK_PYTHON="$(${HOME}/.local/bin/uv python find 3.12 2>/dev/null || true)"
fi

if [[ ! -f scripts/gcp/gcp.env ]]; then
  echo "Copy and edit gcp.env first:"
  echo "  cp scripts/gcp/gcp.env.example scripts/gcp/gcp.env"
  exit 1
fi

bash scripts/gcp/preflight.sh

echo "=== Step 1: Verify GPU quota ==="
bash scripts/gcp/verify_quota.sh

echo ""
echo "=== Step 2: Create VM ==="
bash scripts/gcp/create_vm.sh

echo ""
echo "=== Step 3: Setup SSH ==="
bash scripts/gcp/setup_ssh.sh

echo ""
echo "=== Step 4: Bootstrap PDPO on VM (may take 10-15 min) ==="
bash scripts/gcp/run_remote_bootstrap.sh

echo ""
echo "=== Step 5: Smoke test on GPU ==="
bash scripts/gcp/run_smoke_test.sh

echo ""
echo "=== Step 6: Cost guardrails ==="
bash scripts/gcp/setup_budget_alert.sh

echo ""
echo "=== GCP setup complete ==="
echo "SSH: gcloud compute ssh pdpo-gpu --zone=us-central1-a"
echo "Stop VM: bash scripts/gcp/stop_vm.sh"
