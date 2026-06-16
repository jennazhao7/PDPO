#!/usr/bin/env bash
# Run GPU verify + optional full quick_sb_test on remote VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

GPU="${GPU:-0}"
FULL_TEST="${FULL_TEST:-0}"

REMOTE_CMD='set -euo pipefail
source ~/.bashrc 2>/dev/null || true
export PDPO_ROOT=${HOME}/PDPO
export HF_HOME=${PDPO_ROOT}/.cache/huggingface
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate pdpo
cd ${PDPO_ROOT}
bash scripts/gcp/verify_gpu.sh'

if [[ "${FULL_TEST}" == "1" ]]; then
  REMOTE_CMD="${REMOTE_CMD}
bash scripts/gcp/prepare_smoke_data.sh
GPU=${GPU} bash experiments/quick_sb_test.sh"
fi

gcloud compute ssh "${VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --project="${GCP_PROJECT}" \
  --command="${REMOTE_CMD}"

echo ""
if [[ "${FULL_TEST}" == "1" ]]; then
  echo "Full smoke test complete."
else
  echo "GPU sanity check complete. For full quick_sb_test (slow, downloads Qwen2.5-3B):"
  echo "  FULL_TEST=1 bash scripts/gcp/run_smoke_test.sh"
fi
