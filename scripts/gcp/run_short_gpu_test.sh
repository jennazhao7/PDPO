#!/usr/bin/env bash
# ~5-10 min GPU smoke: verify CUDA + short gpt2 DPO debug train on remote VM.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

echo "Waiting for VM SSH..."
for i in $(seq 1 24); do
  if gcloud compute ssh "${VM_NAME}" --zone="${GCP_ZONE}" --project="${GCP_PROJECT}" \
    --command="echo ok" 2>/dev/null; then break; fi
  sleep 10
done

gcloud compute ssh "${VM_NAME}" --zone="${GCP_ZONE}" --project="${GCP_PROJECT}" --command='
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pdpo
cd ~/PDPO
export CUDA_VISIBLE_DEVICES=0
export HF_HOME=~/PDPO/.cache/huggingface

echo "=== 1/3 GPU check ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; assert torch.cuda.is_available(); print(\"CUDA OK:\", torch.cuda.get_device_name(0))"

echo ""
echo "=== 2/3 Short DPO train (gpt2, debug ~5-10 min) ==="
mkdir -p outputs_gcp_smoke
python -u train_dpo_stage1.py \
  --model gpt2 \
  --data ./preprocessing/truthydpo/dpo_train_ready.jsonl \
  --out ./outputs_gcp_smoke/gpt2-dpo-test \
  --debug --tiny 64 \
  --bsz 2 --ga 2 \
  --max-prompt 128 --max-target 128 --max-len 256

echo ""
echo "=== 3/3 Verify checkpoint ==="
ls -la outputs_gcp_smoke/gpt2-dpo-test/ | head -10
echo ""
echo "SUCCESS: GCP GPU training test completed."
'

echo ""
echo "Test finished. Stop VM to save costs: bash scripts/gcp/stop_vm.sh"
