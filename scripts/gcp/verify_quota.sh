#!/usr/bin/env bash
# Check GPU quota using gcloud compute regions describe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

echo "Project: ${GCP_PROJECT}"
echo "Region:  ${GCP_REGION}"
echo ""

region_json="$(gcloud compute regions describe "${GCP_REGION}" \
  --project="${GCP_PROJECT}" \
  --format=json)"

python3 - <<'PY' "${region_json}" "${GCP_REGION}"
import json, sys
data = json.loads(sys.argv[1])
region = sys.argv[2]
metrics = {
    "NVIDIA_T4_GPUS": "T4 GPUs",
    "NVIDIA_L4_GPUS": "L4 GPUs",
    "NVIDIA_A100_GPUS": "A100 GPUs",
}
ok = False
print(f"GPU quotas in {region}:")
for metric, label in metrics.items():
    limit = 0.0
    for q in data.get("quotas", []):
        if q.get("metric") == metric:
            limit = float(q.get("limit", 0))
            break
    status = "OK" if limit > 0 else "ZERO"
    print(f"  {label}: limit={limit:g} [{status}]")
    if limit > 0:
        ok = True
if not ok:
    print("\nNo GPU quota in this region. Request increase:")
    print(f"  https://console.cloud.google.com/iam-admin/quotas?project={data.get('name','').split('/')[-1]}")
    sys.exit(1)
print("\nQuota check passed.")
PY

echo ""
echo "Available ${GPU_TYPE} in ${GCP_REGION}:"
gcloud compute accelerator-types list \
  --filter="zone~${GCP_REGION} AND name=${GPU_TYPE}" \
  --project="${GCP_PROJECT}" \
  --format="table(name,zone,maximumCardsPerInstance)"
