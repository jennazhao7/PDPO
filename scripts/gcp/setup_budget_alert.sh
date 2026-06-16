#!/usr/bin/env bash
# Print instructions and optionally create a billing budget via gcloud beta.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
require_project

BUDGET_AMOUNT="${BUDGET_AMOUNT:-100}"
BILLING_ACCOUNT="${BILLING_ACCOUNT:-}"

echo "=== Cost guardrails for ${GCP_PROJECT} ==="
echo ""
echo "1. Stop VM when idle (saves GPU compute, keeps disk):"
echo "     bash scripts/gcp/stop_vm.sh"
echo ""
echo "2. Start VM when needed:"
echo "     bash scripts/gcp/start_vm.sh"
echo ""
echo "3. Backup outputs before deleting VM:"
echo "     bash scripts/gcp/backup_outputs.sh"
echo ""
echo "4. Set billing budget alert in Console:"
echo "     https://console.cloud.google.com/billing/budgets?project=${GCP_PROJECT}"
echo "     Recommended thresholds: 50%, 80%, 100% of your credit amount"
echo ""

if [[ -z "${BILLING_ACCOUNT}" ]]; then
  echo "To create budget via CLI, set BILLING_ACCOUNT and re-run:"
  echo "  gcloud billing accounts list"
  echo "  export BILLING_ACCOUNT=XXXXXX-XXXXXX-XXXXXX"
  echo "  export BUDGET_AMOUNT=100"
  echo "  bash scripts/gcp/setup_budget_alert.sh"
  exit 0
fi

if gcloud beta billing budgets create \
  --billing-account="${BILLING_ACCOUNT}" \
  --display-name="PDPO GPU budget" \
  --budget-amount="${BUDGET_AMOUNT}USD" \
  --threshold-rule=percent=50 \
  --threshold-rule=percent=80 \
  --threshold-rule=percent=100 \
  --filter-projects="projects/${GCP_PROJECT}" 2>/dev/null; then
  echo "Budget alert created for ${BUDGET_AMOUNT} USD."
else
  echo "CLI budget creation failed or already exists. Use Console link above."
fi
