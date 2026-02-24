#!/bin/bash
# Push updated stage 1 LoRA adapters to HuggingFace, then remove from local cache.
# Usage: bash lora/push_stage1_to_hf_and_remove.sh [--repo YOUR_REPO] [--dry-run]

set -e
cd "$(dirname "$0")/../.."

REPO_ID="${REPO_ID:-Jennazhao7/pdpo-lora}"
DRY_RUN=false

for arg in "$@"; do
  case "$arg" in
    --repo=*) REPO_ID="${arg#*=}" ;;
    --dry-run) DRY_RUN=true ;;
  esac
done

echo "=== Push stage 1 to HF, then remove local ==="
echo "Repo: $REPO_ID"
echo ""

# Stage 1 adapters to push
STAGE1_ADAPTERS=(
  "models/gpt2-medium-truthy-stage1-rr-eps1:gpt2-medium:truthydpo:1.0"
  "models/gpt2-large-truthy-stage1-rr-eps1:gpt2-large:truthydpo:1.0"
  "models/pythia-1b-truthy-stage1-lora:pythia-1b:truthydpo:1.0"
)

for spec in "${STAGE1_ADAPTERS[@]}"; do
  IFS=':' read -r local_dir base dataset eps <<< "$spec"
  if [[ ! -d "$local_dir" ]]; then
    echo "⚠️  Skip $local_dir (not found)"
    continue
  fi

  echo ">>> Pushing $local_dir -> $REPO_ID ($base/$dataset/stage1/eps_$eps)"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [dry-run] would run push_lora_to_hf.py"
  else
    python lora/push_lora_to_hf.py \
      --repo_id "$REPO_ID" \
      --local_dir "$local_dir" \
      --base "$base" \
      --dataset "$dataset" \
      --stage stage1 \
      --eps "$eps"
  fi
  echo ""
done

echo "=== Removing local stage 1 adapters ==="
for spec in "${STAGE1_ADAPTERS[@]}"; do
  IFS=':' read -r local_dir base dataset eps <<< "$spec"
  if [[ ! -d "$local_dir" ]]; then
    continue
  fi

  echo ">>> Removing $local_dir"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "   [dry-run] would run: rm -rf $local_dir"
  else
    rm -rf "$local_dir"
    echo "   Removed."
  fi
  echo ""
done

echo "=== Note: M2 stack uses local stage1 paths ==="
echo "If you use outputs/stage2_lora or run Stage-2, update to load stage1 from HF:"
echo "  base + HF adapter: Jennazhao7/pdpo-lora (gpt2-medium/truthydpo/stage1/eps_1.0)"
echo ""
echo "=== Optional: clear HuggingFace model cache ==="
echo "  huggingface-cli delete-cache"
echo ""
echo "Done."
