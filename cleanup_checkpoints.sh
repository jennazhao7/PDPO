#!/bin/bash
# cleanup_checkpoints.sh
# Removes intermediate training checkpoints where the final adapter already exists.
# Safe: only deletes when the parent directory has a final adapter or M2_manifest.json.
# DRY RUN by default. Set DRY_RUN=0 to actually delete.

DRY_RUN=${DRY_RUN:-1}

if [ "$DRY_RUN" = "1" ]; then
    echo "=== DRY RUN — nothing will be deleted. Set DRY_RUN=0 to delete. ==="
else
    echo "=== LIVE RUN — deleting intermediate checkpoints ==="
fi

TOTAL_FREED=0
COUNT=0

find /users/jzhao7/PDPO -type d -name "checkpoint-*" | sort | while read CKPT; do
    PARENT=$(dirname "$CKPT")

    # Check if the parent has a final adapter
    HAS_FINAL=false
    [ -f "${PARENT}/adapter_config.json" ]            && HAS_FINAL=true
    [ -f "${PARENT}/fresh_lora/adapter_config.json" ] && HAS_FINAL=true
    [ -f "${PARENT}/stage2/adapter_config.json" ]     && HAS_FINAL=true
    [ -f "${PARENT}/M2_manifest.json" ]               && HAS_FINAL=true

    if $HAS_FINAL; then
        SIZE_KB=$(du -sk "$CKPT" 2>/dev/null | cut -f1)
        SIZE_MB=$((SIZE_KB / 1024))
        echo "  [${SIZE_MB}M] $CKPT"
        if [ "$DRY_RUN" = "0" ]; then
            rm -rf "$CKPT"
        fi
    fi
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "Dry run complete. To delete, run:"
    echo "  DRY_RUN=0 bash $(realpath "$0")"
else
    echo "Cleanup complete."
fi
