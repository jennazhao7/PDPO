#!/bin/bash
# Script to clear old HuggingFace cache models to free up space
# Usage: bash eval/clear_hf_cache.sh [--dry-run]

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
    echo "🔍 DRY RUN MODE - No files will be deleted"
fi

echo "🔍 Checking HuggingFace cache size..."
CACHE_DIR="$HOME/.cache/huggingface"
if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ Cache directory not found: $CACHE_DIR"
    exit 1
fi

echo "Current cache size:"
du -sh "$CACHE_DIR"

echo ""
echo "📊 Largest models in cache:"
du -sh "$CACHE_DIR/hub/models--"* 2>/dev/null | sort -hr | head -15

echo ""
echo "💡 Recommended: Remove old/unused models to free space"
echo ""

# Models to keep (ones you're actively using)
KEEP_MODELS=(
    "models--Jennazhao7--gpt2-large-dpo-m1"
    "models--Setpember--Jon_GPT2L_DPO_3props_epi_05"
    "models--gpt2-large"
    "models--OpenAssistant--reward-model-deberta-v3-large-v2"
)

# Models that can be safely removed (old versions, unused)
REMOVE_MODELS=(
    "models--Setpember--Jon_GPT2L_DPO_props_epi_point1"  # Old PROPS version
    "models--Setpember--Jon_GPT2L_DPO_epi_point1"       # Old version
    "models--Setpember--Jon_GPT2L_DPO_3props_epi_01"     # Old 3PROPS version
    "models--Setpember--Jon_GPT2M_DPSGD_epi_point1"     # Medium model, not needed
    "models--Setpember--Jon_GPT2M_DPO_props_epi_point5" # Medium model, not needed
    "models--Jennazhao7--gpt2-medium-dpo-m1"            # Medium model, not needed
    "models--gpt2-medium"                                # Base model, can re-download
    "models--gpt2"                                        # Small model, can re-download
    "models--textattack--bert-base-uncased-SST-2"        # Unused model
    "models--Setpember--sft_gpt2_medium"                  # Empty/tiny
    "models--Setpember--sft_gpt2_large"                   # Empty/tiny
)

echo "🗑️  Models to remove (safe to delete):"
TOTAL_FREED=0
for model in "${REMOVE_MODELS[@]}"; do
    model_path="$CACHE_DIR/hub/$model"
    if [ -d "$model_path" ]; then
        size=$(du -sh "$model_path" 2>/dev/null | cut -f1)
        size_bytes=$(du -sb "$model_path" 2>/dev/null | cut -f1)
        echo "  - $model ($size)"
        if [ "$DRY_RUN" = false ]; then
            rm -rf "$model_path"
            echo "    ✅ Removed"
        fi
        TOTAL_FREED=$((TOTAL_FREED + size_bytes))
    fi
done

if [ "$DRY_RUN" = false ] && [ $TOTAL_FREED -gt 0 ]; then
    echo ""
    echo "✅ Cache cleared!"
    echo "📊 Space freed: ~$(numfmt --to=iec-i --suffix=B $TOTAL_FREED 2>/dev/null || echo "$((TOTAL_FREED / 1024 / 1024))MB")"
    echo ""
    echo "Remaining cache size:"
    du -sh "$CACHE_DIR"
elif [ "$DRY_RUN" = true ]; then
    echo ""
    echo "💡 To actually remove these files, run without --dry-run:"
    echo "   bash eval/clear_hf_cache.sh"
fi

echo ""
echo "📌 Models kept (actively used):"
for model in "${KEEP_MODELS[@]}"; do
    model_path="$CACHE_DIR/hub/$model"
    if [ -d "$model_path" ]; then
        size=$(du -sh "$model_path" 2>/dev/null | cut -f1)
        echo "  ✓ $model ($size)"
    fi
done
