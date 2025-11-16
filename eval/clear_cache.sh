#!/bin/bash
# Script to clear HuggingFace cache
# Usage: bash eval/clear_cache.sh [all|large|medium|specific]

CACHE_DIR="$HOME/.cache/huggingface"

if [ ! -d "$CACHE_DIR" ]; then
    echo "❌ Cache directory not found: $CACHE_DIR"
    exit 1
fi

echo "🔍 Current cache size:"
du -sh "$CACHE_DIR"
echo ""

case "${1:-all}" in
    all)
        echo "🗑️  Clearing ALL HuggingFace cache..."
        rm -rf "$CACHE_DIR/hub"/*
        echo "✅ Cache cleared!"
        ;;
    large)
        echo "🗑️  Clearing large models (GPT-2 Large, etc.)..."
        rm -rf "$CACHE_DIR/hub/models--gpt2-large"
        rm -rf "$CACHE_DIR/hub/models--Jennazhao7--gpt2-large-dpo-m1"
        rm -rf "$CACHE_DIR/hub/models--Setpember--Jon_GPT2L"*
        echo "✅ Large models cleared!"
        ;;
    medium)
        echo "🗑️  Clearing medium models (GPT-2 Medium, etc.)..."
        rm -rf "$CACHE_DIR/hub/models--gpt2-medium"
        rm -rf "$CACHE_DIR/hub/models--Jennazhao7--gpt2-medium"*
        rm -rf "$CACHE_DIR/hub/models--Setpember--Jon_GPT2M"*
        echo "✅ Medium models cleared!"
        ;;
    reward)
        echo "🗑️  Clearing reward model..."
        rm -rf "$CACHE_DIR/hub/models--OpenAssistant--reward-model-deberta-v3-large-v2"
        echo "✅ Reward model cleared!"
        ;;
    *)
        echo "Usage: bash eval/clear_cache.sh [all|large|medium|reward]"
        echo ""
        echo "Options:"
        echo "  all     - Clear everything (27GB)"
        echo "  large   - Clear GPT-2 Large models (~6GB)"
        echo "  medium  - Clear GPT-2 Medium models (~6GB)"
        echo "  reward  - Clear reward model (~1.7GB)"
        echo ""
        echo "Current cache breakdown:"
        du -sh "$CACHE_DIR/hub/models--"* 2>/dev/null | sort -hr | head -10
        exit 1
        ;;
esac

echo ""
echo "📊 Remaining cache size:"
du -sh "$CACHE_DIR" 2>/dev/null || echo "Cache cleared!"

