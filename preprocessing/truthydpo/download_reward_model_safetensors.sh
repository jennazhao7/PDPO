#!/bin/bash
# Download reward model with safetensors format
# This helps avoid torch version restrictions

MODEL_NAME="OpenAssistant/reward-model-deberta-v3-large-v2"
CACHE_DIR="./cache/reward_model"

echo "📥 Downloading reward model with safetensors..."
echo "   Model: $MODEL_NAME"
echo "   Cache: $CACHE_DIR"

# Use huggingface-cli to download with safetensors preference
huggingface-cli download "$MODEL_NAME" \
    --local-dir "$CACHE_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "✅ Model downloaded to: $CACHE_DIR"
echo ""
echo "💡 Update generate_m1_labels_d2.py to use:"
echo "   --rm_model $CACHE_DIR"


