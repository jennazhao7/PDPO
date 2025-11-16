#!/usr/bin/env python3
"""
Simple script to push a local model to HuggingFace Hub.
Just change the model_dir and repo_id variables below.
"""

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== CONFIGURATION - Change these =====
model_dir = "/users/jzhao7/PDPO/models/truthydpo/pythia-1b-dpo-lora-truthydpo"
repo_id = "Jennazhao7/pythia1b-truthydpo-m1"
private = True
# =========================================

# Check if model_dir has checkpoints (if no config.json in root)
model_path = Path(model_dir)
if not (model_path / "config.json").exists():
    # Look for checkpoints
    checkpoints = sorted([d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"⚠️  No final model found, using latest checkpoint: {latest_checkpoint.name}")
        model_dir = str(latest_checkpoint)
    else:
        print(f"❌ Error: No model files found in {model_dir}")
        exit(1)

print(f"📦 Loading model from: {model_dir}")
print(f"🚀 Pushing to: {repo_id}")
print(f"🔒 Private: {private}")
print()

# Load tokenizer and model
print("Loading tokenizer...")
tok = AutoTokenizer.from_pretrained(model_dir)

print("Loading model...")
mdl = AutoModelForCausalLM.from_pretrained(model_dir)

# Push to HuggingFace
print(f"\n📤 Pushing tokenizer to {repo_id}...")
tok.push_to_hub(repo_id, private=private)

print(f"📤 Pushing model to {repo_id}...")
mdl.push_to_hub(repo_id, private=private)

print(f"\n✅ Successfully pushed model to: https://huggingface.co/{repo_id}")
print(f"💡 You can now delete the local model to free up space:")
print(f"   rm -rf {model_dir}")

