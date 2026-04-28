#!/usr/bin/env python3
"""
Script to push Pythia LoRA models to HuggingFace Hub.
For LoRA adapters, it will merge them before pushing.
Just change the model_dir and repo_id variables below.
"""

import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import peft for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  Warning: peft not available. LoRA adapters cannot be loaded.")

# ===== CONFIGURATION - Change these =====
model_dir = "/users/jzhao7/PDPO/models/hhrlhf/pythia1b-hh-m1"  # Your LoRA adapter directory
repo_id = "Jennazhao7/pythia1b-hh-m1"  # Your HuggingFace repo ID
base_model_id = "EleutherAI/pythia-1b"  # Base model for LoRA (Pythia-1b)
private = True
# =========================================

# Check if model_dir has checkpoints (if no config.json or adapter_config.json in root)
model_path = Path(model_dir)
has_config = (model_path / "config.json").exists()
has_adapter_config = (model_path / "adapter_config.json").exists()

if not has_config and not has_adapter_config:
    # Look for checkpoints
    checkpoints = sorted([d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")])
    if checkpoints:
        latest_checkpoint = checkpoints[-1]
        print(f"⚠️  No final model found, using latest checkpoint: {latest_checkpoint.name}")
        model_dir = str(latest_checkpoint)
        model_path = Path(model_dir)
        has_config = (model_path / "config.json").exists()
        has_adapter_config = (model_path / "adapter_config.json").exists()
    else:
        print(f"❌ Error: No model files found in {model_dir}")
        exit(1)

# Check if it's a LoRA adapter
is_lora = has_adapter_config and not has_config

print(f"📦 Loading model from: {model_dir}")
print(f"🚀 Pushing to: {repo_id}")
print(f"🔒 Private: {private}")
if is_lora:
    print(f"🔧 Detected LoRA adapter - will merge before pushing")
print()

# Load tokenizer
print("Loading tokenizer...")
try:
    tok = AutoTokenizer.from_pretrained(model_dir)
except Exception:
    # If tokenizer not found in LoRA dir, try base model
    if is_lora and base_model_id:
        print(f"   Tokenizer not found in adapter dir, loading from base model...")
        tok = AutoTokenizer.from_pretrained(base_model_id)
    else:
        raise

# Load model
print("Loading model...")
if is_lora:
    if not PEFT_AVAILABLE:
        print(f"❌ Error: LoRA adapter detected but peft is not installed!")
        print(f"   Install with: pip install peft")
        exit(1)
    
    if not base_model_id:
        # Try to get base model from adapter config
        try:
            config = PeftConfig.from_pretrained(model_dir)
            base_model_id = config.base_model_name_or_path
            print(f"   Auto-detected base model: {base_model_id}")
        except Exception as e:
            print(f"❌ Error: Could not auto-detect base model from adapter config.")
            print(f"   Please specify base_model_id in the script.")
            exit(1)
    
    print(f"   Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    
    print(f"   Loading LoRA adapter: {model_dir}")
    model = PeftModel.from_pretrained(base_model, model_dir)
    
    print(f"   Merging LoRA adapter into base model...")
    mdl = model.merge_and_unload()
    print(f"   ✅ Model merged successfully")
else:
    # Regular model
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

# Push to HuggingFace
print(f"\n📤 Pushing tokenizer to {repo_id}...")
tok.push_to_hub(repo_id, private=private)

print(f"📤 Pushing model to {repo_id}...")
mdl.push_to_hub(repo_id, private=private)

print(f"\n✅ Successfully pushed model to: https://huggingface.co/{repo_id}")
if is_lora:
    print(f"💡 Note: Pushed merged model (base + LoRA combined)")
print(f"💡 You can now delete the local model to free up space:")
print(f"   rm -rf {model_dir}")

