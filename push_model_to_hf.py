#!/usr/bin/env python3
"""
Push a local model (or merged LoRA) to Hugging Face Hub.

Usage:
  python push_model_to_hf.py \
    --model_dir /path/to/model \
    --repo_id your-username/your-repo \
    [--base_model_id base-for-lora] \
    [--public] \
    [--hf_token $HF_TOKEN]
"""

import os
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try to import peft for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  Warning: peft not available. LoRA adapters cannot be loaded.")


def parse_args():
    ap = argparse.ArgumentParser(description="Push a local model (or merged LoRA) to Hugging Face Hub.")
    ap.add_argument("--model_dir", required=True, help="Path to model directory (or LoRA adapter).")
    ap.add_argument("--repo_id", required=True, help="Destination repo on HF, e.g., user/model-name.")
    ap.add_argument("--base_model_id", default=None, help="Base model for LoRA adapters (optional, auto-detected if possible).")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN env).")
    ap.add_argument("--public", action="store_true", help="If set, push as public (default is private).")
    return ap.parse_args()


args = parse_args()
model_dir = args.model_dir
repo_id = args.repo_id
base_model_id = args.base_model_id
private = not args.public

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
if args.hf_token:
    os.environ["HF_TOKEN"] = args.hf_token
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

# Push additional files that might not be included automatically
print(f"📤 Pushing additional files...")
from huggingface_hub import HfApi
api = HfApi()

model_path = Path(model_dir)
# Files that push_to_hub should already handle (but we verify they exist)
standard_files = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "model.safetensors",
]

# Additional files that might not be pushed automatically
additional_files = [
    "README.md",
    "training_args.bin",
]

# Verify standard files exist
print("   Verifying standard files...")
for file_name in standard_files:
    file_path = model_path / file_name
    if not file_path.exists() and file_name != "model.safetensors":  # model.safetensors might be in subdir
        # Check for pytorch_model.bin as alternative
        if file_name == "model.safetensors":
            if not (model_path / "pytorch_model.bin").exists():
                print(f"   ⚠️  Warning: {file_name} not found (model weights should be present)")

# Push additional files explicitly
for file_name in additional_files:
    file_path = model_path / file_name
    if file_path.exists():
        try:
            print(f"   Uploading {file_name}...")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="model",
                token=os.environ.get("HF_TOKEN"),
            )
            print(f"   ✅ {file_name} uploaded")
        except Exception as e:
            print(f"   ⚠️  Could not upload {file_name}: {e}")

print(f"\n✅ Successfully pushed model to: https://huggingface.co/{repo_id}")
if is_lora:
    print(f"💡 Note: Pushed merged model (base + LoRA combined)")
print(f"💡 Files pushed:")
print(f"   - Model weights (model.safetensors)")
print(f"   - Config files (config.json, tokenizer files)")
print(f"   - Additional files (README.md, training_args.bin, etc.)")
print(f"\n💡 You can now delete the local model to free up space:")
print(f"   rm -rf {model_dir}")

