#!/usr/bin/env python3
"""
Test script to load and verify two GPT-2 Large models from HuggingFace.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import getpass

def load_model(model_id, device="cuda"):
    """Load a model from HuggingFace with scratch directory caching."""
    print(f"\n{'='*60}")
    print(f"Loading model: {model_id}")
    print(f"{'='*60}")
    
    # Use scratch directory for cache if available
    cache_dir = None
    for scratch_dir in ['/scratch365', '/scratch']:
        if os.path.exists(scratch_dir):
            user_cache_dir = os.path.join(scratch_dir, getpass.getuser(), 'huggingface_cache')
            try:
                os.makedirs(user_cache_dir, exist_ok=True)
                # Test write permissions
                test_file = os.path.join(user_cache_dir, '.test_write')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    cache_dir = user_cache_dir
                    print(f"Using cache directory: {cache_dir}")
                    break
                except (PermissionError, OSError):
                    # Can't write here, skip
                    pass
            except (PermissionError, OSError):
                # Can't create directory, skip
                pass
    
    if cache_dir is None:
        print("Using default HuggingFace cache directory")
    
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Using dtype: {dtype}")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=False
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("✅ Tokenizer loaded")
        
        # Load model
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=None,
            trust_remote_code=False,
            cache_dir=cache_dir
        )
        model = model.to(device).eval()
        print(f"✅ Model loaded on {device}")
        
        # Print model info
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"   Parameters: ~{num_params:.1f}M")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Dtype: {next(model.parameters()).dtype}")
        
        # Test generation
        print("\nTesting generation...")
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   Prompt: {test_prompt}")
        print(f"   Generated: {generated_text}")
        print("✅ Generation test passed")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model {model_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    print("="*60)
    print("GPT-2 Large Model Loading Test")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Model IDs
    model_a_id = "Jennazhao7/gpt2-large-dpo-m1-v2"
    model_b_id = "Setpember/Jon_GPT2L_DPO_props_epi_point1"
    
    # Load Model A
    model_a, tokenizer_a = load_model(model_a_id, device)
    
    # Clear GPU memory before loading next model
    if torch.cuda.is_available():
        del model_a
        del tokenizer_a
        torch.cuda.empty_cache()
        print("\n🧹 Cleared GPU cache")
    
    # Load Model B
    model_b, tokenizer_b = load_model(model_b_id, device)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    if model_a:
        print(f"✅ Model A ({model_a_id}) loaded successfully")
    else:
        print(f"❌ Model A ({model_a_id}) failed to load")
    
    if model_b:
        print(f"✅ Model B ({model_b_id}) loaded successfully")
    else:
        print(f"❌ Model B ({model_b_id}) failed to load")
    
    print("="*60)

if __name__ == "__main__":
    main()

