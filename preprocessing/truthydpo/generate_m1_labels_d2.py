#!/usr/bin/env python3
"""
Generate M1 labels (ℓ_M1) for D2 dataset using the M1 model trained on D1.
M1 model should be trained on D1 first.
"""

import json
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from tqdm import tqdm

# Workaround: Set environment variable to prefer safetensors
os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")

def load_model_and_tokenizer(model_path, device="cuda"):
    """Load M1 model and tokenizer"""
    print(f"📦 Loading M1 model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=False,
    ).to(device).eval()
    
    print(f"✅ Model loaded on {device}")
    return model, tokenizer

@torch.no_grad()
def score_responses_with_rm(prompt, chosen_response, rejected_response, 
                            rm_model, rm_tokenizer, device="cuda"):
    """
    Score responses using reward model.
    Returns: (chosen_score, rejected_score)
    """
    # Format prompts + responses
    chosen_text = f"{prompt}{chosen_response}"
    rejected_text = f"{prompt}{rejected_response}"
    
    # Tokenize for reward model
    chosen_enc = rm_tokenizer(chosen_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    rejected_enc = rm_tokenizer(rejected_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    # Score with reward model
    with torch.no_grad():
        chosen_score = rm_model(**chosen_enc).logits.item()
        rejected_score = rm_model(**rejected_enc).logits.item()
    
    return chosen_score, rejected_score

def generate_m1_labels(d2_file, m1_model_path, output_file, rm_model_path=None, device="cuda"):
    """
    Generate M1 labels for D2 dataset.
    
    For each example in D2:
    1. Score original chosen/rejected responses using reward model
    2. Compare scores to determine ℓ_M1 (1 if chosen > rejected, 0 otherwise)
    """
    print(f"📖 Loading D2 dataset from: {d2_file}")
    
    # Load D2 examples
    examples = []
    with open(d2_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(examples)} examples from D2")
    
    # Load reward model if provided
    if rm_model_path:
        print(f"📦 Loading reward model from: {rm_model_path}")
        print(f"   (Will use cached model if available)")
        rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_path)
        rm_tokenizer.pad_token = rm_tokenizer.eos_token
        
        # Load model - use same approach as test_rm.py which works
        # The model is already cached with safetensors, so transformers will use it automatically
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        try:
            # Use the same approach as test_rm.py (which works with cached model)
            # transformers will automatically use cache and prefer safetensors if available
            rm_model = AutoModelForSequenceClassification.from_pretrained(
                rm_model_path,
                num_labels=1,
                torch_dtype=dtype,  # Use torch_dtype like test_rm.py (works with cache)
            ).to(device).eval()
            print(f"✅ Reward model loaded from cache")
        except ValueError as e:
            # Handle torch version restriction error
            if "torch.load" in str(e) or "torch to at least v2.6" in str(e):
                print(f"⚠️  Torch version check issue detected, trying with safetensors preference...")
                try:
                    # Try explicitly requesting safetensors (available in cache)
                    rm_model = AutoModelForSequenceClassification.from_pretrained(
                        rm_model_path,
                        num_labels=1,
                        torch_dtype=dtype,
                        use_safetensors=True,  # Force safetensors (available in cache)
                    ).to(device).eval()
                    print(f"✅ Reward model loaded from cache (safetensors)")
                except Exception as e2:
                    print(f"❌ Error loading reward model: {e2}")
                    print(f"\n💡 The model is cached at:")
                    print(f"   ~/.cache/huggingface/hub/models--OpenAssistant--reward-model-deberta-v3-large-v2")
                    print(f"   And has safetensors files available.")
                    print(f"\n   Try: pip install safetensors")
                    raise
            else:
                raise
        except Exception as e:
            print(f"❌ Error loading reward model: {e}")
            print(f"\n💡 The model should be cached. Check:")
            print(f"   ~/.cache/huggingface/hub/models--OpenAssistant--reward-model-deberta-v3-large-v2")
            raise
    else:
        rm_model = None
        rm_tokenizer = None
        print(f"⚠️  No reward model provided. Will use a simple heuristic.")
    
    # Generate M1 labels
    print(f"\n🔍 Generating M1 labels for D2...")
    m1_labels = []
    
    for i, example in enumerate(tqdm(examples, desc="Generating M1 labels")):
        prompt = example['prompt']
        chosen = example['chosen']
        rejected = example['rejected']
        
        if rm_model is not None:
            # Use reward model to score original responses
            chosen_score, rejected_score = score_responses_with_rm(
                prompt, chosen, rejected,
                rm_model, rm_tokenizer, device
            )
            
            # ℓ_M1 = 1 if chosen > rejected, 0 otherwise
            label_m1 = 1 if chosen_score > rejected_score else 0
        else:
            # Fallback: simple heuristic (prefer longer response)
            label_m1 = 1 if len(chosen) > len(rejected) else 0
        
        m1_labels.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "label_m1": label_m1,
            "original_index": i
        })
    
    # Save M1 labels
    print(f"\n💾 Saving M1 labels to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for label_data in m1_labels:
            f.write(json.dumps(label_data, ensure_ascii=False) + '\n')
    
    # Print statistics
    m1_ones = sum(1 for x in m1_labels if x['label_m1'] == 1)
    m1_zeros = len(m1_labels) - m1_ones
    print(f"\n📊 M1 Label Statistics:")
    print(f"   ℓ_M1 = 1: {m1_ones} ({m1_ones/len(m1_labels)*100:.1f}%)")
    print(f"   ℓ_M1 = 0: {m1_zeros} ({m1_zeros/len(m1_labels)*100:.1f}%)")
    print(f"✅ M1 labels saved to: {output_file}")
    
    return m1_labels

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate M1 labels for D2 dataset")
    parser.add_argument("--d2_file", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/dpo_train_ready_d2.jsonl",
                       help="Path to D2 dataset JSONL file")
    parser.add_argument("--m1_model", type=str,
                       default="/users/jzhao7/PDPO/models/truthydpo/gpt2-medium-dpo-truthydpo-d1",
                       help="Path to M1 model (trained on D1)")
    parser.add_argument("--output", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/d2_m1_labels.jsonl",
                       help="Output file for M1 labels")
    parser.add_argument("--rm_model", type=str, default="OpenAssistant/reward-model-deberta-v3-large-v2",
                       help="Reward model path (for scoring)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    generate_m1_labels(
        args.d2_file,
        args.m1_model,
        args.output,
        args.rm_model,
        args.device
    )

