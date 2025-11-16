#!/usr/bin/env python3
"""
Score alpaca_eval dataset with a frozen reward model.
This script scores each output in alpaca_eval with a reward model.
"""

import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import json

# Configuration
DEFAULT_RM = "OpenAssistant/reward-model-deberta-v3-large-v2"  # Shelf-ready reward model
MAX_LEN = 512
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def parse_args():
    parser = argparse.ArgumentParser(description="Score alpaca_eval with reward model")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca_eval",
                        help="Alpaca eval dataset name")
    parser.add_argument("--split", type=str, default="eval",
                        help="Dataset split")
    parser.add_argument("--reward_model", type=str, default=DEFAULT_RM,
                        help=f"Reward model to use (default: {DEFAULT_RM})")
    parser.add_argument("--batch_size", type=int, default=BATCH,
                        help="Batch size for scoring")
    parser.add_argument("--output_file", type=str, default="./alpaca_eval_scored.jsonl",
                        help="Output JSONL file with scores")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Using device: {DEVICE}")
    print(f"Using dtype: {DTYPE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Load alpaca_eval dataset
    print(f"\nLoading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, trust_remote_code=True)
    
    # Get the right split
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        if args.split in ds:
            ds = ds[args.split]
        else:
            ds = ds[list(ds.keys())[0]]
            print(f"⚠️  Using split: {list(ds.keys())[0]}")
    else:
        print(f"✅ Using dataset directly")
    
    print(f"Dataset size: {len(ds)} examples")
    print(f"Dataset columns: {ds.column_names}")
    
    # Load reward model
    print(f"\nLoading reward model: {args.reward_model}")
    tok = AutoTokenizer.from_pretrained(args.reward_model, use_fast=True)
    rm = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model, 
        torch_dtype=DTYPE
    ).to(DEVICE)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    print("✅ Reward model loaded and frozen")
    
    # Set pad token if needed
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    @torch.no_grad()
    def score_batch(prompts, responses):
        """Score a batch of prompt-response pairs"""
        texts = [f"{p}\n{r}" if p else r for p, r in zip(prompts, responses)]
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
            logits = rm(**enc).logits.squeeze(-1)  # [B]
        result = logits.float().cpu().tolist()
        # Clear GPU memory
        del enc, logits
        torch.cuda.empty_cache()
        return result
    
    # Score all examples
    print(f"\nScoring {len(ds)} examples with batch size {args.batch_size}...")
    if DEVICE == "cuda":
        print(f"GPU memory before: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    scored_examples = []
    for i in tqdm(range(0, len(ds), args.batch_size), desc="Scoring"):
        # Get batch indices
        batch_indices = list(range(i, min(i+args.batch_size, len(ds))))
        batch = [ds[idx] for idx in batch_indices]
        
        prompts = [ex.get("instruction", "") for ex in batch]
        outputs = [ex.get("output", "") for ex in batch]
        
        scores = score_batch(prompts, outputs)
        
        # Add scores to examples
        for j, ex in enumerate(batch):
            scored_ex = dict(ex)
            scored_ex["reward_score"] = scores[j]
            scored_examples.append(scored_ex)
    
    # Save scored dataset
    print(f"\nSaving scored dataset to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for ex in scored_examples:
            f.write(json.dumps(ex) + '\n')
    
    if DEVICE == "cuda":
        print(f"GPU memory after: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    # Print statistics
    scores = [ex["reward_score"] for ex in scored_examples]
    import numpy as np
    scores = np.array(scores)
    print(f"\n=== Scoring Statistics ===")
    print(f"Total examples scored: {len(scored_examples)}")
    print(f"Mean score: {np.mean(scores):.4f}")
    print(f"Std score: {np.std(scores):.4f}")
    print(f"Min score: {np.min(scores):.4f}")
    print(f"Max score: {np.max(scores):.4f}")
    print(f"\n✅ Scored dataset saved to: {args.output_file}")

if __name__ == "__main__":
    main()

