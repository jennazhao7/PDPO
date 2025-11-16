#!/usr/bin/env python3
"""
Score Alpaca dataset with reward model to compute margins.
This script loads an Alpaca DPO dataset, scores responses with a reward model,
and computes margins (chosen_score - rejected_score).
"""

import math
import torch
import gc
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Configuration
MODEL_ID = "OpenAssistant/reward-model-deberta-v3-large-v2"   # Shelf-ready reward model
MAX_LEN = 512
BATCH = 32   # Batch size for scoring
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

def parse_args():
    parser = argparse.ArgumentParser(description="Score Alpaca dataset with reward model")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca_eval",
                        help="Alpaca dataset name or path")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use")
    parser.add_argument("--batch_size", type=int, default=BATCH,
                        help="Batch size for scoring")
    parser.add_argument("--output_dir", type=str, default="./alpaca_with_margins",
                        help="Output directory for scored dataset (saved to project root)")
    parser.add_argument("--dpo_file", type=str, default=None,
                        help="Optional: Pre-converted DPO JSONL file from project root (e.g., ./alpaca_eval_dpo.jsonl)")
    return parser.parse_args()

print(f"Using device: {DEVICE}")
print(f"Using dtype: {DTYPE}")
print(f"CUDA available: {torch.cuda.is_available()}")

def main():
    args = parse_args()
    
    # Check if we have a pre-converted DPO file
    if args.dpo_file:
        print(f"\nLoading pre-converted DPO dataset from: {args.dpo_file}")
        from datasets import Dataset
        import json
        examples = []
        with open(args.dpo_file, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        ds = Dataset.from_list(examples)
        print(f"✅ Loaded {len(ds)} examples from DPO file")
    else:
        # Load Alpaca dataset
        print(f"\nLoading Alpaca dataset: {args.dataset}")
        try:
            ds = load_dataset(args.dataset, trust_remote_code=True)
            print(f"✅ Loaded {args.dataset}")
        except Exception as e:
            print(f"❌ Could not load dataset {args.dataset}: {e}")
            raise

    # Get the right split - handle both DatasetDict and Dataset
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        # It's a DatasetDict with multiple splits
        if args.split in ds:
            ds = ds[args.split]
        elif "train" in ds:
            ds = ds["train"]
            print(f"⚠️  Using 'train' split instead of '{args.split}'")
        else:
            # Use first available split
            first_split = list(ds.keys())[0]
            ds = ds[first_split]
            print(f"⚠️  Using first available split: {first_split}")
    else:
        # It's already a Dataset object
        print(f"✅ Using dataset directly (single split)")

    print(f"Dataset size: {len(ds)} examples")
    print(f"Dataset columns: {ds.column_names}")
    
    # Check dataset format first
    sample = ds[0] if len(ds) > 0 else {}
    print(f"Sample keys: {list(sample.keys())}")
    
    # Canonicalize dataset columns to prompt/chosen/rejected format
    def canon_alpaca(ex):
        """Canonicalize Alpaca dataset to DPO format"""
        # Handle different Alpaca formats
        ex["prompt"] = ex.get("prompt") or ex.get("instruction") or ex.get("input") or ""
        
        # For DPO datasets, we should have chosen/rejected
        if "chosen" in ex:
            ex["pos"] = ex["chosen"]
            ex["neg"] = ex.get("rejected", "")
        elif "response_chosen" in ex:
            ex["pos"] = ex["response_chosen"]
            ex["neg"] = ex.get("response_rejected", "")
        elif "output_1" in ex and "output_2" in ex:
            ex["pos"] = ex["output_1"]
            ex["neg"] = ex["output_2"]
        elif "response_0" in ex and "response_1" in ex:
            ex["pos"] = ex["response_0"]
            ex["neg"] = ex["response_1"]
        else:
            # Try to find any output columns
            output_cols = [k for k in ex.keys() if "output" in k.lower() or "response" in k.lower()]
            if len(output_cols) >= 2:
                ex["pos"] = ex[output_cols[0]]
                ex["neg"] = ex[output_cols[1]]
            else:
                raise ValueError(
                    f"Dataset doesn't have chosen/rejected format.\n"
                    f"Available columns: {list(ex.keys())}\n"
                    f"Expected: prompt/chosen/rejected or prompt/response_chosen/response_rejected"
                )
        
        return ex

    # Apply canonicalization
    print("\nCanonicalizing dataset format...")
    try:
        ds = ds.map(canon_alpaca)
        print(f"✅ After canonicalization, columns: {ds.column_names}")
    except ValueError as e:
        print(f"❌ Error: {e}")
        print("\nPlease ensure your dataset has chosen/rejected pairs.")
        print("For alpaca_eval, you may need to use a different dataset or convert it to DPO format.")
        raise

    # Load frozen reward model
    print(f"\nLoading reward model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    rm = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    rm.eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    print("✅ Reward model loaded and frozen")

    @torch.no_grad()
    def score_batch(prompts, responses):
        """Score a batch of prompt-response pairs"""
        texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
        enc = tok(texts, return_tensors="pt", truncation=True, max_length=MAX_LEN, padding=True)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.autocast(device_type="cuda", dtype=DTYPE, enabled=(DEVICE == "cuda")):
            logits = rm(**enc).logits.squeeze(-1)  # [B]
        result = logits.float().cpu().tolist()
        # Clear GPU memory
        del enc, logits
        torch.cuda.empty_cache()
        return result

    def add_margins(batch):
        """Add margin scores to a batch of examples"""
        rp = score_batch(batch["prompt"], batch["pos"])
        rn = score_batch(batch["prompt"], batch["neg"])
        m = [pp - nn for pp, nn in zip(rp, rn)]
        return {
            "margin_raw": m,
            "margin_abs": [abs(x) for x in m],
            "reward_chosen": rp,
            "reward_rejected": rn
        }

    print(f"\nStarting scoring with batch size {args.batch_size}")
    if DEVICE == "cuda":
        print(f"GPU memory before processing: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Score the dataset
    ds_scored = ds.map(add_margins, batched=True, batch_size=args.batch_size)

    # Save the scored dataset
    print(f"\nSaving scored dataset to {args.output_dir}...")
    ds_scored.save_to_disk(args.output_dir)
    print(f"✅ Done! Saved to {args.output_dir}")

    if DEVICE == "cuda":
        print(f"GPU memory after processing: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Print summary statistics
    margins = [ex["margin_raw"] for ex in ds_scored]
    import numpy as np
    margins = np.array(margins)
    print(f"\n=== Margin Statistics ===")
    print(f"Total examples: {len(margins)}")
    print(f"Mean margin: {np.mean(margins):.4f}")
    print(f"Std margin: {np.std(margins):.4f}")
    print(f"Min margin: {np.min(margins):.4f}")
    print(f"Max margin: {np.max(margins):.4f}")

if __name__ == "__main__":
    main()
