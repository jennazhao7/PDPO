#!/usr/bin/env python3
"""
Process margins for Alpaca dataset: clip and normalize margins.
This script takes the scored dataset and processes margins for margin-aware flipping.
"""

import numpy as np
import argparse
from datasets import load_from_disk

def parse_args():
    parser = argparse.ArgumentParser(description="Process margins for Alpaca dataset")
    parser.add_argument("--input_dir", type=str, default="./alpaca_with_margins",
                        help="Input directory with scored dataset (from project root)")
    parser.add_argument("--output_dir", type=str, default="./alpaca_with_processed_margins",
                        help="Output directory for processed dataset (saved to project root)")
    parser.add_argument("--clip_min", type=float, default=-6.0,
                        help="Minimum value for clipping margins")
    parser.add_argument("--clip_max", type=float, default=6.0,
                        help="Maximum value for clipping margins")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the scored dataset
    print(f"Loading scored dataset from {args.input_dir}...")
    ds = load_from_disk(args.input_dir)
    
    # Handle different dataset structures
    if isinstance(ds, dict) and "train" in ds:
        ds = ds["train"]
    elif not hasattr(ds, "__getitem__"):
        raise ValueError("Unexpected dataset structure")
    
    margins = np.array(ds['margin_raw'])
    
    print(f"Original margin statistics:")
    print(f"  Count: {len(margins)}")
    print(f"  Mean: {np.mean(margins):.4f}")
    print(f"  Std: {np.std(margins):.4f}")
    print(f"  Min: {np.min(margins):.4f}")
    print(f"  Max: {np.max(margins):.4f}")
    
    # Step 1: Clip margins
    print(f"\nClipping margins to [{args.clip_min}, {args.clip_max}]...")
    margins_clipped = np.clip(margins, args.clip_min, args.clip_max)
    
    print(f"Clipped margin statistics:")
    print(f"  Mean: {np.mean(margins_clipped):.4f}")
    print(f"  Std: {np.std(margins_clipped):.4f}")
    print(f"  Min: {np.min(margins_clipped):.4f}")
    print(f"  Max: {np.max(margins_clipped):.4f}")
    
    # Count how many were clipped
    clipped_count = np.sum((margins != margins_clipped))
    print(f"  Examples clipped: {clipped_count} ({clipped_count/len(margins)*100:.1f}%)")
    
    # Step 2: Normalize the clipped margins
    print("\nNormalizing clipped margins...")
    margins_normalized = (margins_clipped - np.mean(margins_clipped)) / np.std(margins_clipped)
    
    print(f"Normalized margin statistics:")
    print(f"  Mean: {np.mean(margins_normalized):.4f}")
    print(f"  Std: {np.std(margins_normalized):.4f}")
    print(f"  Min: {np.min(margins_normalized):.4f}")
    print(f"  Max: {np.max(margins_normalized):.4f}")
    
    # Add new columns to dataset
    print("\nAdding new columns to dataset...")
    def add_processed_margins(example, idx):
        return {
            "margin_clipped": float(margins_clipped[idx]),
            "margin_normalized": float(margins_normalized[idx])
        }
    
    ds_processed = ds.map(add_processed_margins, with_indices=True)
    
    # Save the enhanced dataset
    print(f"Saving enhanced dataset to {args.output_dir}...")
    ds_processed.save_to_disk(args.output_dir)
    print(f"✅ Done! Saved to {args.output_dir}")
    
    # Show example of the new columns
    print(f"\nExample processed margins:")
    example = ds_processed[0]
    print(f"  Original margin: {example['margin_raw']:.4f}")
    print(f"  Clipped margin: {example['margin_clipped']:.4f}")
    print(f"  Normalized margin: {example['margin_normalized']:.4f}")
    
    print(f"\nDataset now has columns: {ds_processed.column_names}")

if __name__ == "__main__":
    main()

