#!/usr/bin/env python3
"""
Process margins for alpaca_eval scored dataset:
1. Group by instruction and create DPO pairs based on reward scores
2. Compute margins (chosen_score - rejected_score)
3. Clip margins to bounded range for DP sensitivity
4. Normalize margins
"""

import argparse
import json
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Process margins for alpaca_eval")
    parser.add_argument("--input_file", type=str, default="./alpaca_eval_scored.jsonl",
                        help="Input JSONL file with reward scores")
    parser.add_argument("--output_file", type=str, default="./alpaca_eval_dpo.jsonl",
                        help="Output JSONL file with DPO pairs and processed margins")
    parser.add_argument("--clip_percentile_low", type=float, default=1.0,
                        help="Lower percentile for clipping (e.g., 1.0 = 1st percentile)")
    parser.add_argument("--clip_percentile_high", type=float, default=99.0,
                        help="Upper percentile for clipping (e.g., 99.0 = 99th percentile)")
    parser.add_argument("--clip_min", type=float, default=None,
                        help="Optional: Override lower clip bound (if None, uses percentile)")
    parser.add_argument("--clip_max", type=float, default=None,
                        help="Optional: Override upper clip bound (if None, uses percentile)")
    parser.add_argument("--min_pairs_per_instruction", type=int, default=2,
                        help="Minimum number of outputs per instruction to create pairs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load scored dataset
    print(f"Loading scored dataset from {args.input_file}...")
    scored_examples = []
    with open(args.input_file, 'r') as f:
        for line in f:
            scored_examples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(scored_examples)} examples")
    
    # Group by instruction
    print("\nGrouping by instruction...")
    grouped = defaultdict(list)
    for ex in scored_examples:
        instruction = ex.get("instruction", "")
        output = ex.get("output", "")
        score = ex.get("reward_score", 0.0)
        generator = ex.get("generator", "unknown")
        grouped[instruction].append({
            "output": output,
            "score": score,
            "generator": generator,
            "original_data": ex
        })
    
    print(f"Found {len(grouped)} unique instructions")
    
    # Statistics on outputs per instruction
    output_counts = [len(outputs) for outputs in grouped.values()]
    print(f"\nOutputs per instruction statistics:")
    print(f"  Min: {min(output_counts)}")
    print(f"  Max: {max(output_counts)}")
    print(f"  Mean: {sum(output_counts)/len(output_counts):.2f}")
    print(f"  Instructions with >= {args.min_pairs_per_instruction} outputs: {sum(1 for c in output_counts if c >= args.min_pairs_per_instruction)}")
    
    # Create DPO pairs: highest score = chosen, others = rejected
    print(f"\nCreating DPO pairs (highest score = chosen)...")
    dpo_examples = []
    margins_raw = []
    
    stats = {
        "total_instructions": len(grouped),
        "pairs_created": 0,
        "skipped_insufficient": 0
    }
    
    for instruction, outputs in grouped.items():
        if len(outputs) < args.min_pairs_per_instruction:
            stats["skipped_insufficient"] += 1
            continue
        
        # Sort by score (highest first)
        sorted_outputs = sorted(outputs, key=lambda x: x["score"], reverse=True)
        
        # Highest score is chosen, create pairs with all others as rejected
        chosen = sorted_outputs[0]
        for rejected in sorted_outputs[1:]:
            margin = chosen["score"] - rejected["score"]
            margins_raw.append(margin)
            
            dpo_examples.append({
                "prompt": instruction,
                "chosen": chosen["output"],
                "rejected": rejected["output"],
                "chosen_generator": chosen["generator"],
                "rejected_generator": rejected["generator"],
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
                "margin_raw": margin
            })
            stats["pairs_created"] += 1
    
    print(f"\nPairs created: {stats['pairs_created']}")
    print(f"Skipped (insufficient outputs): {stats['skipped_insufficient']}")
    
    if len(margins_raw) == 0:
        print("\n❌ ERROR: No pairs created!")
        print("\nReason: Each instruction only has 1 output. To create DPO pairs, you need multiple outputs per instruction.")
        print("\nSolutions:")
        print("1. Score outputs from multiple generators/models:")
        print("   - Run score_alpaca_eval_rm.py on outputs from different models")
        print("   - Combine scored outputs from multiple generators")
        print("   - Then run this script again")
        print("\n2. Use a reference model to generate additional outputs:")
        print("   - Generate outputs for each instruction using a reference model")
        print("   - Score those outputs")
        print("   - Create pairs: your_model_output vs reference_output")
        print("\n3. Check if your alpaca_eval dataset has multiple generators:")
        print("   - Original alpaca_eval should have outputs from multiple models")
        print("   - Make sure you're scoring outputs from all generators")
        raise ValueError("No pairs created! Each instruction needs multiple outputs to create DPO pairs.")
    
    margins_raw = np.array(margins_raw)
    
    # Print raw margin statistics
    print(f"\n=== Raw Margin Statistics ===")
    print(f"Total pairs: {len(margins_raw)}")
    print(f"Mean margin: {np.mean(margins_raw):.4f}")
    print(f"Std margin: {np.std(margins_raw):.4f}")
    print(f"Min margin: {np.min(margins_raw):.4f}")
    print(f"Max margin: {np.max(margins_raw):.4f}")
    
    # Step 1: Determine clipping bounds (percentile-based or fixed)
    if args.clip_min is not None and args.clip_max is not None:
        # Use fixed bounds if provided
        clip_min = args.clip_min
        clip_max = args.clip_max
        print(f"\nUsing fixed clipping bounds: [{clip_min}, {clip_max}]")
    else:
        # Use percentile-based bounds
        clip_min = np.percentile(margins_raw, args.clip_percentile_low)
        clip_max = np.percentile(margins_raw, args.clip_percentile_high)
        print(f"\nUsing percentile-based clipping bounds:")
        print(f"  {args.clip_percentile_low}th percentile: {clip_min:.4f}")
        print(f"  {args.clip_percentile_high}th percentile: {clip_max:.4f}")
        print(f"  Clipping range: [{clip_min:.4f}, {clip_max:.4f}]")
    
    # Clip margins to bounded range for DP sensitivity
    print(f"\nClipping margins to [{clip_min:.4f}, {clip_max:.4f}]...")
    margins_clipped = np.clip(margins_raw, clip_min, clip_max)
    
    print(f"Clipped margin statistics:")
    print(f"  Mean: {np.mean(margins_clipped):.4f}")
    print(f"  Std: {np.std(margins_clipped):.4f}")
    print(f"  Min: {np.min(margins_clipped):.4f}")
    print(f"  Max: {np.max(margins_clipped):.4f}")
    
    # Count how many were clipped
    clipped_count = np.sum((margins_raw != margins_clipped))
    print(f"  Examples clipped: {clipped_count} ({clipped_count/len(margins_raw)*100:.1f}%)")
    
    # Step 2: Normalize the clipped margins (same formula as truthydpo)
    print("\nNormalizing clipped margins...")
    margins_normalized = (margins_clipped - np.mean(margins_clipped)) / np.std(margins_clipped)
    
    print(f"Normalized margin statistics:")
    print(f"  Mean: {np.mean(margins_normalized):.4f}")
    print(f"  Std: {np.std(margins_normalized):.4f}")
    print(f"  Min: {np.min(margins_normalized):.4f}")
    print(f"  Max: {np.max(margins_normalized):.4f}")
    
    # Add processed margins to DPO examples
    print("\nAdding processed margins to DPO examples...")
    for i, ex in enumerate(dpo_examples):
        ex["margin_clipped"] = float(margins_clipped[i])
        ex["margin_normalized"] = float(margins_normalized[i])
    
    # Save DPO dataset with processed margins
    print(f"\nSaving DPO dataset with processed margins to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for ex in dpo_examples:
            f.write(json.dumps(ex) + '\n')
    
    print(f"✅ Done! Saved {len(dpo_examples)} DPO pairs to {args.output_file}")
    
    # Show example
    print(f"\nExample processed margins:")
    example = dpo_examples[0]
    print(f"  Instruction: {example['prompt'][:80]}...")
    print(f"  Raw margin: {example['margin_raw']:.4f}")
    print(f"  Clipped margin: {example['margin_clipped']:.4f}")
    print(f"  Normalized margin: {example['margin_normalized']:.4f}")

if __name__ == "__main__":
    main()

