#!/usr/bin/env python3
"""
Convert alpaca_eval dataset to DPO format by pairing outputs from different generators.
This script groups by instruction and creates chosen/rejected pairs from different model outputs.
"""

import argparse
from datasets import load_dataset
from collections import defaultdict
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Convert alpaca_eval to DPO format")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca_eval",
                        help="Alpaca eval dataset name")
    parser.add_argument("--split", type=str, default="eval",
                        help="Dataset split to use")
    parser.add_argument("--output_file", type=str, default="./alpaca_eval_dpo.jsonl",
                        help="Output JSONL file (saved to project root)")
    parser.add_argument("--reference_generator", type=str, default="gpt4",
                        help="Reference generator to use as baseline (default: gpt4)")
    parser.add_argument("--min_pairs", type=int, default=2,
                        help="Minimum number of outputs per instruction to create pairs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading dataset: {args.dataset}")
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
    
    # Group by instruction
    print("\nGrouping by instruction...")
    grouped = defaultdict(list)
    all_generators = set()
    for ex in ds:
        instruction = ex.get("instruction", "")
        output = ex.get("output", "")
        generator = ex.get("generator", "unknown")
        all_generators.add(generator)
        grouped[instruction].append({
            "output": output,
            "generator": generator
        })
    
    print(f"Found {len(grouped)} unique instructions")
    print(f"Available generators: {sorted(all_generators)}")
    
    # Check if reference generator exists
    if args.reference_generator not in all_generators:
        print(f"\n⚠️  Warning: Reference generator '{args.reference_generator}' not found!")
        print(f"   Available generators: {sorted(all_generators)}")
        if len(all_generators) > 0:
            # Use first generator as reference
            suggested_ref = sorted(all_generators)[0]
            print(f"   Using '{suggested_ref}' as reference instead")
            args.reference_generator = suggested_ref
        else:
            raise ValueError("No generators found in dataset!")
    
    # Create DPO pairs
    print(f"\nCreating DPO pairs (using {args.reference_generator} as reference)...")
    dpo_examples = []
    stats = {
        "total_instructions": len(grouped),
        "pairs_created": 0,
        "skipped_no_reference": 0,
        "skipped_insufficient_outputs": 0
    }
    
    # Debug: Check distribution of outputs per instruction
    output_counts = [len(outputs) for outputs in grouped.values()]
    if len(output_counts) > 0:
        print(f"\nOutputs per instruction statistics:")
        print(f"  Min: {min(output_counts)}")
        print(f"  Max: {max(output_counts)}")
        print(f"  Mean: {sum(output_counts)/len(output_counts):.2f}")
        print(f"  Instructions with >= 2 outputs: {sum(1 for c in output_counts if c >= 2)}")
    
    for instruction, outputs in grouped.items():
        if len(outputs) < args.min_pairs:
            stats["skipped_insufficient_outputs"] += 1
            continue
        
        # Find reference output (prefer specified generator)
        reference_output = None
        reference_generator = None
        other_outputs = []
        
        for out in outputs:
            if out["generator"] == args.reference_generator:
                reference_output = out["output"]
                reference_generator = out["generator"]
                break
        
        if reference_output is None:
            # Use first output as reference if specified generator not found
            reference_output = outputs[0]["output"]
            reference_generator = outputs[0]["generator"]
            other_outputs = [{"output": out["output"], "generator": out["generator"]} for out in outputs[1:]]
            stats["skipped_no_reference"] += 1
        else:
            # Collect all other outputs
            other_outputs = [{"output": out["output"], "generator": out["generator"]} 
                           for out in outputs if out["generator"] != args.reference_generator]
        
        # Create pairs: reference is chosen, others are rejected
        # For DPO, we typically want the better output as chosen
        # Here we'll create pairs where reference is chosen
        for other in other_outputs:
            dpo_examples.append({
                "prompt": instruction,
                "chosen": reference_output,
                "rejected": other["output"],
                "reference_generator": reference_generator,
                "other_generator": other["generator"]
            })
            stats["pairs_created"] += 1
    
    # Save DPO dataset
    print(f"\nSaving DPO dataset to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for ex in dpo_examples:
            f.write(json.dumps(ex) + '\n')
    
    # Print statistics
    print("\n=== Conversion Statistics ===")
    print(f"Total instructions: {stats['total_instructions']}")
    print(f"Pairs created: {stats['pairs_created']}")
    print(f"Skipped (no reference): {stats['skipped_no_reference']}")
    print(f"Skipped (insufficient outputs): {stats['skipped_insufficient_outputs']}")
    print(f"\n✅ DPO dataset saved to: {args.output_file}")
    print(f"   Total examples: {len(dpo_examples)}")

if __name__ == "__main__":
    main()

