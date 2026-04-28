#!/usr/bin/env python3
"""
Split TruthyDPO DPO dataset into D1 (first half) and D2 (second half).
Preserves the original data file.
"""

import json
import os
import argparse
from pathlib import Path

def split_dataset(input_file, output_d1, output_d2, seed=42):
    """
    Randomly split a JSONL dataset into D1 and D2 (50/50 split).
    
    Args:
        input_file: Path to input JSONL file
        output_d1: Path to output file for D1
        output_d2: Path to output file for D2
        seed: Random seed for reproducibility (default: 42)
    """
    import random
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found!")
    
    print(f"📖 Loading data from {input_file}...")
    
    # Load all examples
    examples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(data)
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    total_examples = len(examples)
    print(f"✅ Loaded {total_examples} examples")
    
    if total_examples == 0:
        raise ValueError("No valid examples found in input file!")
    
    # Set random seed for reproducibility
    random.seed(seed)
    print(f"🎲 Using random seed={seed} for splitting")
    
    # Randomly shuffle all examples
    random.shuffle(examples)
    print(f"🔀 Shuffled dataset")
    
    # Split into two halves (50/50)
    split_point = total_examples // 2
    d1_examples = examples[:split_point]
    d2_examples = examples[split_point:]
    
    print(f"\n📊 Split statistics:")
    print(f"   Total examples: {total_examples}")
    print(f"   D1 (random half): {len(d1_examples)} examples ({len(d1_examples)/total_examples*100:.1f}%)")
    print(f"   D2 (random half): {len(d2_examples)} examples ({len(d2_examples)/total_examples*100:.1f}%)")
    
    # Save D1
    print(f"\n💾 Saving D1 to {output_d1}...")
    with open(output_d1, 'w', encoding='utf-8') as f:
        for example in d1_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"   ✅ Saved {len(d1_examples)} examples to D1")
    
    # Save D2
    print(f"\n💾 Saving D2 to {output_d2}...")
    with open(output_d2, 'w', encoding='utf-8') as f:
        for example in d2_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    print(f"   ✅ Saved {len(d2_examples)} examples to D2")
    
    # Verify original file is preserved
    if os.path.exists(input_file):
        print(f"\n✅ Original file preserved: {input_file}")
    
    print(f"\n🎉 Random split complete!")
    print(f"   D1: {output_d1} ({len(d1_examples)} examples)")
    print(f"   D2: {output_d2} ({len(d2_examples)} examples)")
    print(f"   Original: {input_file} ({total_examples} examples - preserved)")
    print(f"   Random seed used: {seed}")

def main():
    parser = argparse.ArgumentParser(description="Split DPO dataset into D1 and D2")
    parser.add_argument(
        "--input",
        type=str,
        default="dpo_train_ready.jsonl",
        help="Input JSONL file (default: dpo_train_ready.jsonl)"
    )
    parser.add_argument(
        "--output_d1",
        type=str,
        default="dpo_train_ready_d1.jsonl",
        help="Output file for D1 (default: dpo_train_ready_d1.jsonl)"
    )
    parser.add_argument(
        "--output_d2",
        type=str,
        default="dpo_train_ready_d2.jsonl",
        help="Output file for D2 (default: dpo_train_ready_d2.jsonl)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for random splitting (default: 42)"
    )
    
    args = parser.parse_args()
    
    split_dataset(args.input, args.output_d1, args.output_d2, seed=args.seed)

if __name__ == "__main__":
    main()

