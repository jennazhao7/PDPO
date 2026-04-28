#!/usr/bin/env python3
"""
Extract RR labels (ℓ_RR) from D2 dataset.
If D2 doesn't have 'flipped' field, match with privatized dataset to get it.
"""

import json
import argparse

def extract_rr_labels(d2_file, privatized_file, output_file):
    """
    Extract RR labels from D2 dataset.
    If D2 doesn't have 'flipped' field, match with privatized dataset.
    """
    print(f"📖 Loading D2 dataset from: {d2_file}")
    
    d2_examples = []
    with open(d2_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                d2_examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(d2_examples)} examples from D2")
    
    # Check if D2 has 'flipped' field
    has_flipped = any('flipped' in ex or 'label_rr' in ex for ex in d2_examples[:10])
    
    if not has_flipped and privatized_file:
        print(f"📖 D2 missing 'flipped' field. Loading privatized dataset to match...")
        # Load privatized dataset and create a lookup by prompt
        privatized_lookup = {}
        with open(privatized_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    prompt = ex['prompt']
                    privatized_lookup[prompt] = ex
        
        print(f"✅ Loaded {len(privatized_lookup)} examples from privatized dataset")
    else:
        privatized_lookup = None
    
    # Extract RR labels
    rr_examples = []
    matched_count = 0
    
    for i, d2_ex in enumerate(d2_examples):
        # Try to get RR label
        if 'flipped' in d2_ex:
            label_rr = 0 if d2_ex['flipped'] else 1
            flipped = d2_ex['flipped']
        elif 'label_rr' in d2_ex:
            label_rr = d2_ex['label_rr']
            flipped = (label_rr == 0)
        elif privatized_lookup and d2_ex['prompt'] in privatized_lookup:
            # Match with privatized dataset
            priv_ex = privatized_lookup[d2_ex['prompt']]
            label_rr = 0 if priv_ex.get('flipped', False) else 1
            flipped = priv_ex.get('flipped', False)
            matched_count += 1
        else:
            # Fallback: assume not flipped
            print(f"⚠️  Warning: Line {i+1} missing 'flipped' field and couldn't match")
            label_rr = 1
            flipped = False
        
        # Create output with RR label
        rr_example = {
            "prompt": d2_ex['prompt'],
            "chosen": d2_ex['chosen'],
            "rejected": d2_ex['rejected'],
            "label_rr": label_rr,
            "flipped": flipped,
        }
        
        # Preserve other fields if they exist
        if 'margin_normalized' in d2_ex:
            rr_example['margin_normalized'] = d2_ex['margin_normalized']
        elif privatized_lookup and d2_ex['prompt'] in privatized_lookup:
            priv_ex = privatized_lookup[d2_ex['prompt']]
            if 'margin_normalized' in priv_ex:
                rr_example['margin_normalized'] = priv_ex['margin_normalized']
        
        rr_examples.append(rr_example)
    
    if matched_count > 0:
        print(f"✅ Matched {matched_count} examples with privatized dataset")
    
    # Save RR labels
    print(f"\n💾 Saving RR labels to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in rr_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    rr_ones = sum(1 for x in rr_examples if x['label_rr'] == 1)
    rr_zeros = len(rr_examples) - rr_ones
    
    print(f"\n📊 RR Label Statistics:")
    print(f"   Total examples: {len(rr_examples)}")
    print(f"   ℓ_RR = 1: {rr_ones} ({rr_ones/len(rr_examples)*100:.1f}%)")
    print(f"   ℓ_RR = 0: {rr_zeros} ({rr_zeros/len(rr_examples)*100:.1f}%)")
    print(f"   Flipped: {sum(1 for x in rr_examples if x['flipped'])} ({sum(1 for x in rr_examples if x['flipped'])/len(rr_examples)*100:.1f}%)")
    print(f"✅ RR labels saved to: {output_file}")
    
    return rr_examples

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RR labels from D2 dataset")
    parser.add_argument("--d2_file", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/dpo_train_ready_d2.jsonl",
                       help="Path to D2 dataset JSONL")
    parser.add_argument("--privatized_file", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/dpo_privatized_dataset.jsonl",
                       help="Path to privatized dataset JSONL (for matching if D2 lacks 'flipped' field)")
    parser.add_argument("--output", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/d2_rr_labels.jsonl",
                       help="Output file for RR labels")
    
    args = parser.parse_args()
    
    extract_rr_labels(args.d2_file, args.privatized_file, args.output)
