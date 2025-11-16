#!/usr/bin/env python3
"""
Script to prepare DPO dataset for training.
Loads dpo_privatized_dataset.jsonl, keeps only prompt, chosen, and rejected fields,
and saves as dpo_train_ready.jsonl.
"""

import json
import os

def prepare_dpo_dataset():
    """Load DPO dataset and prepare for training."""
    
    input_file = "dpo_privatized_dataset.jsonl"
    output_file = "dpo_train_ready.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        return
    
    print(f"Loading data from {input_file}...")
    
    processed_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Check if required fields exist
                if all(field in data for field in ['prompt', 'chosen', 'rejected']):
                    # Keep only the required fields
                    filtered_data = {
                        'prompt': data['prompt'],
                        'chosen': data['chosen'],
                        'rejected': data['rejected']
                    }
                    
                    # Write to output file
                    outfile.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                else:
                    print(f"Warning: Line {line_num} missing required fields (prompt, chosen, rejected)")
                    skipped_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                skipped_count += 1
                continue
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count} examples")
    print(f"Skipped: {skipped_count} examples")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    prepare_dpo_dataset()

