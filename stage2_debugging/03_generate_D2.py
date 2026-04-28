#!/usr/bin/env python3
"""
Generate a perfectly disjoint "D2" split for HH-RLHF and PKU-SafeRLHF.

Since Stage 1 used indices 0 to 10,000 from the raw HuggingFace datasets,
this script pulls indices 10,000 to 20,000 and normalizes them exactly the same way
to create a 100% disjoint 10k D2 training set for Stage 2.
"""
import os
import sys
import importlib.util
from datasets import load_dataset

def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

pku_mod = load_module_from_file("pku_mod", "experiments/core_result_tonight/01_secure_pku_saferlhf.py")
hh_mod = load_module_from_file("hh_mod", "experiments/core_result_tonight/02_secure_hhrlhf.py")

def generate_d2_split(dataset_id: str, start_idx: int, end_idx: int, out_path: str, norm_fn):
    print(f"Loading {dataset_id}...")
    ds = load_dataset(dataset_id, split="train")
    
    # Take the slice we want
    subset = ds.select(range(start_idx, min(end_idx, len(ds))))
    print(f"Extracting indices {start_idx} to {start_idx+len(subset)} (Total: {len(subset)})")
    
    normalized = []
    for row in subset:
        item = norm_fn(row)
        if item is not None:
            normalized.append(item)
            
    print(f"Extracted {len(normalized)} valid preference pairs.")
    hh_mod.write_jsonl(out_path, normalized)
    print(f"Saved D2 split to: {out_path}\n")
    return len(normalized)

def main():
    os.makedirs("data/hhrlhf_secure", exist_ok=True)
    os.makedirs("data/pku_saferlhf_secure", exist_ok=True)

    print("=== Extracting Disjoint D2 Partition (indices 10k-20k) ===\n")
    generate_d2_split(
        "Anthropic/hh-rlhf", 10000, 20000, 
        "data/hhrlhf_secure/train_pref_D2.jsonl", hh_mod.normalize_row
    )
    
    generate_d2_split(
        "PKU-Alignment/PKU-SafeRLHF", 10000, 20000, 
        "data/pku_saferlhf_secure/train_pref_D2.jsonl", pku_mod.normalize_row
    )

if __name__ == "__main__":
    main()
