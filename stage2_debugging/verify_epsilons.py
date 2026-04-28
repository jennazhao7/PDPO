import json

def verify_flip_rate(clean_path, noisy_path, name):
    clean = [json.loads(l) for l in open(clean_path)]
    noisy = [json.loads(l) for l in open(noisy_path)]
    
    if len(clean) != len(noisy):
        print(f"{name}: Mismatch in length! Clean: {len(clean)}, Noisy: {len(noisy)}")
        # If lengths differ, we might be comparing against a subset (like truthy has 19k full, maybe only 10k flipped)
        # Let's index by ID or Prompt to be safe.
    
    clean_dict = {r.get("id", r["prompt"]): r for r in clean}
    noisy_dict = {r.get("id", r["prompt"]): r for r in noisy}
    
    flips = 0
    total = 0
    
    for k, n_row in noisy_dict.items():
        if k in clean_dict:
            c_row = clean_dict[k]
            # Check if chosen/rejected swapped
            if n_row["chosen"] == c_row["rejected"] and n_row["rejected"] == c_row["chosen"]:
                flips += 1
            total += 1
            
    if total == 0:
        print(f"{name}: Could not match any rows between clean and noisy!")
        return

    flip_rate = flips / total
    print(f"{name}:")
    print(f"  Total matched pairs: {total}")
    print(f"  Flipped pairs: {flips}")
    print(f"  Empirical flip rate: {flip_rate:.4f} ({flip_rate*100:.2f}%)")
    
    # Expected flip rate for eps=1.0 is 1 / (1 + e^1) = 1 / 3.718 = 0.2689
    target = 1.0 / (1.0 + 2.71828)
    print(f"  Target flip rate (eps=1.0): {target:.4f}")
    if abs(flip_rate - target) < 0.02:
        print("  ✅ PASS: Matches ε=1.0 theoretically expected rate")
    else:
        print("  ❌ FAIL: Does not match expected ε=1.0 rate")
    print()

print("=== Verifying Epsilon=1.0 empirical flip rates ===\n")

# 1. TruthyDPO
verify_flip_rate(
    "preprocessing/truthydpo/truthy_dpo_subset.jsonl",
    "outputs_new_models/preprocessing/d1_rr_flipped_eps1.0_seed42.jsonl",
    "TruthyDPO"
)

# 2. HH-RLHF
verify_flip_rate(
    "data/hhrlhf_secure/train_pref.jsonl",
    "data/rr_flipped/hhrlhf_train_eps1.0.jsonl",
    "HH-RLHF"
)

# 3. PKU-SafeRLHF
verify_flip_rate(
    "data/pku_saferlhf_secure/train_pref.jsonl",
    "data/rr_flipped/pku_train_eps1.0.jsonl",
    "PKU-SafeRLHF"
)
