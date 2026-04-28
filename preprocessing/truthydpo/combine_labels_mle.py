#!/usr/bin/env python3
"""
Combine RR labels (ℓ_RR) and M1 labels (ℓ_M1) using MLE rule to get ℓ_PROPS.

From the MLE rule:
Λ(ℓ_RR, ℓ_M1) = (-1)^(ℓ_RR) log((1 - γ_ε) / γ_ε) + (-1)^(ℓ_M1) log((1 - γ_M1) / γ_M1)
ℓ_PROPS = { 1 if Λ ≤ 0; 0 if Λ > 0 }

γ_M1 is estimated from the RR/M1 disagreement rate.
"""

import json
import argparse
import numpy as np

def estimate_gamma_m1(rr_labels, m1_labels):
    """
    Estimate γ_M1 from the RR/M1 disagreement rate.
    
    γ_M1 = P(M1 is wrong) ≈ disagreement_rate / 2
    """
    if len(rr_labels) != len(m1_labels):
        raise ValueError(f"Label lengths don't match: {len(rr_labels)} vs {len(m1_labels)}")
    
    disagreements = sum(1 for rr, m1 in zip(rr_labels, m1_labels) if rr != m1)
    disagreement_rate = disagreements / len(rr_labels)
    
    # γ_M1 = P(M1 is wrong) ≈ disagreement_rate / 2
    gamma_m1 = disagreement_rate / 2.0
    
    # Clamp to reasonable range [0.01, 0.49] to avoid log(0) or log(inf)
    gamma_m1 = max(0.01, min(0.49, gamma_m1))
    
    return gamma_m1, disagreement_rate

def compute_lambda(rr_label, m1_label, gamma_eps, gamma_m1):
    """
    Compute Λ(ℓ_RR, ℓ_M1) = (-1)^(ℓ_RR) log((1 - γ_ε) / γ_ε) + (-1)^(ℓ_M1) log((1 - γ_M1) / γ_M1)
    
    Args:
        rr_label: ℓ_RR ∈ {0, 1}
        m1_label: ℓ_M1 ∈ {0, 1}
        gamma_eps: γ_ε (privacy parameter for RR)
        gamma_m1: γ_M1 (error rate for M1)
    
    Returns:
        Lambda value
    """
    # (-1)^(ℓ_RR) = 1 if ℓ_RR == 0, -1 if ℓ_RR == 1
    sign_rr = 1 if rr_label == 0 else -1
    sign_m1 = 1 if m1_label == 0 else -1
    
    # log((1 - γ) / γ) = log(1 - γ) - log(γ)
    log_ratio_eps = np.log((1 - gamma_eps) / gamma_eps) if gamma_eps > 0 and gamma_eps < 1 else 0
    log_ratio_m1 = np.log((1 - gamma_m1) / gamma_m1) if gamma_m1 > 0 and gamma_m1 < 1 else 0
    
    lambda_val = sign_rr * log_ratio_eps + sign_m1 * log_ratio_m1
    
    return lambda_val

def compute_props_label(rr_label, m1_label, gamma_eps, gamma_m1):
    """
    Compute ℓ_PROPS using MLE rule.
    
    ℓ_PROPS = { 1 if Λ ≤ 0; 0 if Λ > 0 }
    """
    lambda_val = compute_lambda(rr_label, m1_label, gamma_eps, gamma_m1)
    props_label = 1 if lambda_val <= 0 else 0
    return props_label, lambda_val

def combine_labels_mle(rr_file, m1_file, output_file, gamma_eps=0.1):
    """
    Combine RR labels and M1 labels using MLE rule.
    
    Args:
        rr_file: Path to RR labels JSONL (from extract_rr_labels_d2.py)
        m1_file: Path to M1 labels JSONL (from generate_m1_labels_d2.py)
        output_file: Path to output file with ℓ_PROPS labels
        gamma_eps: γ_ε (privacy parameter for RR, default: 0.1)
    """
    print(f"📖 Loading RR labels from: {rr_file}")
    rr_data = []
    with open(rr_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rr_data.append(json.loads(line))
    
    print(f"📖 Loading M1 labels from: {m1_file}")
    m1_data = []
    with open(m1_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                m1_data.append(json.loads(line))
    
    if len(rr_data) != len(m1_data):
        raise ValueError(f"Data lengths don't match: RR={len(rr_data)}, M1={len(m1_data)}")
    
    print(f"✅ Loaded {len(rr_data)} examples")
    
    # Extract labels
    rr_labels = []
    for ex in rr_data:
        if 'label_rr' in ex:
            rr_label = ex['label_rr']
        elif 'flipped' in ex:
            rr_label = 0 if ex['flipped'] else 1
        else:
            rr_label = 1  # Default: assume not flipped
        rr_labels.append(rr_label)
    
    m1_labels = [ex['label_m1'] for ex in m1_data]
    
    # Estimate γ_M1 from disagreement rate
    print(f"\n📊 Estimating γ_M1 from RR/M1 disagreement...")
    gamma_m1, disagreement_rate = estimate_gamma_m1(rr_labels, m1_labels)
    print(f"   Disagreement rate: {disagreement_rate:.4f} ({disagreement_rate*100:.2f}%)")
    print(f"   Estimated γ_M1: {gamma_m1:.4f}")
    print(f"   Using γ_ε: {gamma_eps:.4f}")
    
    # Compute ℓ_PROPS for each example
    print(f"\n🔀 Computing ℓ_PROPS labels using MLE rule...")
    props_data = []
    
    for i, (rr_ex, m1_ex, rr_label, m1_label) in enumerate(zip(rr_data, m1_data, rr_labels, m1_labels)):
        props_label, lambda_val = compute_props_label(rr_label, m1_label, gamma_eps, gamma_m1)
        
        # Create DPO format with ℓ_PROPS label
        if props_label == 1:
            # Keep original: chosen > rejected
            chosen = rr_ex['chosen']
            rejected = rr_ex['rejected']
        else:
            # Flip: rejected > chosen
            chosen = rr_ex['rejected']
            rejected = rr_ex['chosen']
        
        props_data.append({
            "prompt": rr_ex['prompt'],
            "chosen": chosen,
            "rejected": rejected,
            "label_rr": rr_label,
            "label_m1": m1_label,
            "label_props": props_label,
            "lambda": float(lambda_val),
            "gamma_eps": gamma_eps,
            "gamma_m1": gamma_m1,
        })
    
    # Save PROPS labels
    print(f"\n💾 Saving PROPS labels to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for props_ex in props_data:
            f.write(json.dumps(props_ex, ensure_ascii=False) + '\n')
    
    # Print statistics
    props_ones = sum(1 for x in props_data if x['label_props'] == 1)
    props_zeros = len(props_data) - props_ones
    
    print(f"\n📊 PROPS Label Statistics:")
    print(f"   ℓ_PROPS = 1: {props_ones} ({props_ones/len(props_data)*100:.1f}%)")
    print(f"   ℓ_PROPS = 0: {props_zeros} ({props_zeros/len(props_data)*100:.1f}%)")
    print(f"\n📊 Label Agreement:")
    print(f"   RR=M1: {sum(1 for rr, m1 in zip(rr_labels, m1_labels) if rr == m1)} ({sum(1 for rr, m1 in zip(rr_labels, m1_labels) if rr == m1)/len(rr_labels)*100:.1f}%)")
    print(f"   RR≠M1: {sum(1 for rr, m1 in zip(rr_labels, m1_labels) if rr != m1)} ({sum(1 for rr, m1 in zip(rr_labels, m1_labels) if rr != m1)/len(rr_labels)*100:.1f}%)")
    
    print(f"\n✅ PROPS labels saved to: {output_file}")
    
    return props_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine RR and M1 labels using MLE rule")
    parser.add_argument("--rr_file", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/d2_rr_labels.jsonl",
                       help="Path to RR labels JSONL (from extract_rr_labels_d2.py)")
    parser.add_argument("--m1_file", type=str,
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/d2_m1_labels.jsonl",
                       help="Path to M1 labels JSONL (from generate_m1_labels_d2.py)")
    parser.add_argument("--output", type=str, 
                       default="/users/jzhao7/PDPO/preprocessing/truthydpo/d2_props_labels.jsonl",
                       help="Output file for PROPS labels")
    parser.add_argument("--gamma_eps", type=float, default=0.1,
                       help="γ_ε (privacy parameter for RR, default: 0.1)")
    
    args = parser.parse_args()
    
    combine_labels_mle(args.rr_file, args.m1_file, args.output, args.gamma_eps)

