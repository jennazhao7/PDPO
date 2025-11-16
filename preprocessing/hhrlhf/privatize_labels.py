import numpy as np
import json
from datasets import load_from_disk

def sigmoid(x):
    """Sigmoid function for probability calculation"""
    return 1 / (1 + np.exp(-x))

def binary_exponential_mechanism(margins, eps_labels, sensitivity=1.0):
    """
    Binary Exponential Mechanism (equivalent to Randomized Response with prior)
    for DP label flipping.
    
    The probability of keeping the original label is:
    P(keep | margin) = exp(ε * margin / (2 * Δ)) / (exp(ε * margin / (2 * Δ)) + exp(-ε * margin / (2 * Δ)))
                     = sigmoid(ε * margin / Δ)
    
    Where:
    - ε (eps_labels): privacy parameter
    - margin: normalized margin (score difference)
    - Δ (sensitivity): sensitivity of the margin function
    
    Args:
        margins: normalized margins (already scaled for DP sensitivity)
        eps_labels: privacy parameter (epsilon) for label flipping
        sensitivity: sensitivity parameter (default: 1.0, matching normalized margin range)
    
    Returns:
        keep_mask: boolean array indicating which examples to keep as-is
        flip_rate: fraction of examples that were flipped
        p_keep: array of probabilities for keeping original labels
    """
    # Binary exponential mechanism: P(keep) = sigmoid(ε * margin / Δ)
    # Since margins are already normalized to [-sensitivity, sensitivity],
    # we use: P(keep) = sigmoid(ε * margin / sensitivity)
    p_keep = sigmoid(eps_labels * margins / sensitivity)
    
    # Sample whether to keep each label pair
    keep_mask = np.random.random(len(margins)) < p_keep
    
    flip_rate = 1 - np.mean(keep_mask)
    
    return keep_mask, flip_rate, p_keep

def create_dpo_format(example, keep_original=True):
    """Convert example to DPO training format"""
    if keep_original:
        chosen = example['pos']  # HH-RLHF uses 'pos' for chosen
        rejected = example['neg']  # HH-RLHF uses 'neg' for rejected
    else:
        # Flip chosen and rejected
        chosen = example['neg']
        rejected = example['pos']
    
    return {
        "prompt": example['prompt'],
        "chosen": chosen,
        "rejected": rejected,
        "margin_raw": example['margin_raw'],
        "margin_normalized": example['margin_normalized'],
        "flipped": not keep_original
    }

def validate_inputs(dataset, margins):
    """Validate input dataset and margins"""
    print("=== INPUT VALIDATION ===")
    
    # Check dataset structure
    required_cols = ['prompt', 'pos', 'neg', 'margin_raw', 'margin_normalized']
    missing_cols = [col for col in required_cols if col not in dataset.column_names]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print(f"✅ Dataset has all required columns")
    print(f"✅ Dataset size: {len(dataset)} examples")
    
    # Check margins
    if len(margins) != len(dataset):
        raise ValueError(f"Margins length ({len(margins)}) doesn't match dataset size ({len(dataset)})")
    
    print(f"✅ Margins length matches dataset size")
    
    # Check for NaN or infinite values
    nan_count = np.sum(np.isnan(margins))
    inf_count = np.sum(np.isinf(margins))
    
    if nan_count > 0:
        print(f"⚠️  Warning: {nan_count} NaN values in margins")
    if inf_count > 0:
        print(f"⚠️  Warning: {inf_count} infinite values in margins")
    
    if nan_count == 0 and inf_count == 0:
        print(f"✅ No NaN or infinite values in margins")
    
    # Check margin bounds (should be within [-sensitivity, sensitivity])
    if 'dp_sensitivity' in dataset.column_names:
        sensitivity = dataset[0]['dp_sensitivity']
        out_of_bounds = np.sum((margins < -sensitivity) | (margins > sensitivity))
        if out_of_bounds > 0:
            print(f"⚠️  Warning: {out_of_bounds} margins outside expected range [-{sensitivity}, {sensitivity}]")
        else:
            print(f"✅ All margins within expected range [-{sensitivity}, {sensitivity}]")
    
    print("✅ Input validation passed\n")

def print_summary(dataset, margins, p_keep, flip_rate, eps_labels, sensitivity):
    """Print summary statistics"""
    print("=== PRIVATIZATION SUMMARY ===")
    print(f"Total examples: {len(dataset)}")
    print(f"Epsilon (eps_labels): {eps_labels}")
    print(f"Sensitivity (Δ): {sensitivity}")
    print(f"Flip rate: {flip_rate:.4f} ({flip_rate*100:.2f}%)")
    print(f"Keep rate: {1-flip_rate:.4f} ({(1-flip_rate)*100:.2f}%)")
    
    # Margin statistics
    abs_margins = np.abs(margins)
    print(f"\nMargin statistics:")
    print(f"  Mean |margin|: {np.mean(abs_margins):.4f}")
    print(f"  Median |margin|: {np.median(abs_margins):.4f}")
    print(f"  Std |margin|: {np.std(abs_margins):.4f}")
    print(f"  Min margin: {np.min(margins):.4f}")
    print(f"  Max margin: {np.max(margins):.4f}")
    
    # p_keep statistics
    print(f"\np_keep statistics (Binary Exponential Mechanism):")
    print(f"  Mean p_keep: {np.mean(p_keep):.4f}")
    print(f"  Median p_keep: {np.median(p_keep):.4f}")
    print(f"  Min p_keep: {np.min(p_keep):.4f}")
    print(f"  Max p_keep: {np.max(p_keep):.4f}")
    
    # Histogram of p_keep
    print(f"\n10-bin histogram of p_keep:")
    hist, bin_edges = np.histogram(p_keep, bins=10, range=(0, 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    for i, (center, count) in enumerate(zip(bin_centers, hist)):
        percentage = count / len(p_keep) * 100
        bar = "█" * int(percentage / 2)  # Simple text bar
        print(f"  [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]: {count:4d} ({percentage:5.1f}%) {bar}")
    
    return hist, bin_centers

def main():
    # Configuration
    eps_labels = 1.0  # Privacy parameter (epsilon) - adjust as needed
    output_file = "dpo_privatized_dataset.jsonl"
    
    print("Loading processed dataset...")
    dataset = load_from_disk('./props_with_processed_margins_hhrlhf')
    
    # Extract normalized margins
    margins = np.array(dataset['margin_normalized'])
    
    # Get sensitivity from dataset if available
    if 'dp_sensitivity' in dataset.column_names:
        sensitivity = dataset[0]['dp_sensitivity']
        print(f"Using DP sensitivity from dataset: {sensitivity}")
    else:
        sensitivity = 1.0
        print(f"DP sensitivity not found in dataset, using default: {sensitivity}")
    
    # Validate inputs
    validate_inputs(dataset, margins)
    
    print("Applying Binary Exponential Mechanism (RR-with-prior) for label flipping...")
    keep_mask, flip_rate, p_keep = binary_exponential_mechanism(margins, eps_labels, sensitivity)
    
    print("Creating DPO-ready dataset...")
    dpo_examples = []
    
    for i, (example, keep) in enumerate(zip(dataset, keep_mask)):
        dpo_example = create_dpo_format(example, keep_original=keep)
        dpo_examples.append(dpo_example)
    
    # Save DPO-ready dataset
    print(f"Saving DPO dataset to {output_file}...")
    with open(output_file, 'w') as f:
        for example in dpo_examples:
            f.write(json.dumps(example) + '\n')
    
    # Print summary and get histogram data
    hist, bin_centers = print_summary(dataset, margins, p_keep, flip_rate, eps_labels, sensitivity)
    
    # Save summary statistics
    summary = {
        "mechanism": "Binary Exponential Mechanism (RR-with-prior)",
        "total_examples": len(dataset),
        "eps_labels": eps_labels,
        "sensitivity": float(sensitivity),
        "flip_rate": float(flip_rate),
        "keep_rate": float(1 - flip_rate),
        "mean_abs_margin": float(np.mean(np.abs(margins))),
        "median_abs_margin": float(np.median(np.abs(margins))),
        "std_abs_margin": float(np.std(np.abs(margins))),
        "min_margin": float(np.min(margins)),
        "max_margin": float(np.max(margins)),
        "mean_p_keep": float(np.mean(p_keep)),
        "median_p_keep": float(np.median(p_keep)),
        "min_p_keep": float(np.min(p_keep)),
        "max_p_keep": float(np.max(p_keep)),
        "p_keep_histogram": {
            "bin_centers": bin_centers.tolist(),
            "counts": hist.tolist(),
            "percentages": (hist / len(p_keep) * 100).tolist()
        }
    }
    
    with open("privatization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ DPO dataset saved to: {output_file}")
    print(f"✅ Summary saved to: privatization_summary.json")
    print(f"✅ Process completed successfully!")
    print(f"\nNote: This uses the Binary Exponential Mechanism:")
    print(f"  P(keep | margin) = sigmoid(ε * margin / Δ)")
    print(f"  where ε = {eps_labels}, Δ = {sensitivity}")

if __name__ == "__main__":
    main()

