import argparse
import numpy as np
import torch
import json
from datasets import load_from_disk
from collections import Counter
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid function for probability calculation"""
    return 1 / (1 + np.exp(-x))

def squash_confidence(
    abs_margins,
    mode="clipped_linear",
    alpha=5.0,
    threshold=0.0,
    clip_quantiles=(0.05, 0.90),
    cap_quantile=0.8,
    softplus_scale=1.0,
):
    """
    Map |margin| to g in [0,1] without early saturation.
    
    Modes:
      - "clipped_linear": g = clip((c - lo) / (hi - lo), 0, 1) with lo/hi as quantiles.
      - "quantile_cap": g = min(conf / T, 1), with T at cap_quantile (e.g., 0.8/0.9).
      - "linear": percentile-based min-max scaling (temperature-free). Uses clip_quantiles
                  to damp outliers before scaling; conf at lower quantile -> 0, upper -> 1.
      - "sigmoid": normalized sigmoid with g(conf=0)=0 (kept for backward compatibility).
      - "softplus": smooth, sublinear growth; g = softplus(conf / scale) normalized to [0,1].
    """
    if mode == "clipped_linear":
        q_low, q_high = clip_quantiles
        lo, hi = np.quantile(abs_margins, [q_low, q_high])
        if hi <= lo:
            hi = lo + 1e-12
        g = (abs_margins - lo) / (hi - lo)
        return np.clip(g, 0.0, 1.0)
    elif mode == "quantile_cap":
        T = np.quantile(abs_margins, cap_quantile)
        if T <= 0:
            T = 1e-12
        g = abs_margins / (T + 1e-12)
        return np.clip(g, 0.0, 1.0)
    elif mode == "linear":
        q_low, q_high = clip_quantiles
        lo, hi = np.quantile(abs_margins, [q_low, q_high])
        if hi <= lo:
            hi = lo + 1e-12  # avoid divide-by-zero
        g = (abs_margins - lo) / (hi - lo)
        return np.clip(g, 0.0, 1.0)
    elif mode == "sigmoid":
        g_raw = sigmoid(alpha * (abs_margins - threshold))
        g0 = sigmoid(-alpha * threshold)
        return np.clip((g_raw - g0) / (1 - g0), 0.0, 1.0)
    elif mode == "softplus":
        # Softplus scaled then normalized by its value at 1 to map typical magnitudes into (0,1)
        x = abs_margins / max(softplus_scale, 1e-12)
        g_raw = np.log1p(np.exp(x))
        # normalize so g_raw at x=1 maps near ~0.78; then clip to [0,1]
        norm = np.log1p(np.exp(1.0))
        g = g_raw / (norm + 1e-12)
        return np.clip(g, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown squash mode: {mode}")

def sample_privatized_labels(
    margins,
    eps_labels=1.0,
    alpha=5.0,
    threshold=0.0,
    eps_floor=1e-6,
    squash_mode="clipped_linear",
    clip_quantiles=(0.05, 0.90),
    cap_quantile=0.8,
    softplus_scale=1.0,
):
    """
    Sample privatized labels using a DP-safe keep probability:
      eps = eps_labels
      p_rr = exp(eps) / (1 + exp(eps))                   # random-response bound
      conf = |margin|                                   # confidence from margins
      g = squash_confidence(conf, mode=squash_mode, alpha=alpha,
                            threshold=threshold, clip_quantiles=clip_quantiles,
                            cap_quantile=cap_quantile, softplus_scale=softplus_scale)
      p_keep = 0.5 + (p_rr - 0.5) * g                   # interpolate in DP-safe range
      p_keep = clip(p_keep, 0.5 + eps_floor, p_rr - eps_floor)
    
    Args:
        margins: normalized margins (f_delta values)
        eps_labels: privacy parameter for label privatization
        alpha: slope for sigmoid mode (unused in linear/quantile_cap)
        threshold: confidence offset before squashing (sigmoid mode)
        eps_floor: small margin to avoid hitting numerical boundaries
        squash_mode: "clipped_linear" (default), "quantile_cap", "linear", "sigmoid", or "softplus"
        clip_quantiles: low/high quantiles used to scale in clipped_linear/linear modes
        cap_quantile: quantile used as T in quantile_cap mode
        softplus_scale: scale for softplus mode
    
    Returns:
        keep_mask: boolean array indicating which examples to keep as-is
        flip_rate: fraction of examples that were flipped
        p_keep: keep probabilities after clipping/sanity checks
    """
    # Privacy boundary from random response; must be > 0.5 for valid DP budget
    p_rr = np.exp(eps_labels) / (1 + np.exp(eps_labels))
    if p_rr <= 0.5:
        raise ValueError(f"p_rr must exceed 0.5; got {p_rr}")
    
    # Confidence = |margin| (direction handled elsewhere)
    abs_margins = np.abs(margins)
    
    # Magnitude-based squashing (temperature-free default: linear)
    g = squash_confidence(
        abs_margins,
        mode=squash_mode,
        alpha=alpha,
        threshold=threshold,
        clip_quantiles=clip_quantiles,
        cap_quantile=cap_quantile,
        softplus_scale=softplus_scale,
    )
    
    # Interpolate within DP-safe interval
    p_keep = 0.5 + (p_rr - 0.5) * g
    
    # Numerical safety: ensure we stay away from the exact bounds
    lower_bound = 0.5 + eps_floor
    upper_bound = p_rr - eps_floor
    if lower_bound >= upper_bound:
        raise ValueError(
            f"Invalid clipping bounds: lower {lower_bound} >= upper {upper_bound}. "
            "Decrease eps_floor or increase eps_labels."
        )
    p_keep = np.clip(p_keep, lower_bound, upper_bound)
    
    # Sanity checks to ensure DP-safe range
    if np.any(p_keep <= 0.5):
        raise ValueError("p_keep must be strictly greater than 0.5 after clipping")
    if np.any(p_keep >= p_rr):
        raise ValueError("p_keep must be strictly less than p_rr after clipping")
    
    # Sample whether to keep each label pair
    keep_mask = np.random.random(len(margins)) < p_keep
    
    flip_rate = 1 - np.mean(keep_mask)
    
    return keep_mask, flip_rate, p_keep

def create_dpo_format(example, keep_original=True):
    """Convert example to DPO training format"""
    if keep_original:
        chosen = example['chosen']
        rejected = example['rejected']
    else:
        # Flip chosen and rejected
        chosen = example['rejected']
        rejected = example['chosen']
    
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
    required_cols = ['prompt', 'chosen', 'rejected', 'margin_raw', 'margin_normalized']
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
    
    print("✅ Input validation passed\n")

def print_summary(dataset, margins, p_keep, flip_rate, eps_labels):
    """Print summary statistics"""
    print("=== PRIVATIZATION SUMMARY ===")
    print(f"Total examples: {len(dataset)}")
    p_rr = np.exp(eps_labels) / (1 + np.exp(eps_labels))
    print(f"Epsilon (eps_labels): {eps_labels}")
    print(f"Random-response upper bound p_rr: {p_rr:.6f}")
    print(f"Flip rate: {flip_rate:.4f} ({flip_rate*100:.2f}%)")
    print(f"Keep rate: {1-flip_rate:.4f} ({(1-flip_rate)*100:.2f}%)")
    
    # Margin statistics
    abs_margins = np.abs(margins)
    print(f"\nMargin statistics:")
    print(f"  Mean |f_delta|: {np.mean(abs_margins):.4f}")
    print(f"  Median |f_delta|: {np.median(abs_margins):.4f}")
    print(f"  Std |f_delta|: {np.std(abs_margins):.4f}")
    
    # p_keep statistics
    print(f"\np_keep statistics:")
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

def print_decile_table(margins, p_keep, n_bins=10):
    """Print mean |margin| and mean p_keep per decile sorted by |margin|."""
    abs_margins = np.abs(margins)
    order = np.argsort(abs_margins)
    abs_sorted = abs_margins[order]
    p_sorted = p_keep[order]
    bins = np.array_split(np.arange(len(abs_sorted)), n_bins)
    
    print("\nDecile table (sorted by |margin|):")
    print("decile\tcount\tmean|margin|\tmean p_keep")
    for i, idx in enumerate(bins, start=1):
        if len(idx) == 0:
            mean_margin = float("nan")
            mean_p = float("nan")
        else:
            mean_margin = float(np.mean(abs_sorted[idx]))
            mean_p = float(np.mean(p_sorted[idx]))
        print(f"{i:2d}\t{len(idx):5d}\t{mean_margin:.6f}\t{mean_p:.6f}")

def parse_args():
    parser = argparse.ArgumentParser(description="Privatize labels with DP-safe keep probabilities.")
    parser.add_argument("--eps-labels", type=float, default=1.0, help="Privacy parameter epsilon for labels.")
    parser.add_argument("--squash-mode", type=str, default="clipped_linear",
                        choices=["clipped_linear", "quantile_cap", "linear", "sigmoid", "softplus"],
                        help="Squashing mode for confidence → g.")
    parser.add_argument("--cap-quantile", type=float, default=0.8, help="Quantile T for quantile_cap mode.")
    parser.add_argument("--softplus-scale", type=float, default=1.0, help="Scale for softplus squashing.")
    parser.add_argument("--alpha", type=float, default=5.0, help="Alpha for sigmoid squashing.")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for sigmoid squashing.")
    parser.add_argument("--eps-floor", type=float, default=1e-6, help="Margin away from DP bounds.")
    parser.add_argument("--clip-quantiles", type=float, nargs=2, default=(0.05, 0.90),
                        help="Low/high quantiles for linear squashing.")
    parser.add_argument("--dataset-path", type=str, default="./props_with_processed_margins",
                        help="Path to processed dataset on disk.")
    parser.add_argument("--output-file", type=str, default="dpo_privatized_dataset.jsonl",
                        help="Output JSONL filename.")
    parser.add_argument("--no-summary", action="store_true", help="Skip printing summary tables.")
    return parser.parse_args()

def main():
    args = parse_args()
    eps_labels = args.eps_labels
    output_file = args.output_file
    
    print("Loading processed dataset...")
    dataset = load_from_disk(args.dataset_path)
    
    # Extract normalized margins
    margins = np.array(dataset['margin_normalized'])
    
    # Validate inputs
    validate_inputs(dataset, margins)
    
    print("Sampling privatized labels...")
    keep_mask, flip_rate, p_keep = sample_privatized_labels(
        margins,
        eps_labels=eps_labels,
        alpha=args.alpha,
        threshold=args.threshold,
        eps_floor=args.eps_floor,
        squash_mode=args.squash_mode,
        clip_quantiles=tuple(args.clip_quantiles),
        cap_quantile=args.cap_quantile,
        softplus_scale=args.softplus_scale,
    )
    
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
    if args.no_summary:
        hist = np.array([])
        bin_centers = np.array([])
    else:
        hist, bin_centers = print_summary(dataset, margins, p_keep, flip_rate, eps_labels)
        print_decile_table(margins, p_keep, n_bins=10)
    
    # Save summary statistics
    summary = {
        "total_examples": len(dataset),
        "eps_labels": eps_labels,
        "flip_rate": float(flip_rate),
        "keep_rate": float(1 - flip_rate),
        "mean_abs_margin": float(np.mean(np.abs(margins))),
        "median_abs_margin": float(np.median(np.abs(margins))),
        "std_abs_margin": float(np.std(np.abs(margins))),
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

if __name__ == "__main__":
    main()
