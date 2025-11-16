import numpy as np
from datasets import load_from_disk

# Configuration
LOWER_PERCENTILE = 5.0   # Lower percentile for clipping (5th percentile)
UPPER_PERCENTILE = 95.0  # Upper percentile for clipping (95th percentile)
DP_SENSITIVITY = 1.0     # DP sensitivity parameter for normalization

# Load the scored dataset
print("Loading scored dataset...")
ds = load_from_disk('./props_with_margins_hhrlhf')
margins = np.array(ds['margin_raw'])

print(f"Original margin statistics:")
print(f"  Count: {len(margins)}")
print(f"  Mean: {np.mean(margins):.4f}")
print(f"  Std: {np.std(margins):.4f}")
print(f"  Min: {np.min(margins):.4f}")
print(f"  Max: {np.max(margins):.4f}")
print(f"  5th percentile: {np.percentile(margins, 5.0):.4f}")
print(f"  95th percentile: {np.percentile(margins, 95.0):.4f}")

# Step 1: Clip margins based on percentiles
print(f"\nClipping margins based on percentiles [{LOWER_PERCENTILE}th, {UPPER_PERCENTILE}th]...")
lower_bound = np.percentile(margins, LOWER_PERCENTILE)
upper_bound = np.percentile(margins, UPPER_PERCENTILE)

print(f"  Lower bound ({LOWER_PERCENTILE}th percentile): {lower_bound:.4f}")
print(f"  Upper bound ({UPPER_PERCENTILE}th percentile): {upper_bound:.4f}")

margins_clipped = np.clip(margins, lower_bound, upper_bound)

print(f"Clipped margin statistics:")
print(f"  Mean: {np.mean(margins_clipped):.4f}")
print(f"  Std: {np.std(margins_clipped):.4f}")
print(f"  Min: {np.min(margins_clipped):.4f}")
print(f"  Max: {np.max(margins_clipped):.4f}")

# Count how many were clipped
clipped_lower = np.sum(margins < lower_bound)
clipped_upper = np.sum(margins > upper_bound)
clipped_total = clipped_lower + clipped_upper
print(f"  Examples clipped (lower): {clipped_lower} ({clipped_lower/len(margins)*100:.1f}%)")
print(f"  Examples clipped (upper): {clipped_upper} ({clipped_upper/len(margins)*100:.1f}%)")
print(f"  Total clipped: {clipped_total} ({clipped_total/len(margins)*100:.1f}%)")

# Step 2: Normalize for DP sensitivity
print(f"\nNormalizing for DP sensitivity (sensitivity={DP_SENSITIVITY})...")
# The clipping range determines the sensitivity
clipping_range = upper_bound - lower_bound
print(f"  Clipping range: {clipping_range:.4f}")

# Normalize to [-DP_SENSITIVITY, DP_SENSITIVITY] range
# This ensures the sensitivity is bounded by DP_SENSITIVITY
# Formula: normalized = (clipped - center) / (range/2) * DP_SENSITIVITY
center = (upper_bound + lower_bound) / 2
margins_normalized = ((margins_clipped - center) / (clipping_range / 2)) * DP_SENSITIVITY

print(f"Normalized margin statistics:")
print(f"  Mean: {np.mean(margins_normalized):.4f}")
print(f"  Std: {np.std(margins_normalized):.4f}")
print(f"  Min: {np.min(margins_normalized):.4f}")
print(f"  Max: {np.max(margins_normalized):.4f}")
print(f"  Expected range: [-{DP_SENSITIVITY}, {DP_SENSITIVITY}]")

# Verify normalization bounds
if np.any(margins_normalized < -DP_SENSITIVITY) or np.any(margins_normalized > DP_SENSITIVITY):
    print(f"  ⚠️  Warning: Some values outside expected range!")
    print(f"     Actual min: {np.min(margins_normalized):.4f}, max: {np.max(margins_normalized):.4f}")
else:
    print(f"  ✅ All values within expected range [-{DP_SENSITIVITY}, {DP_SENSITIVITY}]")

# Add new columns to dataset
print("\nAdding new columns to dataset...")
def add_processed_margins(example, idx):
    return {
        "margin_clipped": float(margins_clipped[idx]),
        "margin_normalized": float(margins_normalized[idx]),
        "clipping_lower_bound": float(lower_bound),
        "clipping_upper_bound": float(upper_bound),
        "clipping_range": float(clipping_range),
        "dp_sensitivity": float(DP_SENSITIVITY)
    }

ds_processed = ds.map(add_processed_margins, with_indices=True)

# Save the enhanced dataset
print("Saving enhanced dataset...")
ds_processed.save_to_disk("./props_with_processed_margins_hhrlhf")
print("Done! Saved to ./props_with_processed_margins_hhrlhf")

# Show example of the new columns
print(f"\nExample processed margins:")
example = ds_processed[0]
print(f"  Original margin: {example['margin_raw']:.4f}")
print(f"  Clipped margin: {example['margin_clipped']:.4f}")
print(f"  Normalized margin: {example['margin_normalized']:.4f}")
print(f"  Clipping bounds: [{example['clipping_lower_bound']:.4f}, {example['clipping_upper_bound']:.4f}]")
print(f"  DP sensitivity: {example['dp_sensitivity']:.4f}")

print(f"\nDataset now has columns: {ds_processed.column_names}")

# Save clipping parameters for reference
import json
clipping_info = {
    "lower_percentile": LOWER_PERCENTILE,
    "upper_percentile": UPPER_PERCENTILE,
    "lower_bound": float(lower_bound),
    "upper_bound": float(upper_bound),
    "clipping_range": float(clipping_range),
    "dp_sensitivity": float(DP_SENSITIVITY),
    "num_examples": len(margins),
    "clipped_lower_count": int(clipped_lower),
    "clipped_upper_count": int(clipped_upper),
    "clipped_total_count": int(clipped_total)
}

with open("clipping_info.json", "w") as f:
    json.dump(clipping_info, f, indent=2)
print(f"\nClipping parameters saved to clipping_info.json")

