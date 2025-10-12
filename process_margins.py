import numpy as np
from datasets import load_from_disk

# Load the scored dataset
print("Loading scored dataset...")
ds = load_from_disk('./props_with_margins_truthy')
margins = np.array(ds['train']['margin_raw'])

print(f"Original margin statistics:")
print(f"  Count: {len(margins)}")
print(f"  Mean: {np.mean(margins):.4f}")
print(f"  Std: {np.std(margins):.4f}")
print(f"  Min: {np.min(margins):.4f}")
print(f"  Max: {np.max(margins):.4f}")

# Step 1: Clip margins to [-6, 6]
print("\nClipping margins to [-6, 6]...")
margins_clipped = np.clip(margins, -6.0, 6.0)

print(f"Clipped margin statistics:")
print(f"  Mean: {np.mean(margins_clipped):.4f}")
print(f"  Std: {np.std(margins_clipped):.4f}")
print(f"  Min: {np.min(margins_clipped):.4f}")
print(f"  Max: {np.max(margins_clipped):.4f}")

# Count how many were clipped
clipped_count = np.sum((margins != margins_clipped))
print(f"  Examples clipped: {clipped_count} ({clipped_count/len(margins)*100:.1f}%)")

# Step 2: Normalize the clipped margins
print("\nNormalizing clipped margins...")
margins_normalized = (margins_clipped - np.mean(margins_clipped)) / np.std(margins_clipped)

print(f"Normalized margin statistics:")
print(f"  Mean: {np.mean(margins_normalized):.4f}")
print(f"  Std: {np.std(margins_normalized):.4f}")
print(f"  Min: {np.min(margins_normalized):.4f}")
print(f"  Max: {np.max(margins_normalized):.4f}")

# Add new columns to dataset
print("\nAdding new columns to dataset...")
def add_processed_margins(example, idx):
    return {
        "margin_clipped": float(margins_clipped[idx]),
        "margin_normalized": float(margins_normalized[idx])
    }

ds_processed = ds['train'].map(add_processed_margins, with_indices=True)

# Save the enhanced dataset
print("Saving enhanced dataset...")
ds_processed.save_to_disk("./props_with_processed_margins")
print("Done! Saved to ./props_with_processed_margins")

# Show example of the new columns
print(f"\nExample processed margins:")
example = ds_processed[0]
print(f"  Original margin: {example['margin_raw']:.4f}")
print(f"  Clipped margin: {example['margin_clipped']:.4f}")
print(f"  Normalized margin: {example['margin_normalized']:.4f}")

print(f"\nDataset now has columns: {ds_processed.column_names}")


