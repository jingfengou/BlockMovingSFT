"""
Split the parquet dataset into train and test sets (90:10 ratio).
"""

import os
import json
import pandas as pd
import numpy as np

# Paths
INPUT_PARQUET = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output/train.parquet'
OUTPUT_DIR = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output'

# Load data
print(f"Loading data from {INPUT_PARQUET}")
df = pd.read_parquet(INPUT_PARQUET)
print(f"Total samples: {len(df)}")

# Shuffle and split 90:10
np.random.seed(42)
shuffled_indices = np.random.permutation(len(df))
split_idx = int(len(df) * 0.9)
train_indices = shuffled_indices[:split_idx]
test_indices = shuffled_indices[split_idx:]

train_df = df.iloc[train_indices].reset_index(drop=True)
test_df = df.iloc[test_indices].reset_index(drop=True)
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Save train set
train_path = os.path.join(OUTPUT_DIR, 'train_split.parquet')
train_df.to_parquet(train_path, index=False)
print(f"Saved train set to {train_path}")

# Save test set
test_path = os.path.join(OUTPUT_DIR, 'test_split.parquet')
test_df.to_parquet(test_path, index=False)
print(f"Saved test set to {test_path}")

# Create info files
train_info = {
    'num_samples': len(train_df),
    'file_path': train_path,
}
with open(os.path.join(OUTPUT_DIR, 'train_split_info.json'), 'w') as f:
    json.dump(train_info, f, indent=2)

test_info = {
    'num_samples': len(test_df),
    'file_path': test_path,
}
with open(os.path.join(OUTPUT_DIR, 'test_split_info.json'), 'w') as f:
    json.dump(test_info, f, indent=2)

print("\nDone!")
