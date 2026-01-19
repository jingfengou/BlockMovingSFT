"""
Generate proper parquet_info JSON file with num_row_groups.
"""

import os
import json
import pyarrow.parquet as pq

# Paths
TRAIN_PARQUET = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output/train_split.parquet'
OUTPUT_DIR = '/home/ubuntu/oujingfeng/project/spatialreasoning/BlockMovingSFT/data/output'

# Read parquet file metadata
pf = pq.ParquetFile(TRAIN_PARQUET)
num_row_groups = pf.metadata.num_row_groups

print(f"File: {TRAIN_PARQUET}")
print(f"Num row groups: {num_row_groups}")
print(f"Num rows: {pf.metadata.num_rows}")

# Create proper parquet_info format
parquet_info = {
    TRAIN_PARQUET: {
        'num_row_groups': num_row_groups,
        'num_rows': pf.metadata.num_rows
    }
}

# Save
info_path = os.path.join(OUTPUT_DIR, 'train_split_info.json')
with open(info_path, 'w') as f:
    json.dump(parquet_info, f, indent=2)
print(f"Saved to {info_path}")
