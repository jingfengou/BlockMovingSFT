#!/usr/bin/env python3
"""
Download complete BAGEL-7B-MoT model from HuggingFace.

Usage:
    conda activate bagel-canvas
    python download_bagel_complete.py
"""

import os
from huggingface_hub import snapshot_download

# Configuration
MODEL_ID = "ByteDance-Seed/BAGEL-7B-MoT"
LOCAL_DIR = "/home/ubuntu/oujingfeng/project/spatialreasoning/models/BAGEL-7B-MoT-complete"

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
print("This may take a while (~30GB)...")

# Download all files
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"\nDownload complete!")
print(f"Model saved to: {LOCAL_DIR}")

# List downloaded files
print("\nDownloaded files:")
for f in os.listdir(LOCAL_DIR):
    size = os.path.getsize(os.path.join(LOCAL_DIR, f))
    if size > 1024 * 1024:  # > 1MB
        print(f"  {f}: {size / (1024**3):.2f} GB")
    else:
        print(f"  {f}: {size / 1024:.1f} KB")
