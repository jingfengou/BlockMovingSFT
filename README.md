# BlockMoving SFT for BAGEL

This project implements Supervised Fine-Tuning (SFT) for the BAGEL model using BlockMoving Chain-of-Thought (COT) data with interleaved image-text approach from MathCanvas.

## Project Structure

```
BlockMovingSFT/
├── convert_cot_to_parquet.py    # Data format conversion script
├── generate_dataset_handler.py  # Generates dataset handler file
├── data/
│   ├── __init__.py
│   ├── blockmoving_dataset.py   # Dataset handler for training
│   ├── dataset_info.py          # Dataset registry and paths
│   └── train/                    # Converted parquet files (after conversion)
├── configs/
│   └── blockmoving_sft.yaml     # Training configuration
├── scripts/
│   └── train_blockmoving_sft.sh # Training launch script
└── README.md
```

## Quick Start

### 1. Convert Data to Parquet Format

```bash
python convert_cot_to_parquet.py \
    --input E:/project/mydatasets/spatialViz_datasets/BlockMoving/v2/enhanced_cot_cleaned.json \
    --output ./data/train \
    --samples_per_file 100
```

This will:
- Parse COT text with interleaved `<image>` tags
- Split into question (first composite image) and solution (reasoning + step images)
- Convert to parquet format compatible with MathCanvas training pipeline
- Generate `train_parquet_info.json` with metadata

### 2. Update Configuration

Edit `data/dataset_info.py` to update paths:
```python
DATASET_INFO = {
    'blockmoving_cot': {
        'blockmoving_train': {
            'data_dir': '/path/to/BlockMovingSFT/data/train',
            'num_files': <number_of_parquet_files>,
            'parquet_info_path': '/path/to/train_parquet_info.json',
            'num_total_samples': <total_samples>,
        },
    },
}
```

### 3. Start Training

```bash
# Update paths in scripts/train_blockmoving_sft.sh first
cd BlockMovingSFT
bash scripts/train_blockmoving_sft.sh
```

## Data Format

### Source (enhanced_cot_cleaned.json)
```json
{
  "task_id": 0,
  "gt": "A",
  "pred": "A", 
  "cot_text": "<image>task_0/composite.png</image>\n\nFirst, I'll split..."
}
```

### Target (Parquet)
- `id`: Task identifier
- `answer`: Ground truth answer
- `question_interleave`: JSON array of question sequence
- `solution_interleave`: JSON array of reasoning sequence
- `question_images`: List of image bytes
- `solution_images`: List of image bytes

## Requirements

- Python 3.8+
- PyArrow
- Pillow
- PyTorch
- MathCanvas BAGEL-Canvas (for training)

## Integration with MathCanvas

This project extends the MathCanvas BAGEL-Canvas training pipeline. The dataset handler inherits from `InterleavedBaseIterableDataset` and `ParquetStandardIterableDataset` for compatibility.

## License

See LICENSE file for details.
