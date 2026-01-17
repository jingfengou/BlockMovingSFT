# data/dataset_info.py
# Registry and dataset info for BlockMoving SFT

from .blockmoving_dataset import BlockMovingCOTIterableDataset

# Dataset registry - maps dataset type names to handler classes
DATASET_REGISTRY = {
    'blockmoving_cot': BlockMovingCOTIterableDataset,
}

# Dataset information - paths and metadata
# Update these paths according to your environment
DATASET_INFO = {
    'blockmoving_cot': {
        'blockmoving_train': {
            # Path to converted parquet files
            'data_dir': 'E:/project/SPACOT/BlockMovingSFT/data/train',
            # Number of parquet files (will be determined after conversion)
            'num_files': 8,
            # Path to parquet info JSON
            'parquet_info_path': 'E:/project/SPACOT/BlockMovingSFT/data/train/train_parquet_info.json',
            # Total number of samples (will be updated after conversion)
            'num_total_samples': 800,
        },
    },
}


def get_dataset_class(dataset_type: str):
    """Get dataset handler class by type name."""
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[dataset_type]


def get_dataset_info(dataset_type: str, dataset_name: str = None):
    """Get dataset info by type and optionally specific dataset name."""
    if dataset_type not in DATASET_INFO:
        raise ValueError(f"No info for dataset type: {dataset_type}")
    
    type_info = DATASET_INFO[dataset_type]
    
    if dataset_name:
        if dataset_name not in type_info:
            raise ValueError(f"No info for dataset: {dataset_name}")
        return type_info[dataset_name]
    
    return type_info
