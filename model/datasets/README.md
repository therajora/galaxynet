# Datasets Module

This module provides functionality for creating and managing galaxy classification datasets, with support for both PyTorch and Hugging Face formats.

## Files

- `__init__.py` - Module initialization
- `galaxy_dataset.py` - Main dataset class and Hugging Face integration
- `utils.py` - Utility functions for statistics and transformations
- `example_usage.py` - Usage examples
- `upload_dataset.py` - CLI script for uploading datasets to Hugging Face Hub

## Main Features

### GalaxyDataset Class
- PyTorch dataset for galaxy images
- Automatic metadata caching
- Class distribution analysis
- Support for custom transformations

### Hugging Face Integration
- Create Hugging Face datasets from local directories
- Upload datasets to Hugging Face Hub
- Binary dataset creation (Regular vs Peculiar)
- Train/test split functionality

### Utilities
- Calculate dataset statistics (mean, std)
- Create training/validation transformations
- Class weight calculation for balancing
- Image denormalization

## Quick Usage

### Create PyTorch Dataset
```python
from model.datasets import GalaxyDataset

dataset = GalaxyDataset(img_dir="data/complete_sdss")
dataset.print_summary()
```

### Create Hugging Face Dataset
```python
from model.datasets import create_huggingface_dataset

hf_dataset = create_huggingface_dataset("data/complete_sdss")
```

### Upload to Hugging Face Hub
```bash
python model/datasets/upload_dataset.py -d data/complete_sdss -r username/galaxy-dataset
```

## Dataset Structure

Expected directory structure:
```
data/
├── class1/
│   ├── image1.jpg
│   └── image2.png
└── class2/
    ├── image3.jpg
    └── image4.png
```

## Dependencies

- torch
- torchvision
- datasets (Hugging Face)
- huggingface_hub
- PIL
- numpy
- tqdm
