"""
Datasets module for galaxy classification.

This module provides functionality for:
- PyTorch dataset creation
- Upload datasets to Hugging Face Hub
- Metadata management
"""

from .galaxy_dataset import GalaxyDataset, create_huggingface_dataset, upload_to_hub
from .utils import calculate_mean_std, create_transforms

__all__ = [
    'GalaxyDataset',
    'create_huggingface_dataset', 
    'upload_to_hub',
    'calculate_mean_std',
    'create_transforms'
]
