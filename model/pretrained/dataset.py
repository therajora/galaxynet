"""
Unified dataset for pre-trained galaxy classification models.

Integrates with the existing dataset system, allowing the use of local data or Hugging Face Hub.
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add root directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.datasets import GalaxyDataset, create_huggingface_dataset
from model.datasets.utils import create_transforms


class GalaxyPretrainedDataset(GalaxyDataset):
    """
    Dataset for pre-trained models that inherits from the existing GalaxyDataset.
    
    Maintains compatibility with the existing dataset system, allowing
    the use of local data or Hugging Face Hub through configurations.
    """
    
    def __init__(self, img_dir: str, metadata_path: Optional[str] = None, transform=None):
        """
        Initialize the dataset using the existing system.
        
        Args:
            img_dir: Directory containing images organized by class
            metadata_path: Path to metadata file (optional)
            transform: Transformations to be applied to images
        """
        # Use parent class constructor
        super().__init__(img_dir, metadata_path, transform)
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate weights for class balancing."""
        labels = [label for _, label in self.samples]
        class_counts = np.bincount(labels)
        class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        return class_weights


def create_data_loaders(
    dataset: GalaxyPretrainedDataset,
    batch_size: int = 32,
    train_split: float = 0.8,
    num_workers: int = 2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoaders for training and validation.
    
    Args:
        dataset: Complete dataset
        batch_size: Batch size
        train_split: Proportion for training (0.0-1.0)
        num_workers: Number of workers for loading
        random_seed: Seed for reproducibility
        
    Returns:
        Tuple with (train_loader, val_loader)
    """
    # Split the dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split into: {len(train_dataset)} training and {len(val_dataset)} validation")
    
    return train_loader, val_loader


def get_imagenet_transforms(image_size: int = 224, is_train: bool = True):
    """
    Return standard ImageNet transformations for pre-trained models.
    
    Args:
        image_size: Output image size
        is_train: Whether for training (applies data augmentation)
        
    Returns:
        torchvision transformations
    """
    # ImageNet normalization values
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    return create_transforms(
        mean=torch.tensor(imagenet_mean),
        std=torch.tensor(imagenet_std),
        img_size=(image_size, image_size),
        augment=is_train
    )
