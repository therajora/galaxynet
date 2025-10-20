"""
Utilities for galaxy datasets.

Helper functions for statistics calculation and transformations.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple
from tqdm import tqdm


def calculate_mean_std(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and standard deviation of a dataset.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Tuple with (mean, std)
    """
    print("Calculating dataset mean and standard deviation...")
    
    # Temporary transformation for calculation
    temp_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])
    
    # Apply temporary transformation if needed
    original_transform = getattr(dataset, 'transform', None)
    if original_transform is None:
        dataset.transform = temp_transform
    
    # Calculate statistics
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = len(dataset)
    
    for i in tqdm(range(total_samples), desc="Calculating statistics"):
        image, _ = dataset[i]
        if isinstance(image, torch.Tensor):
            # If already tensor, use directly
            img_tensor = image
        else:
            # If numpy array, convert
            img_tensor = temp_transform(image)
        
        # Calculate statistics per channel
        for j in range(3):
            mean[j] += img_tensor[j, :, :].mean()
            std[j] += img_tensor[j, :, :].std()
    
    # Normalize
    mean /= total_samples
    std /= total_samples
    
    # Restore original transformation
    if original_transform is None:
        dataset.transform = None
    
    print(f"Calculated mean: {mean}")
    print(f"Calculated standard deviation: {std}")
    
    return mean, std


def create_transforms(
    mean: torch.Tensor, 
    std: torch.Tensor, 
    img_size: Tuple[int, int] = (224, 224),
    augment: bool = True
) -> transforms.Compose:
    """
    Create transformations for training and validation.
    
    Args:
        mean: Mean of RGB channels
        std: Standard deviation of RGB channels
        img_size: Image size (height, width)
        augment: Whether to include augmentations for training
    
    Returns:
        Composed transformations
    """
    if augment:
        # Transformations for training (with augmentation)
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train_transforms
    else:
        # Transformations for validation/test (without augmentation)
        val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean=mean, std=std)
        ])
        return val_transforms


def create_visualization_transforms(img_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Create transformations for visualization (without normalization).
    
    Args:
        img_size: Image size
    
    Returns:
        Transformations for visualization
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(img_size, antialias=True)
    ])


def denormalize_image(
    tensor: torch.Tensor, 
    mean: torch.Tensor, 
    std: torch.Tensor
) -> torch.Tensor:
    """
    Denormalize a tensor image.
    
    Args:
        tensor: Normalized image
        mean: Mean used in normalization
        std: Standard deviation used in normalization
    
    Returns:
        Denormalized image
    """
    # Clone tensor to avoid modifying original
    denorm = tensor.clone()
    
    # Denormalize per channel
    for t, m, s in zip(denorm, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp values between 0 and 1
    denorm = torch.clamp(denorm, 0, 1)
    
    return denorm


def get_class_weights(dataset: Dataset) -> torch.Tensor:
    """
    Calculate weights for class balancing.
    
    Args:
        dataset: PyTorch dataset
    
    Returns:
        Tensor with weights for each class
    """
    # Count samples per class
    class_counts = {}
    for _, label in dataset:
        if isinstance(label, torch.Tensor):
            label = label.item()
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Calculate weights (inverse frequency)
    total_samples = len(dataset)
    num_classes = len(class_counts)
    
    weights = torch.zeros(num_classes)
    for class_idx, count in class_counts.items():
        weights[class_idx] = total_samples / (num_classes * count)
    
    print(f"Class weights: {weights}")
    return weights
