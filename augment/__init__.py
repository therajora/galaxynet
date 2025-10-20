"""
Data Augmentation Module for Galaxy Classification.

This module provides functionality for:
- Image augmentation using Albumentations
- Class balancing
- Data processing pipeline
"""

from .augmenter import DataAugmenter
from .balancer import DataBalancer
from .pipeline import AugmentationPipeline

__version__ = "1.0.0"
__author__ = "Galaxy Classification Team"

__all__ = [
    'DataAugmenter',
    'DataBalancer', 
    'AugmentationPipeline'
]
