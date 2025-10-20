"""
Module for training pre-trained galaxy classification models.

This module provides a unified interface for training models like ResNet50,
EfficientNet and Vision Transformer (ViT) using JSON configuration files.
"""

from .dataset import GalaxyPretrainedDataset
from .model_factory import create_pretrained_model, get_available_models
from .trainer import PretrainedTrainer
from .config_manager import ConfigManager, load_config
from .train_pretrained import main as train_main

__all__ = [
    'GalaxyPretrainedDataset',
    'create_pretrained_model', 
    'get_available_models',
    'PretrainedTrainer',
    'ConfigManager',
    'load_config',
    'train_main'
]
