"""
From_scratch module for training CNN models from scratch.
"""

from .dataset import GalaxyDataset
from .model import GalaxyNetCNN
from .predictor import GalaxyPredictor
from .trainer import GalaxyTrainer

__all__ = ['GalaxyDataset', 'GalaxyNetCNN', 'GalaxyPredictor', 'GalaxyTrainer']
