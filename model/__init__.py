"""
Models module for galaxy classification.
"""

from .from_scratch.dataset import GalaxyDataset
from .from_scratch.model import GalaxyNetCNN
from .from_scratch.predictor import GalaxyPredictor

__all__ = ['GalaxyDataset', 'GalaxyNetCNN', 'GalaxyPredictor']
