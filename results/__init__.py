"""
Results module for trained models.
"""

from .version_manager import ModelVersionManager, get_version_manager
from .training_logger import TrainingLogger, create_training_logger

__all__ = [
    'ModelVersionManager', 
    'get_version_manager',
    'TrainingLogger', 
    'create_training_logger'
]
