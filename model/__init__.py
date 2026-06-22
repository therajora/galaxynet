"""
Models module for galaxy classification.
"""

__all__ = ["GalaxyDataset", "GalaxyNetCNN", "GalaxyPredictor"]


def __getattr__(name):
    if name == "GalaxyDataset":
        from .from_scratch.dataset import GalaxyDataset

        return GalaxyDataset
    if name == "GalaxyNetCNN":
        from .from_scratch.model import GalaxyNetCNN

        return GalaxyNetCNN
    if name == "GalaxyPredictor":
        from .from_scratch.predictor import GalaxyPredictor

        return GalaxyPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
