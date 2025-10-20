"""
Data Visualization Module.

This submodule implements tools for creating visualizations
of galaxy data, including image mosaics organized by class.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .mosaic_generator import MosaicGenerator

__all__ = [
    'MosaicGenerator'
]
