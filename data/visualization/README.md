# Visualization Module

Simple module for creating galaxy image mosaics.

## Files

- `mosaic_generator.py` - Class to create organized image mosaics
- `create_mosaics.py` - CLI script to generate mosaics
- `__init__.py` - Module initialization

## Quick Usage

```python
from data.visualization import MosaicGenerator

generator = MosaicGenerator(output_dir="data/mosaics")

# Create mosaic for a dataset
generator.create_mosaic_for_dataset("sdss")

# Create comparison mosaic
generator.create_comparison_mosaic(sdss_dir, splus_dir)
```

## CLI Usage

```bash
# Create all mosaics
python create_mosaics.py --mode all

# Create single dataset mosaic
python create_mosaics.py --mode single --dataset sdss

# Create comparison mosaic
python create_mosaics.py --mode comparison
```

## What It Does

Creates visual mosaics with:
- 4 rows x 5 columns layout
- Images organized by class (Regular, Peculiar)
- Galaxy names displayed on images
- Comparison mosaics between SDSS and S-PLUS
- Data augmentation variation mosaics

