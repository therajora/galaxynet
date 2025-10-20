# Data Augmentation Module

Simple module for augmenting and balancing galaxy image datasets.

## Files

- `augmenter.py` - Applies image transformations using Albumentations
- `balancer.py` - Balances class distribution
- `pipeline.py` - Combines augmentation and balancing
- `example_usage.py` - Usage examples

## Quick Usage

```python
from augment import AugmentationPipeline

pipeline = AugmentationPipeline(num_augmentations_per_image=5, seed=42)

stats = pipeline.run_complete_pipeline(
    input_dir="data/images",
    output_dir="data/augmented",
    source="sdss",
    balance_strategy="custom",
    balance_data=True
)
```

## What It Does

1. Takes galaxy images organized by class folders
2. Applies transformations: flips, rotations, brightness, noise
3. Balances minority class by creating more augmented copies
4. Outputs expanded and balanced dataset

## Transformations Applied

- Horizontal/Vertical flips (50% each)
- Rotation + scale + translation (80%)
- Brightness/contrast changes (70%)
- ISO noise (30%)
- Motion blur (20%)

