# Model Module

This module provides functionality for galaxy classification models, including datasets, training from scratch, and pre-trained model training.

## Structure

```
model/
├── __init__.py
├── datasets/          # Dataset creation and management
├── from_scratch/      # CNN training from scratch
└── pretrained/        # Pre-trained model training
```

## Submodules

### datasets/
- PyTorch dataset creation
- Hugging Face dataset integration
- Dataset upload to Hugging Face Hub
- Metadata management

### from_scratch/
- Custom CNN architectures
- Training from scratch
- Model evaluation and prediction
- Training utilities

### pretrained/
- Pre-trained model training (ResNet, EfficientNet, ViT)
- Configuration management
- Advanced training features
- Model upload to Hugging Face Hub

## Quick Usage

```python
from model import GalaxyDataset, GalaxyNetCNN, create_pretrained_model

# From scratch training
dataset = GalaxyDataset('data/images')
model = GalaxyNetCNN(num_classes=2)

# Pre-trained model
pretrained_model = create_pretrained_model('resnet50', num_classes=2)
```

## Dependencies

- torch
- torchvision
- transformers
- huggingface_hub
- scikit-learn
- matplotlib
- tqdm
- albumentations
- opencv-python
- pillow
