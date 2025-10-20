# Pre-trained Models Module

This module provides functionality for training pre-trained models for galaxy classification.

## Files

- `__init__.py` - Module initialization
- `dataset.py` - Dataset classes for pre-trained models
- `model_factory.py` - Factory for creating pre-trained models
- `trainer.py` - Training class with advanced features
- `config_manager.py` - Configuration management
- `train_pretrained.py` - Main training script
- `upload_model.py` - Model upload to Hugging Face Hub

## Supported Models

- **EfficientNet**: B0, B1, B2, B3, B4
- **ResNet**: 50, 50v2, ResNeXt-50
- **Vision Transformer**: ViT-Base/16, ViT-Base/32

## Quick Usage

```python
from model.pretrained import create_pretrained_model, PretrainedTrainer

# Create model
model = create_pretrained_model('resnet50', num_classes=2)

# Train with configuration
trainer = PretrainedTrainer(model, ...)
trainer.train(train_loader, val_loader)
```

## Dependencies

- torch
- torchvision
- transformers
- huggingface_hub
- scikit-learn
- matplotlib
- tqdm
