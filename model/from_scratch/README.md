# From Scratch Module

This module provides functionality for training CNN models from scratch for galaxy classification.

## Files

- `__init__.py` - Module initialization
- `dataset.py` - Custom PyTorch dataset for galaxy images
- `model.py` - CNN architectures (GalaxyNetCNN and GalaxyNetCNNV2)
- `predictor.py` - Prediction utilities for trained models
- `trainer.py` - Training class with versioning and logging
- `train_from_scratch.py` - Main training script
- `example_usage.py` - Usage examples

## Main Features

### GalaxyDataset Class
- Custom PyTorch dataset for galaxy images
- Automatic metadata caching
- Class distribution analysis
- Sample visualization

### CNN Models
- **GalaxyNetCNN**: Simple 3-block CNN with ReLU and MaxPool
- **GalaxyNetCNNV2**: Enhanced CNN with BatchNorm and more layers
- Automatic weight initialization
- Model information and parameter counting

### GalaxyTrainer Class
- Complete training pipeline
- Early stopping and best model saving
- Training history tracking and plotting
- Integration with versioning system
- Comprehensive logging

### GalaxyPredictor Class
- Single and batch prediction
- Model evaluation and accuracy calculation
- Confidence scoring
- Random sample prediction

## Quick Usage

### Basic Training
```python
from model.from_scratch import GalaxyDataset, create_model, GalaxyTrainer

# Create dataset
dataset = GalaxyDataset("data/complete_sdss", "metadata.pth")

# Create model
model = create_model("simple", num_classes=2)

# Train
trainer = GalaxyTrainer(model)
trainer.train(train_loader, val_loader, num_epochs=10)
```

### Command Line Training
```bash
python model/from_scratch/train_from_scratch.py \
    --data_dir data/complete_sdss \
    --model_type simple \
    --num_epochs 10 \
    --batch_size 32
```

### Prediction
```python
from model.from_scratch import GalaxyPredictor

predictor = GalaxyPredictor(model, class_names, mean, std)
result = predictor.predict_single("image.jpg")
predictor.print_prediction_result(result)
```

## Model Architectures

### GalaxyNetCNN (Simple)
- 3 convolutional blocks (16, 32, 64 channels)
- 2 fully connected layers (512, num_classes)
- Dropout for regularization
- ~1.2M parameters

### GalaxyNetCNNV2 (Enhanced)
- 4 convolutional blocks with BatchNorm
- 3 fully connected layers (1024, 512, num_classes)
- Enhanced regularization
- ~8.5M parameters

## Dependencies

- torch
- torchvision
- matplotlib
- numpy
- tqdm
- cv2 (OpenCV)
- PIL
