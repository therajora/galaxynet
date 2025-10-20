# GalaxyNet - Galaxy Classification System

A comprehensive system for galaxy classification using data from SDSS and S-PLUS surveys, featuring data collection, processing, model training, validation, and domain shift analysis.

## Overview

GalaxyNet is a complete machine learning pipeline for classifying galaxies as Regular or Irregular/Peculiar using images from two major astronomical surveys:

- **SDSS (Sloan Digital Sky Survey)**: 10,847 original images
- **S-PLUS (Southern Photometric Local Universe Survey)**: 787 original images

The system includes data augmentation, multiple model architectures, comprehensive validation, and domain shift analysis between the two surveys.

## Key Features

- **Complete Data Pipeline**: From raw catalog data to processed datasets
- **Multiple Model Architectures**: CNNs from scratch, pre-trained models (ResNet, EfficientNet, ViT)
- **Data Augmentation**: Advanced augmentation and class balancing
- **Comprehensive Validation**: Model evaluation, benchmarking, and visualization
- **Domain Shift Analysis**: Cross-survey performance analysis
- **Unified Interface**: Single script to control all operations
- **Hugging Face Integration**: Dataset and model sharing capabilities

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda env create -f conda_env/environment.yml
conda activate galaxynet

# Or use the unified script
./run.sh setup-environment
```

### 2. Download Data

```bash
# Download SDSS and S-PLUS images
./run.sh collect-data
```

### 3. Train Models

```bash
# Train pre-trained model
./run.sh train-pretrained -m efficientnet_b0

# Train CNN from scratch
./run.sh train-from-scratch -e 20

# Generate Kaggle configurations
./run.sh generate-kaggle-configs --model all
```

### 4. Validate Models

```bash
# Validate single model
./run.sh validate --mode single -n resnet50_v1 -p model.pth

# Run comprehensive benchmarking
./run.sh benchmark

# Execute validation tests
./run.sh validation-tests
```

## Project Structure

```ini
galaxynet/
├── run.sh                     # Unified command interface
├── collect/                   # Data collection from SDSS/S-PLUS
├── augment/                   # Data augmentation and balancing
├── model/                     # Model architectures and training
│   ├── from_scratch/         # CNN training from scratch
│   ├── pretrained/           # Pre-trained model training
│   └── datasets/             # Dataset management
├── validation/                # Model validation and benchmarking
├── domain_shift/             # Cross-survey analysis
├── data/                     # Processed datasets and metadata
├── results/                  # Training results and model outputs
├── utils/                    # Utility scripts and tools
└── conda_env/               # Environment configuration
```

## Unified Command Interface

The project provides a single script (`run.sh`) that unifies all operations:

### Training Commands

```bash
./run.sh train-pretrained -m efficientnet_b0    # Train pre-trained models
./run.sh train-from-scratch -e 20               # Train CNNs from scratch
./run.sh generate-kaggle-configs --model all    # Generate Kaggle configs
```

### Validation Commands

```bash
./run.sh validate --mode single -n resnet50_v1 -p model.pth  # Validate model
./run.sh benchmark                              # Compare multiple models
./run.sh validation-tests                       # Run validation tests
./run.sh validation-plots                       # Generate visualizations
```

### Data Commands

```bash
./run.sh augment -i data/raw -o data/augmented  # Data augmentation
./run.sh upload-dataset -r username/dataset     # Upload to Hugging Face
./run.sh clean-augmented                        # Clean augmented data
```

### Analysis Commands

```bash
./run.sh domain-shift                          # Domain shift analysis
./run.sh domain-shift-tests                    # Domain shift tests
./run.sh augment-tests                         # Augmentation tests
```

### Information Commands

```bash
./run.sh status                                # Project status
./run.sh list                                  # List available scripts
./run.sh help                                  # Complete help
```

## Dataset Information

### Original Data

- **SDSS**: 10,847 images (72.2% Regular, 27.8% Irregular/Peculiar)
- **S-PLUS**: 787 images (84.9% Regular, 15.1% Irregular/Peculiar)

### After Augmentation and Balancing

- **SDSS**: 79,539 images (59.1% Regular, 40.9% Irregular/Peculiar)
- **S-PLUS**: 6,150 images (65.2% Regular, 34.8% Irregular/Peculiar)

### Augmentation Strategy

- 5 augmentations per original image
- Custom balancing strategy
- Transformations: flips, rotations, brightness/contrast, noise, motion blur

## Model Architectures

### From Scratch CNNs

- **GalaxyNetCNN**: Simple 3-block CNN with ReLU and MaxPool
- **GalaxyNetCNNV2**: Enhanced CNN with BatchNorm and more layers

### Pre-trained Models

- **ResNet**: ResNet50, ResNet101
- **EfficientNet**: EfficientNet-B0, EfficientNet-B3
- **Vision Transformer**: ViT-B/16, ViT-B/32

## Validation and Analysis

### Model Validation

- Comprehensive metrics calculation (accuracy, precision, recall, F1, AUC)
- Confusion matrix analysis
- ROC curve generation
- Model benchmarking and ranking

### Domain Shift Analysis

- Cross-survey performance evaluation
- Statistical distance metrics (Wasserstein, Jensen-Shannon)
- Feature space analysis
- Calibration shift detection

### Survey Access

- SciServer (for SDSS)
- S-PLUS credentials (for S-PLUS)

### Hugging Face Integration

- datasets 2.0+
- huggingface_hub 0.10+

## Usage Examples

### Python API Usage

```python
# Data collection
from collect import SDSSDownloader, SPLUSDownloader
sdss = SDSSDownloader()
sdss.run()

# Data augmentation
from augment import AugmentationPipeline
pipeline = AugmentationPipeline(num_augmentations_per_image=5)
pipeline.run_complete_pipeline("data/images", "data/augmented")

# Model training
from model.from_scratch import GalaxyDataset, GalaxyTrainer
dataset = GalaxyDataset("data/complete_sdss")
trainer = GalaxyTrainer(model)
trainer.train(train_loader, val_loader, num_epochs=10)

# Model validation
from validation import ModelValidator
validator = ModelValidator(model_name="efficientnet_b0", model_path="model.pth")
results = validator.validate_dataset(test_loader)

# Domain shift analysis
from domain_shift import DomainShiftAnalyzer
analyzer = DomainShiftAnalyzer(model_name="vit_b_32", model_path="model.pth")
results = analyzer.analyze_domain_shift("data/complete_sdss", "data/complete_splus")
```

### Command Line Usage

```bash
# Complete workflow
./run.sh collect-data                    # Download data
./run.sh augment                         # Augment and balance
./run.sh train-pretrained -m efficientnet_b0  # Train model
./run.sh validate --mode single -n efficientnet_b0 -p model.pth  # Validate
./run.sh domain-shift                    # Analyze domain shift
```

## Results and Outputs

The system generates comprehensive outputs including:

- **Model Performance**: Detailed validation reports with all metrics
- **Benchmark Results**: Comparative analysis of multiple models
- **Visualizations**: Professional plots and charts for analysis
- **Domain Shift Analysis**: Cross-survey performance evaluation
- **Augmentation Reports**: Data processing statistics and summaries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This project requires credentials for SDSS (SciServer) and S-PLUS access. Please ensure you have the necessary permissions before running data collection scripts.
