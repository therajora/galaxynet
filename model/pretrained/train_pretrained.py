"""
Main script for pre-trained model training.

Allows training ResNet50, EfficientNet and ViT using JSON configuration files.
Supports local data or Hugging Face Hub.
"""

import argparse
import os
import sys
import torch
import random
import numpy as np
from pathlib import Path
from typing import Optional

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.pretrained.config_manager import ConfigManager
from model.pretrained.dataset import GalaxyPretrainedDataset, create_data_loaders, get_imagenet_transforms
from model.pretrained.model_factory import create_pretrained_model, print_model_summary, count_parameters
from model.pretrained.trainer import PretrainedTrainer
from model.datasets import create_huggingface_dataset
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import Dataset
from PIL import Image


def setup_seeds(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_device(device_config: str) -> torch.device:
    """Set up training device."""
    if device_config == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_config)
    
    print(f"Using device: {device}")
    return device


def load_huggingface_dataset(repo_id: str, cache_dir: Optional[str] = None):
    """
    Load dataset from Hugging Face Hub directly.
    
    Args:
        repo_id: Repository ID (e.g., "username/dataset-name")
        cache_dir: Directory for local cache
        
    Returns:
        Hugging Face dataset
    """
    print(f"Loading Hugging Face dataset: {repo_id}")
    
    try:
        # Download the dataset
        dataset = hf_load_dataset(repo_id, cache_dir=cache_dir)
        
        # If dataset has split, use train
        if isinstance(dataset, dict):
            if 'train' in dataset:
                dataset = dataset['train']
            else:
                # Use first available split
                dataset = list(dataset.values())[0]
        
        print(f"Dataset loaded: {len(dataset)} samples")
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise


def calculate_complete_metrics(model, val_loader, device, model_name):
    """
    Calculate complete metrics including confusion matrix.
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: Device (cuda/cpu)
        model_name: Model name
        
    Returns:
        Dictionary with complete metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    import torch.nn.functional as F
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("Calculating predictions...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            
            # Get predictions
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Positive class probability
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary')
    recall = recall_score(all_labels, all_predictions, average='binary')
    f1 = f1_score(all_labels, all_predictions, average='binary')
    roc_auc = roc_auc_score(all_labels, all_probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'confusion_matrix': cm.tolist(),
        'model_name': model_name,
        'num_samples': len(all_labels)
    }
    
    print(f"Metrics calculated for {len(all_labels)} samples")
    return metrics


class HuggingFaceDataset(Dataset):
    """
    Dataset wrapper for Hugging Face datasets.
    """
    
    def __init__(self, hf_dataset, transform=None):
        """
        Initialize the dataset.
        
        Args:
            hf_dataset: Hugging Face dataset
            transform: Transformations to be applied
        """
        self.hf_dataset = hf_dataset
        self.transform = transform
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        
        # Extract image and label
        image = item['image']
        label = item.get('label', item.get('labels', 0))
        
        # Convert to PIL if necessary
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self):
        """Return class distribution."""
        from collections import Counter
        labels = []
        for i in range(len(self.hf_dataset)):
            item = self.hf_dataset[i]
            label = item.get('label', item.get('labels', 0))
            labels.append(label)
        return dict(Counter(labels))
    
    def get_class_weights(self):
        """Calculate weights for class balancing."""
        import torch
        import numpy as np
        
        # Get class distribution
        class_dist = self.get_class_distribution()
        labels = list(class_dist.keys())
        counts = list(class_dist.values())
        
        # Calculate weights (inverse of frequency)
        weights = 1.0 / torch.tensor(counts, dtype=torch.float)
        return weights


def load_dataset(config_manager: ConfigManager) -> Dataset:
    """
    Load dataset based on configuration.
    
    Args:
        config_manager: Configuration manager
        
    Returns:
        Loaded dataset
    """
    data_config = config_manager.get_data_config()
    
    if data_config.data_source == "local":
        print(f"Loading local dataset from: {data_config.data_dir}")
        
        # Create transformations
        transforms = get_imagenet_transforms(data_config.image_size)
        
        # Create dataset
        dataset = GalaxyPretrainedDataset(
            img_dir=data_config.data_dir,
            metadata_path=data_config.metadata_path,
            transform=transforms
        )
        
    elif data_config.data_source == "huggingface":
        print(f"Loading Hugging Face dataset: {data_config.huggingface_repo}")
        
        # Load dataset from Hugging Face directly
        hf_dataset = load_huggingface_dataset(
            repo_id=data_config.huggingface_repo,
            cache_dir=None  # Use default Hugging Face cache
        )
        
        # Create transformations
        transforms = get_imagenet_transforms(data_config.image_size)
        
        # Create dataset wrapper
        dataset = HuggingFaceDataset(
            hf_dataset=hf_dataset,
            transform=transforms
        )
        
    else:
        raise ValueError(f"Data source '{data_config.data_source}' not supported")
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Print class information
    if hasattr(dataset, 'classes'):
        print(f"Classes: {dataset.classes}")
    elif hasattr(dataset, 'hf_dataset'):
        # For HuggingFaceDataset, extract classes from original dataset
        labels = set()
        for i in range(min(100, len(dataset))):  # Sample first 100 to check classes
            _, label = dataset[i]
            labels.add(label)
        print(f"Classes found: {sorted(labels)}")
    
    # Show class distribution
    class_dist = dataset.get_class_distribution()
    print(f"Class distribution:")
    for class_name, count in class_dist.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    return dataset


def create_model(config_manager: ConfigManager, num_classes: int) -> torch.nn.Module:
    """
    Create model based on configuration.
    
    Args:
        config_manager: Configuration manager
        num_classes: Number of classes
        
    Returns:
        Created model
    """
    model_config = config_manager.get_model_config()
    
    print(f"Creating model: {model_config.name}")
    
    model = create_pretrained_model(
        model_name=model_config.name,
        num_classes=num_classes,
        pretrained=model_config.pretrained,
        freeze_backbone=model_config.freeze_backbone,
        dropout_rate=model_config.dropout_rate
    )
    
    # Show model information
    print_model_summary(model, model_config.name)
    
    return model


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Train pre-trained models for galaxy classification"
    )
    
    # Required arguments
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to JSON configuration file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for training"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Seed for reproducibility"
    )
    
    parser.add_argument(
        "--validate-only", 
        action="store_true",
        help="Only validate configuration without training"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    setup_seeds(args.seed)
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config_manager = ConfigManager(args.config)
    
    # Validate configuration
    print("Validating configuration...")
    errors = config_manager.validate_config()
    if errors:
        print("Errors found in configuration:")
        for error in errors:
            print(f"  - {error}")
        return 1
    
    print("Configuration valid!")
    
    # Show configuration
    config_manager.print_config()
    
    if args.validate_only:
        print("Validation completed successfully!")
        return 0
    
    # Set up device
    device = setup_device(args.device)
    
    # Load dataset
    dataset = load_dataset(config_manager)
    
    # Create model
    # Determine number of classes
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, 'hf_dataset'):
        # For HuggingFaceDataset, count unique classes
        labels = set()
        for i in range(min(1000, len(dataset))):  # Sample to determine classes
            _, label = dataset[i]
            labels.add(label)
        num_classes = len(labels)
        print(f"Number of classes detected: {num_classes}")
    else:
        num_classes = 2  # Default for binary classification
        print(f"Using default number of classes: {num_classes}")
    
    model = create_model(config_manager, num_classes)
    
    # Create DataLoaders
    data_config = config_manager.get_data_config()
    print(f"Creating DataLoaders...")
    train_loader, val_loader = create_data_loaders(
        dataset=dataset,
        batch_size=data_config.batch_size,
        train_split=data_config.train_split,
        num_workers=data_config.num_workers,
        random_seed=args.seed
    )
    
    # Calculate class weights if necessary
    loss_config = config_manager.get_loss_config()
    class_weights = None
    if loss_config.use_class_weights:
        if hasattr(dataset, 'get_class_weights'):
            class_weights = dataset.get_class_weights()
            print(f"Using class weights: {class_weights}")
        else:
            print("Dataset does not support class weights, using uniform weights")
    
    # Create trainer
    print(f"Initializing trainer...")
    trainer = PretrainedTrainer(
        model=model,
        model_config=config_manager.get_model_config(),
        data_config=config_manager.get_data_config(),
        training_config=config_manager.get_training_config(),
        optimizer_config=config_manager.get_optimizer_config(),
        loss_config=config_manager.get_loss_config(),
        experiment_config=config_manager.get_experiment_config(),
        logging_config=config_manager.get_logging_config(),
        device=device
    )
    
    # Execute training
    print(f"Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights
    )
    
    print(f"Training completed successfully!")
    print(f"Best accuracy: {trainer.best_val_acc:.2f}% (Epoch {trainer.best_epoch})")
    
    # Calculate complete metrics on validation set
    print("Calculating complete metrics...")
    complete_metrics = calculate_complete_metrics(
        model=trainer.model,
        val_loader=val_loader,
        device=device,
        model_name=config_manager.get_model_config().name
    )
    
    # Save metrics in standardized format
    from model.pretrained.upload_model import save_training_metrics
    metrics_path = save_training_metrics(
        model_name=config_manager.get_model_config().name,
        metrics=complete_metrics,
        output_dir=config_manager.get_logging_config().save_dir.replace("models", "metrics")
    )
    
    print(f"Metrics saved to: {metrics_path}")
    print(f"Final accuracy: {complete_metrics['accuracy']:.4f}")
    print(f"F1-Score: {complete_metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {complete_metrics['roc_auc']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
