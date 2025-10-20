#!/usr/bin/env python3
"""
Usage example for the from_scratch module for CNN training.
"""

import os
import sys
from pathlib import Path

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.from_scratch.dataset import (
    GalaxyDataset, calculate_mean_std, create_transforms
)
from model.from_scratch.model import create_model
from model.from_scratch.trainer import GalaxyTrainer, create_data_loaders
from model.from_scratch.predictor import GalaxyPredictor


def example_basic_training():
    """Basic training example."""
    print("=" * 60)
    print("BASIC TRAINING EXAMPLE")
    print("=" * 60)
    
    # Configuration
    data_dir = "data/complete_sdss"  # Adjust as needed
    metadata_file = "galaxy_metadata_example.pth"
    batch_size = 16
    num_epochs = 2
    learning_rate = 0.001
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Run the augmentation pipeline first to generate the data.")
        return
    
    # 1. Create dataset and calculate statistics
    print("Creating dataset...")
    temp_transform, _ = create_transforms()
    temp_dataset = GalaxyDataset(
        img_dir=data_dir,
        metadata_path=metadata_file,
        transform=temp_transform
    )
    
    # Calculate statistics
    print("Calculating dataset statistics...")
    dataset_mean, dataset_std = calculate_mean_std(temp_dataset)
    
    # 2. Create final transformations
    train_transforms, viz_transforms = create_transforms(
        mean=dataset_mean, 
        std=dataset_std
    )
    
    # 3. Create final dataset
    full_dataset = GalaxyDataset(
        img_dir=data_dir,
        metadata_path=metadata_file,
        transform=train_transforms
    )
    
    # Show dataset information
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")
    
    class_dist = full_dataset.get_class_distribution()
    for class_name, count in class_dist.items():
        percentage = (count / len(full_dataset)) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")
    
    # 4. Create DataLoaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader = create_data_loaders(
        dataset=full_dataset,
        batch_size=batch_size,
        train_split=0.8
    )
    
    # 5. Create model
    print("\nCreating model...")
    num_classes = len(full_dataset.classes)
    model = create_model(
        model_type='simple',
        num_classes=num_classes,
        input_size=(224, 224),
        dropout_rate=0.5
    )
    
    # Show model information
    model.print_model_summary()
    
    # 6. Create trainer
    trainer = GalaxyTrainer(model, save_dir="example_model_artifacts")
    
    # 7. Train model
    print(f"\nStarting training for {num_epochs} epochs...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_best=True
    )
    
    # 8. Test predictions
    print(f"\nTesting predictions...")
    
    # Load best model
    trainer.load_model("best_model.pth")
    
    # Create predictor
    predictor = GalaxyPredictor(
        model=model,
        class_names=full_dataset.classes,
        mean=dataset_mean,
        std=dataset_std
    )
    
    # Make predictions on random samples
    results = predictor.predict_random_sample(data_dir, num_samples=3)
    
    for result in results:
        predictor.print_prediction_result(result, show_all_probs=True)
    
    print("\nExample completed!")


def example_model_comparison():
    """Example comparing different model types."""
    print("=" * 60)
    print("MODEL COMPARISON EXAMPLE")
    print("=" * 60)
    
    # Create different model types
    model_types = ['simple', 'v2']
    
    for model_type in model_types:
        print(f"\nModel: {model_type}")
        print("-" * 30)
        
        model = create_model(
            model_type=model_type,
            num_classes=2,  # reg vs irr_pec
            input_size=(224, 224),
            dropout_rate=0.5
        )
        
        # Show model information
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            print(f"Parameters: {info['total_parameters']:,}")
            print(f"FC size: {info['fc_input_size']}")
        else:
            from model.from_scratch.model import count_parameters
            params = count_parameters(model)
            print(f"Parameters: {params['total']:,}")


def example_dataset_visualization():
    """Dataset visualization example."""
    print("=" * 60)
    print("DATASET VISUALIZATION EXAMPLE")
    print("=" * 60)
    
    data_dir = "data/complete_sdss"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    # Create dataset for visualization
    viz_transforms, _ = create_transforms()
    dataset = GalaxyDataset(
        img_dir=data_dir,
        metadata_path="temp_metadata.pth",
        transform=viz_transforms
    )
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Classes: {dataset.classes}")
    
    # Visualize some samples
    print("\nVisualizing dataset samples...")
    for i in range(min(3, len(dataset))):
        dataset.visualize_sample(idx=i)


if __name__ == "__main__":
    print("FROM_SCRATCH MODULE USAGE EXAMPLES")
    print("=" * 60)
    
    # Choose which example to run
    examples = {
        '1': ("Basic Training", example_basic_training),
        '2': ("Model Comparison", example_model_comparison),
        '3': ("Dataset Visualization", example_dataset_visualization),
    }
    
    print("Choose an example:")
    for key, (name, _) in examples.items():
        print(f"  {key}) {name}")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        try:
            func()
        except Exception as e:
            print(f"Error during execution: {e}")
    else:
        print("Invalid choice")
