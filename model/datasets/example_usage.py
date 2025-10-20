#!/usr/bin/env python3
"""
Example usage of the datasets module.

This script demonstrates how to use the datasets module functionality
to create PyTorch datasets and upload to Hugging Face Hub.
"""

import os
import sys
from pathlib import Path

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from model.datasets.galaxy_dataset import (
    GalaxyDataset, create_huggingface_dataset, upload_to_hub,
    create_binary_dataset, split_and_upload_dataset
)
from model.datasets.utils import calculate_mean_std, create_transforms


def example_pytorch_dataset():
    """Example of PyTorch dataset creation."""
    print("="*60)
    print("EXAMPLE 1: PYTORCH DATASET")
    print("="*60)
    
    # Path to data (adjust as needed)
    data_dir = "data/complete_sdss"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Skipping example.")
        return
    
    # Create PyTorch dataset
    print(f"Creating PyTorch dataset for: {data_dir}")
    dataset = GalaxyDataset(img_dir=data_dir)
    
    # Show summary
    dataset.print_summary()
    
    # Calculate statistics
    print("\nCalculating dataset statistics...")
    mean, std = calculate_mean_std(dataset)
    
    # Create transformations
    train_transforms = create_transforms(mean, std, augment=True)
    val_transforms = create_transforms(mean, std, augment=False)
    
    print(f"PyTorch dataset created successfully!")
    print(f"   - Total samples: {len(dataset)}")
    print(f"   - Classes: {dataset.classes}")
    print(f"   - Mean: {mean}")
    print(f"   - Standard deviation: {std}")


def example_huggingface_dataset():
    """Example of Hugging Face dataset creation."""
    print("\n" + "="*60)
    print("EXAMPLE 2: HUGGING FACE DATASET")
    print("="*60)
    
    data_dir = "data/complete_sdss"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Skipping example.")
        return
    
    # Create Hugging Face dataset
    print(f"Creating Hugging Face dataset for: {data_dir}")
    hf_dataset = create_huggingface_dataset(data_dir)
    
    print(f"Hugging Face dataset created successfully!")
    print(f"   - Total samples: {len(hf_dataset)}")
    print(f"   - Features: {hf_dataset.features}")
    
    # Show example
    if len(hf_dataset) > 0:
        sample = hf_dataset[0]
        print(f"   - Sample example:")
        print(f"     * Label: {sample['label']}")
        print(f"     * Class name: {hf_dataset.features['label'].names[sample['label']]}")
        print(f"     * Image size: {sample['image'].size}")


def example_binary_dataset():
    """Example of binary dataset creation."""
    print("\n" + "="*60)
    print("EXAMPLE 3: BINARY DATASET")
    print("="*60)
    
    data_dir = "data/complete_sdss"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Skipping example.")
        return
    
    # Create binary dataset
    print(f"Creating binary dataset for: {data_dir}")
    binary_dataset = create_binary_dataset(
        img_dir=data_dir,
        regular_classes=['reg'],
        peculiar_classes=['irr_pec']
    )
    
    print(f"Binary dataset created successfully!")
    print(f"   - Total samples: {len(binary_dataset)}")
    print(f"   - Classes: {binary_dataset.features['label'].names}")
    
    # Show distribution
    regular_count = sum(1 for sample in binary_dataset if sample['label'] == 0)
    peculiar_count = sum(1 for sample in binary_dataset if sample['label'] == 1)
    
    print(f"   - Regular: {regular_count} ({regular_count/len(binary_dataset)*100:.1f}%)")
    print(f"   - Peculiar: {peculiar_count} ({peculiar_count/len(binary_dataset)*100:.1f}%)")


def example_upload_simulation():
    """Simulated upload example (without actual upload)."""
    print("\n" + "="*60)
    print("EXAMPLE 4: UPLOAD SIMULATION")
    print("="*60)
    
    data_dir = "data/complete_sdss"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found. Skipping example.")
        return
    
    print("Simulating upload to Hugging Face Hub...")
    print("   (For real upload, use the upload_dataset.py script)")
    
    # Simulate dataset creation
    hf_dataset = create_huggingface_dataset(data_dir)
    
    print(f"Dataset prepared for upload:")
    print(f"   - Repository: username/galaxy-dataset")
    print(f"   - Samples: {len(hf_dataset)}")
    print(f"   - Classes: {hf_dataset.features['label'].names}")
    
    # Simulate split
    dataset_dict = hf_dataset.train_test_split(test_size=0.2)
    print(f"   - Train: {len(dataset_dict['train'])} samples")
    print(f"   - Test: {len(dataset_dict['test'])} samples")
    
    print("\nFor real upload, run:")
    print("   python model/datasets/upload_dataset.py -d data/complete_sdss -r username/galaxy-dataset --split 0.2")


def main():
    """Run all examples."""
    print("DATASETS MODULE USAGE EXAMPLES")
    print("This script demonstrates the datasets module functionality.")
    
    # Check if data is available
    data_dirs = ["data/complete_sdss", "data/complete_splus", "data/images/sdss", "data/images/splus"]
    available_dirs = [d for d in data_dirs if os.path.exists(d)]
    
    if not available_dirs:
        print("\nNo data directories found.")
        print("Expected directories:")
        for d in data_dirs:
            print(f"   - {d}")
        print("\nRun the data collection and augmentation pipeline first.")
        return
    
    print(f"\nData directories found: {available_dirs}")
    
    # Run examples
    example_pytorch_dataset()
    example_huggingface_dataset()
    example_binary_dataset()
    example_upload_simulation()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETED!")
    print("="*60)
    print("For more information, see:")
    print("   - model/datasets/README.md")
    print("   - python model/datasets/upload_dataset.py --help")


if __name__ == "__main__":
    main()
