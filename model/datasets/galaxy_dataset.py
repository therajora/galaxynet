"""
Dataset for galaxy classification with Hugging Face Hub support.

This module combines PyTorch dataset functionality with Hugging Face upload.
"""

import os
import cv2
import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Generator
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as PILImage

# Hugging Face imports
from datasets import Dataset as HFDataset, Features, ClassLabel, Image
from huggingface_hub import HfApi, login
import os


class GalaxyDataset(Dataset):
    """
    PyTorch dataset for galaxy classification.
    
    Combines local dataset functionality with persistent metadata.
    """

    def __init__(self, img_dir: str, metadata_path: Optional[str] = None, transform=None):
        """
        Initialize the dataset.
        
        Args:
            img_dir: Directory with images organized by class
            metadata_path: Path to save/load metadata (optional)
            transform: Transformations to be applied to images
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # If not specified, use a default file in the image directory
        if metadata_path is None:
            metadata_path = os.path.join(img_dir, "galaxy_metadata.pth")
        self.metadata_path = metadata_path

        if os.path.exists(self.metadata_path):
            self._load_metadata()
        else:
            self._create_metadata()

    def _create_metadata(self):
        """Create dataset metadata and save to file."""
        print(f"Creating metadata for {self.img_dir}...")
        
        # Find all classes (subdirectories)
        self.classes = sorted([
            d for d in os.listdir(self.img_dir) 
            if os.path.isdir(os.path.join(self.img_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all samples
        self.samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.img_dir, class_name)
            
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, file_name), 
                        class_idx
                    ))
        
        # Save metadata
        metadata = {
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'samples': self.samples,
            'img_dir': self.img_dir
        }
        
        torch.save(metadata, self.metadata_path)
        print(f"Metadata saved to: {self.metadata_path}")
        print(f"Classes found: {self.classes}")
        print(f"Total samples: {len(self.samples)}")

    def _load_metadata(self):
        """Load metadata from file."""
        print(f"Loading metadata from: {self.metadata_path}")
        metadata = torch.load(self.metadata_path)
        self.classes = metadata['classes']
        self.class_to_idx = metadata['class_to_idx']
        self.samples = metadata['samples']
        print(f"Classes loaded: {self.classes}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_class_distribution(self) -> Dict[str, int]:
        """Return class distribution."""
        distribution = {}
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            count = sum(1 for _, label in self.samples if label == class_idx)
            distribution[class_name] = count
        return distribution

    def print_summary(self):
        """Print dataset summary."""
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Directory: {self.img_dir}")
        print(f"Classes: {self.classes}")
        print(f"Total samples: {len(self.samples)}")
        print("\nDistribution by class:")
        distribution = self.get_class_distribution()
        for class_name, count in distribution.items():
            percentage = (count / len(self.samples)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
        print("="*50)


def create_huggingface_dataset(
    img_dir: str, 
    class_mapping: Optional[Dict[str, int]] = None,
    class_names: Optional[List[str]] = None
) -> HFDataset:
    """
    Create a Hugging Face dataset from an image directory.
    
    Args:
        img_dir: Directory with images organized by class
        class_mapping: Mapping of class names to labels (optional)
        class_names: Class names for the dataset (optional)
    
    Returns:
        Hugging Face dataset
    """
    print(f"Creating Hugging Face dataset for: {img_dir}")
    
    # If not specified, use default mapping
    if class_mapping is None:
        classes = sorted([
            d for d in os.listdir(img_dir) 
            if os.path.isdir(os.path.join(img_dir, d))
        ])
        class_mapping = {cls_name: i for i, cls_name in enumerate(classes)}
    
    if class_names is None:
        class_names = list(class_mapping.keys())
    
    def generate_dataset() -> Generator[Dict, None, None]:
        """Generator to create the dataset."""
        for class_name, label in class_mapping.items():
            class_path = os.path.join(img_dir, class_name)
            
            if not os.path.isdir(class_path):
                print(f"Warning: Directory {class_path} not found")
                continue
                
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_file)
                    
                    try:
                        # Load and convert image
                        image = PILImage.open(image_path).convert("RGB")
                        
                        yield {
                            "image": image,
                            "label": label
                        }
                    except Exception as e:
                        print(f"Error loading {image_path}: {e}")
                        continue

    # Define dataset features
    features = Features({
        'image': Image(),
        'label': ClassLabel(num_classes=len(class_names), names=class_names)
    })

    # Create dataset
    hf_dataset = HFDataset.from_generator(generate_dataset, features=features)
    
    print(f"Hugging Face dataset created with {len(hf_dataset)} samples")
    return hf_dataset


def upload_to_hub(
    dataset: HFDataset,
    repo_id: str,
    commit_message: str = "Upload galaxy dataset",
    token: Optional[str] = None,
    private: bool = False
) -> bool:
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        dataset: Hugging Face dataset
        repo_id: Repository ID (e.g., "username/dataset-name")
        commit_message: Commit message
        token: Authentication token (optional, uses environment variable)
        private: Whether repository should be private
    
    Returns:
        True if upload was successful
    """
    try:
        # Authentication
        if token:
            login(token=token)
        else:
            # Try to use token from environment variable
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if hf_token:
                login(token=hf_token)
            else:
                print("Error: Hugging Face token not found.")
                print("Set HUGGINGFACE_TOKEN environment variable or pass token as parameter.")
                return False
        
        print(f"Uploading to: {repo_id}")
        
        # Upload
        dataset.push_to_hub(
            repo_id=repo_id,
            commit_message=commit_message,
            private=private
        )
        
        print(f"Upload completed successfully!")
        print(f"Dataset available at: https://huggingface.co/datasets/{repo_id}")
        return True
        
    except Exception as e:
        print(f"Error during upload: {e}")
        return False


def create_binary_dataset(
    img_dir: str,
    regular_classes: List[str] = None,
    peculiar_classes: List[str] = None
) -> HFDataset:
    """
    Create a binary dataset (Regular vs Peculiar) from multiple classes.
    
    Args:
        img_dir: Directory with images organized by class
        regular_classes: List of classes considered "regular"
        peculiar_classes: List of classes considered "peculiar"
    
    Returns:
        Binary Hugging Face dataset
    """
    print("Creating binary dataset (Regular vs Peculiar)...")
    
    # Default classes if not specified
    if regular_classes is None:
        regular_classes = ['reg', 'regular']
    if peculiar_classes is None:
        peculiar_classes = ['irr_pec', 'peculiar', 'irregular']
    
    # Binary mapping
    binary_mapping = {}
    all_classes = sorted([
        d for d in os.listdir(img_dir) 
        if os.path.isdir(os.path.join(img_dir, d))
    ])
    
    for class_name in all_classes:
        if any(reg in class_name.lower() for reg in regular_classes):
            binary_mapping[class_name] = 0  # Regular
        elif any(pec in class_name.lower() for pec in peculiar_classes):
            binary_mapping[class_name] = 1  # Peculiar
        else:
            print(f"Warning: Class '{class_name}' not mapped to binary")
    
    return create_huggingface_dataset(
        img_dir=img_dir,
        class_mapping=binary_mapping,
        class_names=['regular', 'peculiar']
    )


def split_and_upload_dataset(
    img_dir: str,
    repo_id: str,
    test_size: float = 0.2,
    binary: bool = False,
    commit_message: str = "Upload galaxy dataset with train/test split",
    token: Optional[str] = None,
    private: bool = False
) -> bool:
    """
    Create dataset, split into train/test and upload to Hub.
    
    Args:
        img_dir: Directory with images
        repo_id: Repository ID
        test_size: Test set proportion
        binary: Whether to create binary dataset
        commit_message: Commit message
        token: Authentication token
        private: Whether repository should be private
    
    Returns:
        True if successful
    """
    try:
        # Create dataset
        if binary:
            dataset = create_binary_dataset(img_dir)
        else:
            dataset = create_huggingface_dataset(img_dir)
        
        # Split into train and test
        print(f"Splitting dataset into train/test (test_size={test_size})...")
        dataset_dict = dataset.train_test_split(test_size=test_size)
        
        print(f"Train: {len(dataset_dict['train'])} samples")
        print(f"Test: {len(dataset_dict['test'])} samples")
        
        # Upload
        return upload_to_hub(
            dataset=dataset_dict,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
            private=private
        )
        
    except Exception as e:
        print(f"Error during creation and upload: {e}")
        return False
