"""
Dataset module for galaxy classification.
Based on classification/dataset.py with adaptations for the project.
"""

import os
import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Seed for reproducibility
random.seed(42)


class GalaxyDataset(Dataset):
    """
    Custom dataset for galaxy images.
    
    Args:
        img_dir (str): Directory containing images organized by class
        metadata_path (str): Path to save/load metadata
        transform (callable, optional): Transformations to be applied
    """

    def __init__(self, img_dir, metadata_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        if os.path.exists(metadata_path):
            self._load_metadata(metadata_path)
        else:
            self._create_metadata(metadata_path)

    def _create_metadata(self, metadata_path):
        """Create dataset metadata and save to file."""
        self.classes = sorted([d for d in os.listdir(self.img_dir) 
                              if os.path.isdir(os.path.join(self.img_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.img_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, file_name), class_idx))
        
        metadata = {
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'samples': self.samples
        }
        
        torch.save(metadata, metadata_path)
        print(f"Metadata created and saved to: {metadata_path}")
        print(f"Classes found: {self.classes}")
        print(f"Total samples: {len(self.samples)}")

    def _load_metadata(self, metadata_path):
        """Load metadata from file."""
        metadata = torch.load(metadata_path)
        self.classes = metadata['classes']
        self.class_to_idx = metadata['class_to_idx']
        self.samples = metadata['samples']
        print(f"Metadata loaded from: {metadata_path}")
        print(f"Classes: {self.classes}")
        print(f"Total samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

    def get_class_distribution(self):
        """Returns the class distribution in the dataset."""
        distribution = {}
        for _, label in self.samples:
            class_name = self.classes[label]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution

    def visualize_sample(self, idx=None, figsize=(10, 5)):
        """Visualizes a dataset sample."""
        if idx is None:
            idx = random.randint(0, len(self.samples) - 1)
        
        original_image, label = self.samples[idx]
        class_name = self.classes[label]
        
        # Load original image
        image = cv2.imread(original_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations if they exist
        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image
        
        plt.figure(figsize=figsize)
        plt.suptitle(f"Sample from Class '{class_name}' (Index: {idx})", fontsize=16)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        if isinstance(transformed_image, torch.Tensor):
            # If tensor, convert to numpy for visualization
            if transformed_image.dim() == 3:
                # Denormalize if necessary
                if hasattr(self, '_is_normalized') and self._is_normalized:
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    transformed_image = std * transformed_image + mean
                img_to_show = transformed_image.permute(1, 2, 0).numpy()
                img_to_show = np.clip(img_to_show, 0, 1)
            else:
                img_to_show = transformed_image
        else:
            img_to_show = transformed_image
            
        plt.imshow(img_to_show)
        plt.title("Transformed Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


def calculate_mean_std(dataset, batch_size=64, num_workers=2):
    """
    Calculate mean and standard deviation of the dataset.
    
    Args:
        dataset: Dataset to calculate statistics
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
    
    Returns:
        tuple: (mean, std) tensors
    """
    # Temporary transformation only for ToTensor and Resize
    temp_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True)
    ])
    
    # Create temporary dataset with basic transformation
    temp_dataset = GalaxyDataset(
        img_dir=dataset.img_dir,
        metadata_path=dataset.metadata_path if hasattr(dataset, 'metadata_path') else 'temp_metadata.pth',
        transform=temp_transform
    )
    
    loader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data, _ in tqdm(loader, desc="Calculating Statistics"):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    
    return mean, std


def create_transforms(mean=None, std=None, image_size=(224, 224)):
    """
    Create transformations for training and visualization.
    
    Args:
        mean (tensor, optional): Mean for normalization
        std (tensor, optional): Standard deviation for normalization
        image_size (tuple): Image size (height, width)
    
    Returns:
        tuple: (train_transforms, viz_transforms)
    """
    # Transformations for visualization
    viz_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(image_size, antialias=True),
    ])
    
    # Transformations for training
    if mean is not None and std is not None:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        # Use ImageNet default normalization
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return train_transforms, viz_transforms
