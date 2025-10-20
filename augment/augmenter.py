"""
Image Augmentation Module for Galaxy Images.
Modularized and improved version.
"""

import os
import cv2
import random
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from typing import List, Optional


class DataAugmenter:
    """Class for augmenting galaxy images."""
    
    def __init__(self, num_augmentations_per_image: int = 5, seed: int = 42):
        """
        Initialize the augmenter.
        
        Args:
            num_augmentations_per_image: Number of augmentations per image
            seed: Seed for reproducibility
        """
        self.num_augmentations_per_image = num_augmentations_per_image
        self.seed = seed
        random.seed(seed)
        
    def get_augmentation_pipeline(self) -> A.Compose:
        """
        Define and return the Albumentations transformation pipeline.
        Techniques chosen for being safe and relevant for astronomical images.
        """
        return A.Compose([
            # Horizontal and vertical flipping. Sky orientation is random.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            # Rotation, scale and translation. Simulates small orientation and distance differences.
            # rotate_limit=30: Rotates up to 30 degrees in any direction.
            # scale_limit=0.1: Zoom up to 10% (from 90% to 110% of original size).
            # shift_limit=0.06: Shifts image up to 6% of its dimension.
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=30, p=0.8),

            # Changes brightness and contrast. Simulates different observation conditions.
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),

            # Adds ISO noise to make model more robust to image quality.
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),

            # Light motion blur, can simulate slightly worse atmospheric seeing.
            A.MotionBlur(p=0.2),
        ])
    
    def plot_samples(self, original_image: cv2.Mat, augmented_images: List[cv2.Mat], 
                    save_path: Optional[str] = None):
        """
        Plot original image and augmented images.
        
        Args:
            original_image: Original image
            augmented_images: List of augmented images
            save_path: Path to save plot (optional)
        """
        num_total_plots = 1 + len(augmented_images)
        
        fig, axes = plt.subplots(1, num_total_plots, figsize=(20, 5))
        
        # Plot original image
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(original_rgb)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Plot augmented images
        for i, aug_image in enumerate(augmented_images):
            aug_rgb = cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB)
            ax = axes[i + 1]
            ax.imshow(aug_rgb)
            ax.set_title(f'Augmented #{i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def get_all_image_paths(self, input_dir: str) -> List[str]:
        """
        Get all image paths in input directory.
        
        Args:
            input_dir: Input directory
            
        Returns:
            List of image paths
        """
        all_image_paths = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_image_paths.append(os.path.join(root, file))
        return all_image_paths
    
    def augment_single_image(self, image_path: str, output_dir: str, 
                           galaxy_type: str, source: str = "unknown") -> int:
        """
        Augment a single image.
        
        Args:
            image_path: Original image path
            output_dir: Output directory
            galaxy_type: Galaxy type (folder name)
            source: Image source (sdss, splus, etc.)
            
        Returns:
            Number of augmented images created
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}. Skipping.")
            return 0
        
        # Extract file information
        original_image_name = os.path.basename(image_path)
        base_name, extension = os.path.splitext(original_image_name)
        
        # Create output directory
        type_output_dir = os.path.join(output_dir, galaxy_type)
        os.makedirs(type_output_dir, exist_ok=True)
        
        # Save original image
        original_output_path = os.path.join(type_output_dir, f"{base_name}_{source}_original{extension}")
        cv2.imwrite(original_output_path, image)
        
        # Augmentation pipeline
        pipeline = self.get_augmentation_pipeline()
        
        # Generate and save augmented versions
        created_count = 0
        for i in range(self.num_augmentations_per_image):
            augmented_data = pipeline(image=image)
            augmented_image = augmented_data['image']
            
            new_image_name = f"{base_name}_aug_{source}_{i+1}{extension}"
            output_path = os.path.join(type_output_dir, new_image_name)
            
            cv2.imwrite(output_path, augmented_image)
            created_count += 1
        
        return created_count
    
    def run_augmentation(self, input_dir: str, output_dir: str, 
                        source: str = "unknown", show_progress: bool = True) -> dict:
        """
        Run complete augmentation process.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            source: Image source
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with process statistics
        """
        print("--- Starting Data Augmentation Process ---")
        
        # Get all image paths
        all_image_paths = self.get_all_image_paths(input_dir)
        
        print(f"Found {len(all_image_paths)} original images to augment.")
        print(f"Generating {self.num_augmentations_per_image} variations per image.")
        print(f"Results will be saved to '{output_dir}'.")
        
        # Statistics
        stats = {
            'total_original_images': len(all_image_paths),
            'total_augmented_created': 0,
            'images_by_type': {},
            'failed_images': []
        }
        
        # Main loop
        if show_progress:
            image_paths = tqdm(all_image_paths, desc="Augmenting images")
        else:
            image_paths = all_image_paths
            
        for image_path in image_paths:
            # Extract galaxy type from path
            parts = image_path.split(os.sep)
            galaxy_type = parts[-2] if len(parts) > 1 else "unknown"
            
            # Initialize counter for this type
            if galaxy_type not in stats['images_by_type']:
                stats['images_by_type'][galaxy_type] = 0
            
            # Augment image
            created = self.augment_single_image(image_path, output_dir, galaxy_type, source)
            
            if created > 0:
                stats['total_augmented_created'] += created
                stats['images_by_type'][galaxy_type] += created
            else:
                stats['failed_images'].append(image_path)
        
        print("\n--- Augmentation Process Completed Successfully! ---")
        print(f"New images saved to '{output_dir}'.")
        print(f"Total augmented images created: {stats['total_augmented_created']}")
        
        return stats
