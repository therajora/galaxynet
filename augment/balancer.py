"""
Data Balancing Module for Galaxy Classification.
"""

import os
import random
from typing import Dict, List
import shutil


class DataBalancer:
    """Class for balancing data between classes."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize the balancer.
        
        Args:
            seed: Seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
    
    def analyze_data_distribution(self, data_dir: str) -> Dict[str, int]:
        """
        Analyze current data distribution.
        
        Args:
            data_dir: Directory with data
            
        Returns:
            Dictionary with count per class
        """
        distribution = {}
        
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                # Count images in folder
                image_count = 0
                for file in os.listdir(class_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_count += 1
                distribution[class_name] = image_count
        
        return distribution
    
    def calculate_target_distribution(self, current_distribution: Dict[str, int], 
                                    strategy: str = "mean") -> Dict[str, int]:
        """
        Calculate target distribution based on chosen strategy.
        
        Args:
            current_distribution: Current distribution
            strategy: Balancing strategy ("mean", "max", "min", "custom")
            
        Returns:
            Dictionary with target distribution
        """
        if not current_distribution:
            return {}
        
        values = list(current_distribution.values())
        
        if strategy == "mean":
            target = int(sum(values) / len(values))
        elif strategy == "max":
            target = max(values)
        elif strategy == "min":
            target = min(values)
        elif strategy == "custom":
            # Custom strategy: use intermediate value
            target = int((min(values) + max(values)) / 2)
        else:
            raise ValueError(f"Strategy '{strategy}' not recognized")
        
        return {class_name: target for class_name in current_distribution.keys()}
    
    def get_images_needed_per_class(self, current_distribution: Dict[str, int], 
                                  target_distribution: Dict[str, int]) -> Dict[str, int]:
        """
        Calculate how many images are needed per class.
        
        Args:
            current_distribution: Current distribution
            target_distribution: Target distribution
            
        Returns:
            Dictionary with number of images needed per class
        """
        needed = {}
        for class_name in current_distribution:
            current = current_distribution[class_name]
            target = target_distribution[class_name]
            needed[class_name] = max(0, target - current)
        
        return needed
    
    def select_images_for_augmentation(self, class_dir: str, num_needed: int) -> List[str]:
        """
        Select images for augmentation based on criteria.
        
        Args:
            class_dir: Class directory
            num_needed: Number of images needed
            
        Returns:
            List of selected image paths
        """
        # Get all original images (non-augmented)
        original_images = []
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Prioritize original images (without "_aug_" in name)
                if "_aug_" not in file and "_original" not in file:
                    original_images.append(os.path.join(class_dir, file))
        
        if not original_images:
            # If no original images, use all
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    original_images.append(os.path.join(class_dir, file))
        
        # Select images randomly
        if len(original_images) <= num_needed:
            return original_images
        else:
            return random.sample(original_images, num_needed)
    
    def create_balanced_dataset(self, input_dir: str, output_dir: str, 
                              strategy: str = "custom", 
                              num_augmentations_per_image: int = 5,
                              use_temp_dir: bool = True) -> Dict:
        """
        Create a balanced dataset.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            strategy: Balancing strategy
            num_augmentations_per_image: Number of augmentations per image
            
        Returns:
            Dictionary with balancing statistics
        """
        print("--- Starting Data Balancing ---")
        
        # Analyze current distribution
        current_dist = self.analyze_data_distribution(input_dir)
        print(f"Current distribution: {current_dist}")
        
        # Calculate target distribution
        target_dist = self.calculate_target_distribution(current_dist, strategy)
        print(f"Target distribution: {target_dist}")
        
        # Calculate needed images
        needed = self.get_images_needed_per_class(current_dist, target_dist)
        print(f"Images needed per class: {needed}")
        
        # Use temporary directory if input_dir == output_dir
        if use_temp_dir and input_dir == output_dir:
            temp_output_dir = output_dir + "_temp"
            print(f"Using temporary directory: {temp_output_dir}")
        else:
            temp_output_dir = output_dir
        
        # Create output directory
        os.makedirs(temp_output_dir, exist_ok=True)
        
        # Statistics
        stats = {
            'original_distribution': current_dist,
            'target_distribution': target_dist,
            'images_needed': needed,
            'total_augmentations_created': 0,
            'augmentations_by_class': {}
        }
        
        # Process each class
        for class_name in current_dist:
            class_input_dir = os.path.join(input_dir, class_name)
            class_output_dir = os.path.join(temp_output_dir, class_name)
            
            if not os.path.exists(class_input_dir):
                continue
            
            # Copy all existing images
            os.makedirs(class_output_dir, exist_ok=True)
            for file in os.listdir(class_input_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    src = os.path.join(class_input_dir, file)
                    dst = os.path.join(class_output_dir, file)
                    # Only copy if not the same file
                    if src != dst:
                        shutil.copy2(src, dst)
            
            # If more images are needed, select for augmentation
            if needed[class_name] > 0:
                selected_images = self.select_images_for_augmentation(
                    class_input_dir, needed[class_name]
                )
                
                print(f"Class '{class_name}': selected {len(selected_images)} images for augmentation")
                
                # Calculate how many augmentations are needed per image
                images_needed = needed[class_name]
                if len(selected_images) > 0:
                    augs_per_image = max(1, images_needed // len(selected_images))
                else:
                    augs_per_image = num_augmentations_per_image
                
                augmentations_created = 0
                for img_path in selected_images:
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    extension = os.path.splitext(img_path)[1]
                    
                    # Create multiple copies (simulating augmentation)
                    for i in range(augs_per_image):
                        if augmentations_created >= images_needed:
                            break
                            
                        new_name = f"{base_name}_balanced_{i+1}{extension}"
                        dst = os.path.join(class_output_dir, new_name)
                        
                        # Check if file exists to avoid errors
                        if not os.path.exists(dst):
                            shutil.copy2(img_path, dst)
                            augmentations_created += 1
                
                stats['augmentations_by_class'][class_name] = augmentations_created
                stats['total_augmentations_created'] += augmentations_created
        
        # Move files from temporary directory if needed
        if use_temp_dir and input_dir == output_dir and temp_output_dir != output_dir:
            print("Moving files from temporary directory...")
            # Remove destination directory if exists
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            # Move temporary directory to destination
            shutil.move(temp_output_dir, output_dir)
        
        # Check final distribution
        final_dist = self.analyze_data_distribution(output_dir)
        stats['final_distribution'] = final_dist
        
        print(f"Final distribution: {final_dist}")
        print("--- Balancing Complete! ---")
        
        return stats
    
    def print_balance_report(self, stats: Dict):
        """
        Print balancing report.
        
        Args:
            stats: Balancing statistics
        """
        print("\n=== BALANCING REPORT ===")
        print(f"Original distribution: {stats['original_distribution']}")
        print(f"Target distribution: {stats['target_distribution']}")
        print(f"Final distribution: {stats['final_distribution']}")
        print(f"Total augmentations created: {stats['total_augmentations_created']}")
        print("\nAugmentations per class:")
        for class_name, count in stats['augmentations_by_class'].items():
            print(f"  {class_name}: {count} images")
