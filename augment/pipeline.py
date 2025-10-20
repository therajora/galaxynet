"""
Complete Data Augmentation and Balancing Pipeline.
"""

from typing import Dict
from .augmenter import DataAugmenter
from .balancer import DataBalancer


class AugmentationPipeline:
    """Complete pipeline for data augmentation and balancing."""
    
    def __init__(self, num_augmentations_per_image: int = 5, seed: int = 42):
        """
        Initialize the pipeline.
        
        Args:
            num_augmentations_per_image: Number of augmentations per image
            seed: Seed for reproducibility
        """
        self.augmenter = DataAugmenter(num_augmentations_per_image, seed)
        self.balancer = DataBalancer(seed)
        self.num_augmentations_per_image = num_augmentations_per_image
    
    def run_complete_pipeline(self, input_dir: str, output_dir: str, 
                            source: str = "unknown",
                            balance_strategy: str = "custom",
                            balance_data: bool = True) -> Dict:
        """
        Run complete augmentation and balancing pipeline.
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            source: Image source (sdss, splus, etc.)
            balance_strategy: Balancing strategy
            balance_data: Whether to balance data
            
        Returns:
            Dictionary with complete statistics
        """
        print("=== STARTING COMPLETE AUGMENTATION PIPELINE ===")
        
        # Step 1: Initial augmentation
        print("\n--- Step 1: Initial Augmentation ---")
        augment_stats = self.augmenter.run_augmentation(
            input_dir, output_dir, source, show_progress=True
        )
        
        # Step 2: Balancing (if requested)
        balance_stats = None
        if balance_data:
            print("\n--- Step 2: Data Balancing ---")
            balance_stats = self.balancer.create_balanced_dataset(
                output_dir, output_dir, balance_strategy, self.num_augmentations_per_image, use_temp_dir=True
            )
        
        # Final statistics
        final_stats = {
            'augmentation_stats': augment_stats,
            'balance_stats': balance_stats,
            'pipeline_completed': True
        }
        
        print("\n=== PIPELINE COMPLETED SUCCESSFULLY! ===")
        return final_stats
    
    def run_augmentation_only(self, input_dir: str, output_dir: str, 
                            source: str = "unknown") -> Dict:
        """
        Run augmentation only (no balancing).
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            source: Image source
            
        Returns:
            Augmentation statistics
        """
        return self.augmenter.run_augmentation(input_dir, output_dir, source)
    
    def run_balancing_only(self, input_dir: str, output_dir: str, 
                         strategy: str = "custom") -> Dict:
        """
        Run balancing only (no additional augmentation).
        
        Args:
            input_dir: Input directory
            output_dir: Output directory
            strategy: Balancing strategy
            
        Returns:
            Balancing statistics
        """
        return self.balancer.create_balanced_dataset(input_dir, output_dir, strategy)
    
    def analyze_current_data(self, data_dir: str) -> Dict:
        """
        Analyze current data.
        
        Args:
            data_dir: Directory with data
            
        Returns:
            Analysis of current distribution
        """
        distribution = self.balancer.analyze_data_distribution(data_dir)
        
        total_images = sum(distribution.values())
        print(f"\n=== CURRENT DATA ANALYSIS ===")
        print(f"Total images: {total_images}")
        print("Distribution by class:")
        
        for class_name, count in distribution.items():
            percentage = (count / total_images) * 100 if total_images > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%)")
        
        return {
            'distribution': distribution,
            'total_images': total_images,
            'classes': list(distribution.keys()),
            'is_balanced': len(set(distribution.values())) <= 1
        }
    
    def suggest_balancing_strategy(self, data_dir: str) -> str:
        """
        Suggest balancing strategy based on current data.
        
        Args:
            data_dir: Directory with data
            
        Returns:
            Suggested strategy
        """
        distribution = self.balancer.analyze_data_distribution(data_dir)
        
        if not distribution:
            return "custom"
        
        values = list(distribution.values())
        min_val, max_val = min(values), max(values)
        
        # If difference is too large, suggest more conservative strategy
        if max_val > min_val * 3:
            return "mean"
        elif max_val > min_val * 2:
            return "custom"
        else:
            return "max"
    
    def print_pipeline_summary(self, stats: Dict):
        """
        Print summary of executed pipeline.
        
        Args:
            stats: Pipeline statistics
        """
        print("\n=== PIPELINE SUMMARY ===")
        
        if 'augmentation_stats' in stats:
            aug_stats = stats['augmentation_stats']
            print(f"Original images processed: {aug_stats['total_original_images']}")
            print(f"Augmentations created: {aug_stats['total_augmented_created']}")
        
        if 'balance_stats' in stats and stats['balance_stats']:
            balance_stats = stats['balance_stats']
            print(f"Balancing executed: Yes")
            print(f"Augmentations for balancing: {balance_stats['total_augmentations_created']}")
        else:
            print(f"Balancing executed: No")
        
        print(f"Pipeline completed: {'Yes' if stats.get('pipeline_completed', False) else 'No'}")
