"""
Example usage of the augmentation and balancing module.
"""

from augment import AugmentationPipeline, DataAugmenter, DataBalancer


def example_basic_augmentation():
    """Basic augmentation example."""
    print("=== Example: Basic Augmentation ===")
    
    augmenter = DataAugmenter(num_augmentations_per_image=3, seed=42)
    
    # Configuration
    input_dir = "../data/images/sdss"  # Adjust as needed
    output_dir = "augmented_sdss"
    source = "sdss"
    
    # Run augmentation
    stats = augmenter.run_augmentation(input_dir, output_dir, source)
    
    print(f"Augmentations created: {stats['total_augmented_created']}")
    return stats


def example_balancing_only():
    """Balancing only example."""
    print("\n=== Example: Balancing Only ===")
    
    balancer = DataBalancer(seed=42)
    
    # Configuration
    input_dir = "../data/images/sdss"
    output_dir = "balanced_sdss"
    strategy = "custom"
    
    # Run balancing
    stats = balancer.create_balanced_dataset(input_dir, output_dir, strategy)
    
    balancer.print_balance_report(stats)
    return stats


def example_complete_pipeline():
    """Complete pipeline example."""
    print("\n=== Example: Complete Pipeline ===")
    
    pipeline = AugmentationPipeline(num_augmentations_per_image=5, seed=42)
    
    # Configuration
    input_dir = "../data/images/sdss"
    output_dir = "complete_pipeline_output"
    source = "sdss"
    balance_strategy = "custom"
    
    # Analyze current data
    print("Analyzing current data...")
    analysis = pipeline.analyze_current_data(input_dir)
    
    # Suggest strategy
    suggested_strategy = pipeline.suggest_balancing_strategy(input_dir)
    print(f"Suggested strategy: {suggested_strategy}")
    
    # Run complete pipeline
    stats = pipeline.run_complete_pipeline(
        input_dir, output_dir, source, balance_strategy, balance_data=True
    )
    
    # Print summary
    pipeline.print_pipeline_summary(stats)
    
    return stats


def example_analyze_data():
    """Data analysis example."""
    print("\n=== Example: Data Analysis ===")
    
    pipeline = AugmentationPipeline()
    
    # Analyze SDSS data
    print("Analyzing SDSS data...")
    sdss_analysis = pipeline.analyze_current_data("../data/images/sdss")
    
    # Analyze S-PLUS data
    print("\nAnalyzing S-PLUS data...")
    splus_analysis = pipeline.analyze_current_data("../data/images/splus")
    
    return sdss_analysis, splus_analysis


def main():
    """Main function with examples."""
    print("Augmentation and Balancing Module Usage Examples")
    print("=" * 60)
    
    try:
        # Example 1: Data analysis
        example_analyze_data()
        
        # Example 2: Basic augmentation
        # example_basic_augmentation()
        
        # Example 3: Balancing only
        # example_balancing_only()
        
        # Example 4: Complete pipeline
        # example_complete_pipeline()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Check that input directories exist and contain images.")


if __name__ == "__main__":
    main()
