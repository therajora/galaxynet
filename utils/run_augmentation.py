#!/usr/bin/env python3
"""
Main script to execute data augmentation and balancing.
"""

import argparse
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from augment import AugmentationPipeline, DataAugmenter, DataBalancer


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Galaxy Data Augmentation and Balancing Pipeline"
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory with images'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for processed images'
    )
    
    parser.add_argument(
        '--source', '-s',
        default='unknown',
        help='Image source (sdss, splus, etc.)'
    )
    
    parser.add_argument(
        '--augmentations', '-a',
        type=int,
        default=5,
        help='Number of augmentations per image (default: 5)'
    )
    
    parser.add_argument(
        '--strategy', '-st',
        choices=['mean', 'max', 'min', 'custom'],
        default='custom',
        help='Balancing strategy (default: custom)'
    )
    
    parser.add_argument(
        '--mode', '-m',
        choices=['augment', 'balance', 'complete', 'analyze'],
        default='complete',
        help='Execution mode (default: complete)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("=== Data Augmentation and Balancing Pipeline ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Source: {args.source}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print()
    
    try:
        if args.mode == 'analyze':
            # Analysis only
            pipeline = AugmentationPipeline(seed=args.seed)
            analysis = pipeline.analyze_current_data(args.input)
            suggested = pipeline.suggest_balancing_strategy(args.input)
            print(f"Suggested strategy: {suggested}")
            
        elif args.mode == 'augment':
            # Augmentation only
            augmenter = DataAugmenter(args.augmentations, args.seed)
            stats = augmenter.run_augmentation(args.input, args.output, args.source)
            
        elif args.mode == 'balance':
            # Balancing only
            balancer = DataBalancer(args.seed)
            stats = balancer.create_balanced_dataset(
                args.input, args.output, args.strategy, args.augmentations
            )
            balancer.print_balance_report(stats)
            
        elif args.mode == 'complete':
            # Complete pipeline
            pipeline = AugmentationPipeline(args.augmentations, args.seed)
            
            # Analyze data first
            print("Analyzing current data...")
            analysis = pipeline.analyze_current_data(args.input)
            suggested = pipeline.suggest_balancing_strategy(args.input)
            print(f"Suggested strategy: {suggested}")
            print(f"Using strategy: {args.strategy}")
            print()
            
            # Execute pipeline
            stats = pipeline.run_complete_pipeline(
                args.input, args.output, args.source, 
                args.strategy, balance_data=True
            )
            
            # Print summary
            pipeline.print_pipeline_summary(stats)
        
        print("\n✅ Process completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
