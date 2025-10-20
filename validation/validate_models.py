"""
Main script for galaxy classification model validation.

Allows:
- Validate a single model
- Run benchmark of multiple models
- Generate comparison reports
"""

import argparse
import sys
import os
import torch
from pathlib import Path
from typing import Dict, List

# Add root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.pretrained.dataset import GalaxyPretrainedDataset, create_data_loaders, get_imagenet_transforms
from validation.validator import ModelValidator
from validation.benchmark import ModelBenchmark


def validate_single_model(
    model_name: str,
    model_path: str,
    data_dir: str,
    output_dir: str = None,
    device: str = "auto"
) -> Dict:
    """
    Validate a single model.
    
    Args:
        model_name: Model name
        model_path: Path to trained model
        data_dir: Data directory
        output_dir: Output directory
        device: Device
        
    Returns:
        Validation results
    """
    print(f"Validating model: {model_name}")
    
    # Configure device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create test dataset
    transforms = get_imagenet_transforms(224, is_train=False)
    test_dataset = GalaxyPretrainedDataset(
        img_dir=data_dir,
        metadata_path=None,
        transform=transforms
    )
    
    # Create DataLoader (use entire dataset as test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create validator
    validator = ModelValidator(
        model_name=model_name,
        model_path=model_path,
        device=device,
        class_names=['Regular', 'Peculiar']
    )
    
    # Execute validation
    results = validator.validate_dataset(
        test_loader=test_loader,
        save_results=True,
        output_dir=output_dir
    )
    
    return results


def run_benchmark(
    model_paths: Dict[str, str],
    data_dir: str,
    output_dir: str = "benchmark_results",
    device: str = "auto"
) -> None:
    """
    Run benchmark of multiple models.
    
    Args:
        model_paths: Dictionary {model_name: model_path}
        data_dir: Data directory
        output_dir: Output directory
        device: Device
    """
    print(f"Running benchmark of {len(model_paths)} models...")
    
    # Configure device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Create test dataset
    transforms = get_imagenet_transforms(224, is_train=False)
    test_dataset = GalaxyPretrainedDataset(
        img_dir=data_dir,
        metadata_path=None,
        transform=transforms
    )
    
    # Create DataLoader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create benchmark
    benchmark = ModelBenchmark(
        test_loader=test_loader,
        class_names=['Regular', 'Peculiar'],
        device=device,
        output_dir=output_dir
    )
    
    # Execute benchmark
    results_df = benchmark.run_benchmark(model_paths, save_results=True)
    benchmark.print_summary()
    
    return results_df


def find_trained_models(results_dir: str = "results/models") -> Dict[str, str]:
    """
    Find trained models in results directory.
    
    Args:
        results_dir: Results directory
        
    Returns:
        Dictionary {model_name: model_path}
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        return {}
    
    model_paths = {}
    
    # Search for model files
    for model_file in results_path.rglob("best_model.pth"):
        # Extract model name from path
        model_name = model_file.parent.name
        model_paths[model_name] = str(model_file)
    
    print(f"Found {len(model_paths)} trained models:")
    for name, path in model_paths.items():
        print(f"  - {name}: {path}")
    
    return model_paths


def main():
    parser = argparse.ArgumentParser(description="Galaxy classification model validation")
    
    # Operation mode
    parser.add_argument("--mode", choices=["single", "benchmark", "find"], required=True,
                       help="Operation mode: single (one model), benchmark (multiple), find (find models)")
    
    # Parameters for individual validation
    parser.add_argument("--model-name", type=str, help="Model name for individual validation")
    parser.add_argument("--model-path", type=str, help="Model path for individual validation")
    
    # General parameters
    parser.add_argument("--data-dir", type=str, default="data/complete_sdss",
                       help="Test data directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                       help="Device for validation")
    parser.add_argument("--results-dir", type=str, default="results/models",
                       help="Results directory to find models")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.model_name or not args.model_path:
            print("For 'single' mode, --model-name and --model-path are required")
            return
        
        output_dir = args.output_dir or f"validation_results/{args.model_name}"
        
        try:
            results = validate_single_model(
                model_name=args.model_name,
                model_path=args.model_path,
                data_dir=args.data_dir,
                output_dir=output_dir,
                device=args.device
            )
            print(f"Validation completed successfully!")
            
        except Exception as e:
            print(f"Error during validation: {e}")
    
    elif args.mode == "benchmark":
        # Find models automatically
        model_paths = find_trained_models(args.results_dir)
        
        if not model_paths:
            print("No trained models found")
            return
        
        output_dir = args.output_dir or "benchmark_results"
        
        try:
            results_df = run_benchmark(
                model_paths=model_paths,
                data_dir=args.data_dir,
                output_dir=output_dir,
                device=args.device
            )
            print(f"Benchmark completed successfully!")
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
    
    elif args.mode == "find":
        model_paths = find_trained_models(args.results_dir)
        if model_paths:
            print(f"\nTo run benchmark with these models:")
            print(f"python validation/validate_models.py --mode benchmark --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
