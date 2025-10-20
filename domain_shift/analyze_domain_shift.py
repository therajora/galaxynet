"""
Main script for domain shift analysis.

This script allows running domain shift analyses between SDSS and S-PLUS
using trained models.
"""

import argparse
import sys
import torch
from pathlib import Path
import json
import os
from typing import List, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from domain_shift.analyzer import DomainShiftAnalyzer
from model.pretrained.train_pretrained import setup_seeds, setup_device
from results.version_manager import get_version_manager


def find_trained_models(results_dir: Path = Path("results/models")) -> List[Dict[str, Any]]:
    """
    Find available trained models.
    
    Args:
        results_dir: Directory with trained models
        
    Returns:
        List of model configurations
    """
    models = []
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return models
    
    # Search for .pth files
    for model_file in results_dir.rglob("*.pth"):
        if model_file.name in ["best_model.pth", "final_model.pth"]:
            # Extract information from path
            model_path = model_file.parent
            model_name = model_path.name
            
            # Try to infer model type from name
            if "resnet" in model_name.lower():
                model_type = "resnet50_v1"
            elif "efficientnet" in model_name.lower():
                model_type = "efficientnet_b0"
            elif "vit" in model_name.lower():
                model_type = "vit_b_16"
            else:
                model_type = "resnet50_v1"  # Default
            
            models.append({
                'model_name': model_name,
                'model_type': model_type,
                'model_path': str(model_file),
                'model_dir': str(model_path)
            })
    
    return models


def analyze_single_model(
    model_name: str,
    model_path: str,
    sdss_data_dir: Path,
    splus_data_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 32
):
    """
    Analyze domain shift for a specific model.
    
    Args:
        model_name: Model name
        model_path: Path to model
        sdss_data_dir: Directory with SDSS data
        splus_data_dir: Directory with S-PLUS data
        output_dir: Output directory
        device: Computing device
        batch_size: Batch size
    """
    print(f"Analyzing domain shift for model: {model_name}")
    
    # Create specific output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create analyzer
    analyzer = DomainShiftAnalyzer(
        model_name=model_name,
        model_path=Path(model_path),
        device=device,
        output_dir=model_output_dir
    )
    
    # Run analysis
    results = analyzer.analyze_domain_shift(
        sdss_data_dir=sdss_data_dir,
        splus_data_dir=splus_data_dir,
        batch_size=batch_size
    )
    
    print(f"Analysis completed for {model_name}")
    print(f"Results saved to: {model_output_dir}")
    
    return results


def analyze_multiple_models(
    models_config: List[Dict[str, Any]],
    sdss_data_dir: Path,
    splus_data_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 32
):
    """
    Analyze domain shift for multiple models.
    
    Args:
        models_config: List of model configurations
        sdss_data_dir: Directory with SDSS data
        splus_data_dir: Directory with S-PLUS data
        output_dir: Output directory
        device: Computing device
        batch_size: Batch size
    """
    print(f"Analyzing domain shift for {len(models_config)} models...")
    
    all_results = {}
    
    for config in models_config:
        model_name = config['model_name']
        model_path = config['model_path']
        
        try:
            results = analyze_single_model(
                model_name=model_name,
                model_path=model_path,
                sdss_data_dir=sdss_data_dir,
                splus_data_dir=splus_data_dir,
                output_dir=output_dir,
                device=device,
                batch_size=batch_size
            )
            all_results[model_name] = results
            
        except Exception as e:
            print(f"Error analyzing model {model_name}: {e}")
            continue
    
    # Save general comparison
    comparison_path = output_dir / "models_comparison.json"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False, default=str)
    
    print(f"Model comparison saved to: {comparison_path}")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Domain Shift Analysis between SDSS and S-PLUS"
    )
    
    # Main arguments
    parser.add_argument("--mode", type=str, required=True,
                       choices=["single", "multiple", "find"],
                       help="Operation mode: single (one model), multiple (multiple), find (find models)")
    
    # Arguments for single mode
    parser.add_argument("--model-name", type=str,
                       help="Model name (e.g., resnet50_v1)")
    parser.add_argument("--model-path", type=str,
                       help="Path to model .pth file")
    
    # Data arguments
    parser.add_argument("--sdss-data-dir", type=str, default="data/complete_sdss",
                       help="Directory with SDSS data")
    parser.add_argument("--splus-data-dir", type=str, default="data/complete_splus",
                       help="Directory with S-PLUS data")
    
    # Output arguments
    parser.add_argument("--output-dir", type=str, default="domain_shift_results",
                       help="Directory to save results")
    
    # Technical arguments
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Computing device")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed for reproducibility")
    
    # Arguments for finding models
    parser.add_argument("--results-dir", type=str, default="results/models",
                       help="Directory where trained models are stored")
    
    args = parser.parse_args()
    
    # Initial setup
    setup_seeds(args.seed)
    device = setup_device(args.device)
    
    # Convert paths
    sdss_data_dir = Path(args.sdss_data_dir)
    splus_data_dir = Path(args.splus_data_dir)
    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    
    # Check if data directories exist
    if not sdss_data_dir.exists():
        print(f"Error: SDSS directory not found: {sdss_data_dir}")
        sys.exit(1)
    
    if not splus_data_dir.exists():
        print(f"Error: S-PLUS directory not found: {splus_data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.mode == "find":
        print("Searching for trained models...")
        models = find_trained_models(results_dir)
        
        if not models:
            print("No trained models found.")
            print(f"Searched in: {results_dir}")
            print("Train some models first using:")
            print("  ./run.sh train-pretrained -m efficientnet_b0")
            return
        
        print(f"Found {len(models)} models:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model['model_name']} ({model['model_type']})")
            print(f"     Path: {model['model_path']}")
        
        print(f"\nTo analyze all models:")
        print(f"python {Path(__file__).name} --mode multiple --sdss-data-dir {args.sdss_data_dir} --splus-data-dir {args.splus_data_dir}")
        
        print(f"\nTo analyze a specific model:")
        print(f"python {Path(__file__).name} --mode single --model-name {models[0]['model_type']} --model-path {models[0]['model_path']}")
    
    elif args.mode == "single":
        if not args.model_name or not args.model_path:
            parser.error("For 'single' mode, --model-name and --model-path are required.")
        
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"Error: Model not found: {model_path}")
            sys.exit(1)
        
        results = analyze_single_model(
            model_name=args.model_name,
            model_path=args.model_path,
            sdss_data_dir=sdss_data_dir,
            splus_data_dir=splus_data_dir,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size
        )
        
        print("Domain shift analysis completed successfully!")
        print(f"Results saved to: {output_dir / args.model_name}")
    
    elif args.mode == "multiple":
        print("Searching for models for multiple analysis...")
        models = find_trained_models(results_dir)
        
        if not models:
            print("No models found for multiple analysis.")
            print("Use --mode find to see available models.")
            return
        
        print(f"Analyzing {len(models)} models...")
        
        # Convert to expected format
        models_config = [
            {
                'model_name': model['model_type'],
                'model_path': model['model_path']
            }
            for model in models
        ]
        
        results = analyze_multiple_models(
            models_config=models_config,
            sdss_data_dir=sdss_data_dir,
            splus_data_dir=splus_data_dir,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size
        )
        
        print("Multiple domain shift analysis completed!")
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
