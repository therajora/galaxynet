"""
Main script for galaxy classification model validation.

Allows:
- Validate a single model
- Run benchmark of multiple models
- Generate comparison reports
"""

import sys
import torch
from pathlib import Path
from typing import Any, Dict

# Add root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.pretrained.dataset import GalaxyPretrainedDataset, create_data_loaders, get_imagenet_transforms
from validation.validator import ModelValidator
from validation.benchmark import ModelBenchmark
from validation.validation_cli import main as validation_main

CLASS_NAMES = ["Regular", "Peculiar"]


def resolve_device(device: str):
    """Resolve the runtime device preserving the current CLI behavior."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_test_loader(data_dir: str):
    """Build the default validation loader used by single and benchmark flows."""
    transforms = get_imagenet_transforms(224, is_train=False)
    test_dataset = GalaxyPretrainedDataset(
        img_dir=data_dir,
        metadata_path=None,
        transform=transforms,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def _run_single_validation(
    *,
    model_name: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
    device: str,
):
    print(f"Validating model: {model_name}")

    resolved_device = resolve_device(device)
    test_loader = build_test_loader(data_dir)
    validator = ModelValidator(
        model_name=model_name,
        model_path=model_path,
        device=resolved_device,
        class_names=CLASS_NAMES,
    )
    return validator.validate_dataset(
        test_loader=test_loader,
        save_results=True,
        output_dir=output_dir,
    )


def validate_single_entry(
    *,
    model_name: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
    device: str,
) -> Dict:
    """Run single-model validation and return a structured payload."""
    _run_single_validation(
        model_name=model_name,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
    return {
        "success": True,
        "message": "Validation completed successfully!",
        "output_dir": output_dir,
        "results_path": f"{output_dir}/metrics.json",
        "model_paths": None,
    }


def _run_benchmark_validation(
    *,
    model_paths: Dict[str, str],
    data_dir: str,
    output_dir: str,
    device: str,
) -> Any:
    print(f"Running benchmark of {len(model_paths)} models...")

    resolved_device = resolve_device(device)
    test_loader = build_test_loader(data_dir)
    benchmark = ModelBenchmark(
        test_loader=test_loader,
        class_names=CLASS_NAMES,
        device=resolved_device,
        output_dir=output_dir,
    )
    results = benchmark.run_benchmark(model_paths, save_results=True)
    benchmark.print_summary()
    return results


def run_benchmark_entry(
    *,
    model_paths: Dict[str, str],
    data_dir: str,
    output_dir: str,
    device: str,
) -> Dict:
    """Run the benchmark flow and return a structured payload."""
    _run_benchmark_validation(
        model_paths=model_paths,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
    return {
        "success": True,
        "message": "Benchmark completed successfully!",
        "output_dir": output_dir,
        "results_path": f"{output_dir}/benchmark_comparison.csv",
        "model_paths": model_paths,
    }


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
    return _run_single_validation(
        model_name=model_name,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )


def run_benchmark(
    model_paths: Dict[str, str],
    data_dir: str,
    output_dir: str = "benchmark_results",
    device: str = "auto"
) -> Any:
    """
    Run benchmark of multiple models.
    
    Args:
        model_paths: Dictionary {model_name: model_path}
        data_dir: Data directory
        output_dir: Output directory
        device: Device
    """
    return _run_benchmark_validation(
        model_paths=model_paths,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )


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
    return validation_main()


if __name__ == "__main__":
    raise SystemExit(main())
