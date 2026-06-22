#!/usr/bin/env python3
"""
Main script for training CNN models from scratch.
Based on scratch.py with modularization and improvements.
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path

# Add root directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Relative imports
from model.from_scratch.dataset import (
    GalaxyDataset, calculate_mean_std, create_transforms
)
from model.from_scratch.model import create_model, count_parameters
from model.from_scratch.trainer import GalaxyTrainer, create_data_loaders
from model.from_scratch.predictor import GalaxyPredictor


def setup_seeds(seed=42):
    """Setup seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Seeds configured: {seed}")


def run_from_scratch_entry(
    *,
    data_dir: str,
    device: str,
    seed: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    model_name: str,
    predict_samples: int,
    evaluate: bool,
    metadata_file: str = "galaxy_metadata.pth",
    model_type: str = "simple",
    input_size: tuple[int, int] = (224, 224),
    dropout_rate: float = 0.5,
    train_split: float = 0.8,
    num_workers: int = 2,
    save_dir: str = "model_artifacts",
) -> dict:
    """Run the from-scratch training flow and return a structured summary."""
    setup_seeds(seed)

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    print(f"Selected device: {resolved_device}")

    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return {
            "success": False,
            "message": f"Data directory not found: {data_dir}",
            "output_dir": None,
            "artifacts_path": None,
            "metrics_path": None,
        }

    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("CNN TRAINING FOR GALAXY CLASSIFICATION")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Model type: {model_type}")
    print(f"Image size: {input_size}")
    print(f"Batch size: {batch_size}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)

    print("\nCreating dataset and calculating statistics...")

    temp_transform, _ = create_transforms()
    temp_dataset = GalaxyDataset(
        img_dir=data_dir,
        metadata_path=metadata_file,
        transform=temp_transform
    )

    dataset_mean, dataset_std = calculate_mean_std(temp_dataset)
    print(f"Calculated mean: {dataset_mean}")
    print(f"Calculated standard deviation: {dataset_std}")

    train_transforms, viz_transforms = create_transforms(
        mean=dataset_mean,
        std=dataset_std,
        image_size=tuple(input_size)
    )

    full_dataset = GalaxyDataset(
        img_dir=data_dir,
        metadata_path=metadata_file,
        transform=train_transforms
    )

    class_dist = full_dataset.get_class_distribution()
    print(f"\nClass distribution:")
    for class_name, count in class_dist.items():
        percentage = (count / len(full_dataset)) * 100
        print(f"  {class_name}: {count} images ({percentage:.1f}%)")

    print(f"\nCreating DataLoaders...")
    train_loader, val_loader = create_data_loaders(
        dataset=full_dataset,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
        random_seed=seed
    )

    print(f"\nCreating model...")
    num_classes = len(full_dataset.classes)
    model = create_model(
        model_type=model_type,
        num_classes=num_classes,
        input_size=tuple(input_size),
        dropout_rate=dropout_rate
    )

    if hasattr(model, 'print_model_summary'):
        model.print_model_summary()
    else:
        params = count_parameters(model)
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable parameters: {params['trainable']:,}")

    trainer = GalaxyTrainer(
        model=model,
        device=resolved_device,
        save_dir=save_dir,
        use_versioning=True,
        model_name=model_name,
        model_type=model_type,
        dataset_name=os.path.basename(data_dir),
        description=f"Training with {num_epochs} epochs, lr={learning_rate}"
    )

    print(f"\nStarting training...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        save_best=True,
        patience=5
    )

    artifacts_path = os.path.join(trainer.save_dir, f"{model_name}_artifacts.pth")
    artifacts = {
        'mean': dataset_mean,
        'std': dataset_std,
        'class_names': full_dataset.classes,
        'model_type': model_type,
        'input_size': list(input_size),
        'num_classes': num_classes
    }
    torch.save(artifacts, artifacts_path)
    print(f"Artifacts saved to: {artifacts_path}")

    trainer.plot_training_history(save_plot=True)

    if predict_samples > 0:
        print(f"\nTesting predictions on {predict_samples} samples...")

        trainer.load_model("best_model.pth")

        predictor = GalaxyPredictor(
            model=model,
            class_names=full_dataset.classes,
            mean=dataset_mean,
            std=dataset_std,
            device=resolved_device
        )

        results = predictor.predict_random_sample(data_dir, predict_samples)

        for result in results:
            predictor.print_prediction_result(result, show_all_probs=True)

    if evaluate:
        print(f"\nEvaluating model...")
        metrics = predictor.evaluate_accuracy(data_dir, num_samples=200)
        predictor.print_evaluation_results(metrics)

    print(f"\nTraining completed successfully!")
    print(f"Models saved to: {trainer.save_dir}")
    print(f"Artifacts saved to: {artifacts_path}")

    return {
        "success": True,
        "message": "Training completed successfully!",
        "output_dir": trainer.save_dir,
        "artifacts_path": artifacts_path,
        "metrics_path": None,
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="CNN training for galaxy classification")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data/complete_sdss",
                       help="Directory with training data")
    parser.add_argument("--metadata_file", type=str, default="galaxy_metadata.pth",
                       help="Metadata file")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default="simple", choices=["simple", "v2"],
                       help="Model type")
    parser.add_argument("--input_size", type=int, nargs=2, default=[224, 224],
                       help="Image size (height width)")
    parser.add_argument("--dropout_rate", type=float, default=0.5,
                       help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Training proportion")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (cuda/cpu/auto)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42,
                       help="Seed for reproducibility")
    
    # Saving arguments
    parser.add_argument("--save_dir", type=str, default="model_artifacts",
                       help="Directory to save models")
    parser.add_argument("--model_name", type=str, default="galaxy_cnn",
                       help="Model name")
    
    # Prediction arguments
    parser.add_argument("--predict_samples", type=int, default=5,
                       help="Number of samples for prediction after training")
    parser.add_argument("--evaluate", action="store_true",
                       help="Evaluate model after training")
    
    args = parser.parse_args()
    run_from_scratch_entry(
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        model_name=args.model_name,
        predict_samples=args.predict_samples,
        evaluate=args.evaluate,
        metadata_file=args.metadata_file,
        model_type=args.model_type,
        input_size=tuple(args.input_size),
        dropout_rate=args.dropout_rate,
        train_split=args.train_split,
        num_workers=args.num_workers,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
