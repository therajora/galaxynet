#!/usr/bin/env python3
"""
Example usage of the model versioning system.
"""

import sys
from pathlib import Path

# Add root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from results.version_manager import get_version_manager
from results.training_logger import create_training_logger


def example_version_management():
    """Example of version management."""
    print("=" * 60)
    print("VERSION MANAGEMENT EXAMPLE")
    print("=" * 60)
    
    # Get manager
    manager = get_version_manager()
    
    # Create a new version
    version_id = manager.create_new_version(
        model_name="galaxy_cnn",
        model_type="simple",
        dataset_name="complete_sdss",
        description="Training example"
    )
    
    print(f"Version created: {version_id}")
    
    # Simulate training
    print("\nSimulating training...")
    
    # Update status to completed
    metrics = {
        'best_val_accuracy': 0.85,
        'best_epoch': 8,
        'final_epoch': 10,
        'final_train_loss': 0.25,
        'final_val_loss': 0.30,
        'final_train_acc': 0.90,
        'final_val_acc': 0.85
    }
    
    manager.update_version_status(version_id, 'completed', metrics)
    
    # List versions
    print("\nAvailable versions:")
    manager.print_versions_summary()
    
    # Show version information
    print(f"\nVersion information for {version_id}:")
    version_info = manager.get_version_info(version_id)
    if version_info:
        for key, value in version_info.items():
            print(f"  {key}: {value}")


def example_logging():
    """Example of logger usage."""
    print("\n" + "=" * 60)
    print("LOGGING EXAMPLE")
    print("=" * 60)
    
    # Create logger
    version_id = "example_logging_20241229_143022"
    logger = create_training_logger(version_id)
    
    # Log configurations
    config = {
        'num_epochs': 10,
        'learning_rate': 0.001,
        'batch_size': 32,
        'device': 'cuda'
    }
    logger.log_config(config)
    
    # Log dataset information
    dataset_info = {
        'total_samples': 1000,
        'classes': ['reg', 'irr_pec'],
        'train_samples': 800,
        'val_samples': 200
    }
    logger.log_dataset_info(dataset_info)
    
    # Log model information
    model_info = {
        'model_type': 'GalaxyNetCNN',
        'num_classes': 2,
        'total_parameters': 1234567,
        'trainable_parameters': 1234567
    }
    logger.log_model_info(model_info)
    
    # Simulate some epochs
    for epoch in range(1, 4):
        train_loss = 0.8 - epoch * 0.1
        train_acc = 60 + epoch * 5
        val_loss = 0.7 - epoch * 0.08
        val_acc = 65 + epoch * 4
        lr = 0.001 * (0.9 ** epoch)
        
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, lr)
        
        if epoch == 2:
            logger.log_best_model(epoch, val_acc)
    
    # Log completion
    logger.log_training_complete(3, 2, 73.0, 120.5)
    
    # Save metrics
    metrics = {
        'best_val_accuracy': 0.73,
        'best_epoch': 2,
        'final_epoch': 3
    }
    logger.save_metrics(metrics, "results/metrics/example_metrics.json")
    
    # Close logger
    logger.close()
    
    print(f"\nLog saved to: results/logs/{version_id}.log")


def example_cleanup():
    """Example of version cleanup."""
    print("\n" + "=" * 60)
    print("CLEANUP EXAMPLE")
    print("=" * 60)
    
    manager = get_version_manager()
    
    # Create several example versions
    for i in range(7):
        version_id = manager.create_new_version(
            model_name="test_model",
            model_type="simple",
            dataset_name="test_dataset",
            description=f"Test version {i+1}"
        )
        
        # Mark as completed
        metrics = {'best_val_accuracy': 0.8 + i * 0.01}
        manager.update_version_status(version_id, 'completed', metrics)
    
    print("Versions created:")
    manager.print_versions_summary("test_model")
    
    # Clean old versions (keep 3)
    print("\nCleaning old versions (keeping 3 most recent)...")
    manager.cleanup_old_versions("test_model", keep_count=3)
    
    print("\nVersions after cleanup:")
    manager.print_versions_summary("test_model")


if __name__ == "__main__":
    print("MODEL VERSIONING SYSTEM USAGE EXAMPLES")
    print("=" * 60)
    
    examples = {
        '1': ("Version Management", example_version_management),
        '2': ("Training Logging", example_logging),
        '3': ("Version Cleanup", example_cleanup),
    }
    
    print("Choose an example:")
    for key, (name, _) in examples.items():
        print(f"  {key}) {name}")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        try:
            func()
        except Exception as e:
            print(f"Error during execution: {e}")
    else:
        print("Invalid choice")
