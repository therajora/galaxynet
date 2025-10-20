# `galaxynet/results/` Module

This module provides a comprehensive system for managing trained model versions, logging training processes, and organizing experimental results within the GalaxyNet project.

## Files:

- `__init__.py`: Initializes the `results` package, exposing key components.
- `version_manager.py`: Implements `ModelVersionManager` for creating, tracking, and managing different versions of trained models with automatic directory organization and metadata storage.
- `training_logger.py`: Provides `TrainingLogger` for comprehensive logging of training processes, including configurations, metrics, and progress tracking.
- `example_usage.py`: Demonstrates how to use the versioning system and logging functionality with practical examples.
- `manage_versions.py`: Command-line script for managing model versions (list, show details, cleanup old versions).
- `README.md`: This file, summarizing the module.

## Functionality:

- **Version Management**: Automatically creates unique version IDs, organizes model files, and tracks training metadata.
- **Training Logging**: Comprehensive logging system that records training configurations, dataset information, model details, epoch progress, and final metrics.
- **File Organization**: Automatically creates and manages directory structures for models, logs, and metrics.
- **Cleanup Operations**: Provides utilities to remove old model versions while keeping the most recent ones.
- **Command-Line Interface**: Easy-to-use CLI for managing versions without writing code.

## Quick Usage Example (from `example_usage.py`):

To create a new model version and log training:

```python
from results.version_manager import get_version_manager
from results.training_logger import create_training_logger

# Create version manager
manager = get_version_manager()

# Create new version
version_id = manager.create_new_version(
    model_name="galaxy_cnn",
    model_type="simple", 
    dataset_name="complete_sdss",
    description="Training example"
)

# Create logger
logger = create_training_logger(version_id)

# Log training information
logger.log_config({'num_epochs': 10, 'learning_rate': 0.001})
logger.log_epoch(1, 0.8, 60.0, 0.7, 65.0, 0.001)
logger.log_training_complete(10, 8, 85.0, 120.5)

# Update version status
manager.update_version_status(version_id, 'completed', {'best_val_accuracy': 0.85})
```

## Command-Line Usage (from `manage_versions.py`):

```bash
# List all versions
python results/manage_versions.py list

# List versions for specific model
python results/manage_versions.py list --model-name galaxy_cnn

# Show detailed version information
python results/manage_versions.py show galaxy_cnn_simple_20241229_143022

# Clean old versions (keep 3 most recent)
python results/manage_versions.py cleanup --model-name galaxy_cnn --keep-count 3
```

## Dependencies:

- `pathlib`
- `json`
- `datetime`
- `logging`
- `argparse`
