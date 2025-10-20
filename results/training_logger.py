"""
Custom logger for model training.
"""

import os
import logging
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class TrainingLogger:
    """
    Custom logger for model training.
    """
    
    def __init__(self, log_file: str, version_id: str):
        """
        Initialize the logger.
        
        Args:
            log_file: Log file path
            version_id: Model version ID
        """
        self.log_file = Path(log_file)
        self.version_id = version_id
        
        # Create directory if it doesn't exist
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logger
        self.logger = logging.getLogger(f"training_{version_id}")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Message format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initial log
        self.logger.info(f"Starting training - Version: {version_id}")
        self.logger.info("=" * 60)

    def log_config(self, config: Dict[str, Any]):
        """
        Log training configurations.
        
        Args:
            config: Dictionary with configurations
        """
        self.logger.info("TRAINING CONFIGURATIONS")
        self.logger.info("-" * 40)
        
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("-" * 40)

    def log_dataset_info(self, dataset_info: Dict[str, Any]):
        """
        Log dataset information.
        
        Args:
            dataset_info: Dataset information
        """
        self.logger.info("DATASET INFORMATION")
        self.logger.info("-" * 40)
        
        for key, value in dataset_info.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("-" * 40)

    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model information.
        
        Args:
            model_info: Model information
        """
        self.logger.info("MODEL INFORMATION")
        self.logger.info("-" * 40)
        
        for key, value in model_info.items():
            self.logger.info(f"{key}: {value}")
        
        self.logger.info("-" * 40)

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float, val_acc: float, learning_rate: float = None):
        """
        Log results of an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            learning_rate: Current learning rate
        """
        lr_str = f", LR: {learning_rate:.6f}" if learning_rate else ""
        
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{lr_str}"
        )

    def log_best_model(self, epoch: int, val_acc: float):
        """
        Log when a new best model is found.
        
        Args:
            epoch: Best model epoch
            val_acc: Validation accuracy
        """
        self.logger.info(f"NEW BEST MODEL! Epoch {epoch}, Accuracy: {val_acc:.2f}%")

    def log_early_stopping(self, epoch: int, patience: int):
        """
        Log when early stopping is activated.
        
        Args:
            epoch: Current epoch
            patience: Configured patience
        """
        self.logger.warning(f"Early stopping activated at epoch {epoch} (patience: {patience})")

    def log_training_complete(self, total_epochs: int, best_epoch: int, 
                            best_val_acc: float, training_time: float):
        """
        Log training completion.
        
        Args:
            total_epochs: Total epochs executed
            best_epoch: Best model epoch
            best_val_acc: Best validation accuracy
            training_time: Total training time
        """
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("-" * 40)
        self.logger.info(f"Total epochs: {total_epochs}")
        self.logger.info(f"Best epoch: {best_epoch}")
        self.logger.info(f"Best accuracy: {best_val_acc:.2f}%")
        self.logger.info(f"Total time: {training_time:.2f} seconds")
        self.logger.info("=" * 60)

    def log_error(self, error: Exception):
        """
        Log errors during training.
        
        Args:
            error: Exception occurred
        """
        self.logger.error(f"TRAINING ERROR: {str(error)}")
        self.logger.error(f"Error type: {type(error).__name__}")

    def log_prediction_results(self, results: Dict[str, Any]):
        """
        Log test prediction results.
        
        Args:
            results: Prediction results
        """
        self.logger.info("PREDICTION RESULTS")
        self.logger.info("-" * 40)
        
        for result in results:
            if 'error' in result:
                self.logger.error(f"Error in {result['image_path']}: {result['error']}")
            else:
                self.logger.info(
                    f"Image: {os.path.basename(result['image_path'])} | "
                    f"Prediction: {result['predicted_class']} | "
                    f"Confidence: {result['confidence']:.2%}"
                )

    def log_evaluation_results(self, metrics: Dict[str, Any]):
        """
        Log evaluation results.
        
        Args:
            metrics: Evaluation metrics
        """
        self.logger.info("EVALUATION RESULTS")
        self.logger.info("-" * 40)
        self.logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        self.logger.info(f"Total Samples: {metrics['total_samples']}")
        self.logger.info(f"Correct Predictions: {metrics['correct_predictions']}")
        
        self.logger.info("Accuracy by Class:")
        for class_name, accuracy in metrics['class_accuracies'].items():
            correct = metrics['class_correct'][class_name]
            total = metrics['class_totals'][class_name]
            self.logger.info(f"  {class_name}: {accuracy:.2%} ({correct}/{total})")

    def save_metrics(self, metrics: Dict[str, Any], metrics_file: str):
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Metrics to save
            metrics_file: Metrics file path
        """
        metrics_path = Path(metrics_file)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        metrics_data = {
            'version_id': self.version_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'metrics': metrics
        }
        
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Metrics saved to: {metrics_path}")

    def close(self):
        """Close the logger."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        
        self.logger.info("Logger closed.")


def create_training_logger(version_id: str, logs_dir: str = "results/logs") -> TrainingLogger:
    """
    Create a training logger.
    
    Args:
        version_id: Model version ID
        logs_dir: Logs directory
        
    Returns:
        TrainingLogger: Configured logger
    """
    log_file = Path(logs_dir) / f"{version_id}.log"
    return TrainingLogger(str(log_file), version_id)
