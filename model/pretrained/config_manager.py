"""
Configuration manager for pre-trained model training.

Allows loading and validating configurations from JSON files.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Model configurations."""
    name: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.0


@dataclass
class DataConfig:
    """Data configurations."""
    data_source: str = "local"  # "local" or "huggingface"
    data_dir: Optional[str] = "data/complete_sdss"  # Can be None for Hugging Face
    metadata_path: Optional[str] = None
    huggingface_repo: Optional[str] = None  # For Hugging Face datasets
    image_size: int = 224
    train_split: float = 0.8
    batch_size: int = 32
    num_workers: int = 2


@dataclass
class TrainingConfig:
    """Training configurations."""
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    scheduler: str = "step"  # step, cosine, plateau
    scheduler_params: Dict[str, Any] = None
    early_stopping: bool = True
    patience: int = 5
    save_best: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configurations."""
    name: str = "adam"  # adam, sgd, adamw
    lr: float = 0.001
    weight_decay: float = 0.0001
    momentum: float = 0.9  # For SGD
    betas: List[float] = None  # For Adam


@dataclass
class LossConfig:
    """Loss function configurations."""
    name: str = "crossentropy"
    use_class_weights: bool = True
    label_smoothing: float = 0.0


@dataclass
class ExperimentConfig:
    """Experiment configurations."""
    experiment_name: str = "galaxy_classification"
    description: str = ""
    random_seed: int = 42
    device: str = "auto"  # auto, cuda, cpu
    mixed_precision: bool = False


@dataclass
class LoggingConfig:
    """Logging configurations."""
    log_dir: str = "results/logs"
    save_dir: str = "results/models"
    log_interval: int = 10
    save_interval: int = 1
    plot_training: bool = True


class ConfigManager:
    """Configuration manager for training."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to JSON configuration file
        """
        self.config_path = config_path
        self.config = self._create_default_config()
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration."""
        return {
            "model": asdict(ModelConfig()),
            "data": asdict(DataConfig()),
            "training": asdict(TrainingConfig()),
            "optimizer": asdict(OptimizerConfig()),
            "loss": asdict(LossConfig()),
            "experiment": asdict(ExperimentConfig()),
            "logging": asdict(LoggingConfig())
        }
    
    def load_config(self, config_path: str) -> None:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        # Update default configuration with loaded values
        self._update_config(self.config, loaded_config)
        self.config_path = config_path
        
        print(f"Configuration loaded from: {config_path}")
    
    def _update_config(self, default_config: Dict[str, Any], loaded_config: Dict[str, Any]) -> None:
        """
        Update default configuration with loaded values.
        
        Args:
            default_config: Default configuration
            loaded_config: Loaded configuration
        """
        for section, values in loaded_config.items():
            if section in default_config:
                if isinstance(values, dict):
                    for key, value in values.items():
                        if key in default_config[section]:
                            default_config[section][key] = value
                        else:
                            print(f"Warning: Unknown key '{key}' in section '{section}'")
                else:
                    print(f"Warning: Section '{section}' should be a dictionary")
            else:
                print(f"Warning: Unknown section '{section}'")
    
    def save_config(self, save_path: str) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            save_path: Path to save configuration
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4, ensure_ascii=False)
        
        print(f"Configuration saved to: {save_path}")
    
    def get_model_config(self) -> ModelConfig:
        """Return model configuration."""
        return ModelConfig(**self.config["model"])
    
    def get_data_config(self) -> DataConfig:
        """Return data configuration."""
        return DataConfig(**self.config["data"])
    
    def get_training_config(self) -> TrainingConfig:
        """Return training configuration."""
        return TrainingConfig(**self.config["training"])
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Return optimizer configuration."""
        return OptimizerConfig(**self.config["optimizer"])
    
    def get_loss_config(self) -> LossConfig:
        """Return loss function configuration."""
        return LossConfig(**self.config["loss"])
    
    def get_experiment_config(self) -> ExperimentConfig:
        """Return experiment configuration."""
        return ExperimentConfig(**self.config["experiment"])
    
    def get_logging_config(self) -> LoggingConfig:
        """Return logging configuration."""
        return LoggingConfig(**self.config["logging"])
    
    def validate_config(self) -> List[str]:
        """
        Validate current configuration.
        
        Returns:
            List of errors found (empty if valid)
        """
        errors = []
        
        # Basic validations
        model_config = self.get_model_config()
        data_config = self.get_data_config()
        training_config = self.get_training_config()
        
        # Model validation
        from .model_factory import AVAILABLE_MODELS
        if model_config.name not in AVAILABLE_MODELS:
            errors.append(f"Model '{model_config.name}' not supported")
        
        # Data validation
        if data_config.data_source == "local":
            if not data_config.data_dir:
                errors.append("data_dir must be specified for data_source='local'")
            elif not os.path.exists(data_config.data_dir):
                errors.append(f"Data directory not found: {data_config.data_dir}")
        elif data_config.data_source == "huggingface":
            if not data_config.huggingface_repo:
                errors.append("huggingface_repo must be specified for data_source='huggingface'")
        else:
            errors.append(f"data_source must be 'local' or 'huggingface', received: {data_config.data_source}")
        
        if not (0 < data_config.train_split < 1):
            errors.append("train_split must be between 0 and 1")
        
        if data_config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        # Training validation
        if training_config.num_epochs <= 0:
            errors.append("num_epochs must be positive")
        
        if training_config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        return errors
    
    def print_config(self) -> None:
        """Print current configuration."""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        
        for section, values in self.config.items():
            print(f"\n[{section.upper()}]")
            for key, value in values.items():
                print(f"  {key}: {value}")
        
        print("="*60)
    
    def update_config(self, section: str, key: str, value: Any) -> None:
        """
        Update a specific value in the configuration.
        
        Args:
            section: Configuration section
            key: Key to be updated
            value: New value
        """
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
            print(f"Configuration updated: {section}.{key} = {value}")
        else:
            print(f"Error: Section '{section}' or key '{key}' not found")


def load_config(config_path: str) -> ConfigManager:
    """
    Utility function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigManager with loaded configuration
    """
    return ConfigManager(config_path)


def create_config_template(save_path: str) -> None:
    """
    Create a configuration template.
    
    Args:
        save_path: Path to save the template
    """
    config_manager = ConfigManager()
    config_manager.save_config(save_path)
    print(f"Configuration template created at: {save_path}")
