"""
Training module for galaxy classification models.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Tuple
import json
from datetime import datetime
import sys
from pathlib import Path

# Add root directory to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from results.version_manager import get_version_manager
from results.training_logger import create_training_logger


class GalaxyTrainer:
    """
    Class for training galaxy classification models.
    """
    
    def __init__(self, model, device=None, save_dir="model_artifacts", 
                 use_versioning=True, model_name="galaxy_cnn", model_type="simple",
                 dataset_name="complete_sdss", description=""):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model for training
            device: Device for training (cuda/cpu)
            save_dir: Directory to save models and artifacts
            use_versioning: Whether to use versioning system
            model_name: Model name for versioning
            model_type: Model type for versioning
            dataset_name: Dataset name for versioning
            description: Version description
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_dir = save_dir
        self.use_versioning = use_versioning
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        # Versioning system
        self.version_id = None
        self.version_manager = None
        self.training_logger = None
        
        if use_versioning:
            self.version_manager = get_version_manager()
            self.version_id = self.version_manager.create_new_version(
                model_name=model_name,
                model_type=model_type,
                dataset_name=dataset_name,
                description=description
            )
            self.training_logger = create_training_logger(self.version_id)
            
            # Update save_dir to use version directory
            version_info = self.version_manager.get_version_info(self.version_id)
            if version_info:
                self.save_dir = str(Path(version_info['model_path']).parent)
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"Trainer initialized - Device: {self.device}")
        print(f"Save directory: {self.save_dir}")
        if self.version_id:
            print(f"Version: {self.version_id}")

    def train_epoch(self, train_loader, criterion, optimizer) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training DataLoader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{running_loss / (train_pbar.n + 1):.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc

    def validate_epoch(self, val_loader, criterion) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: Validation DataLoader
            criterion: Loss function
            
        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{val_loss / (val_pbar.n + 1):.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = val_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001,
              criterion=None, optimizer=None, scheduler=None, 
              save_best=True, patience=5, dataset_info=None, model_info=None) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            save_best: Whether to save the best model
            patience: Patience for early stopping
            
        Returns:
            dict: Training history
        """
        # Default configurations
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Learning rate: {learning_rate}")
        print(f"Device: {self.device}")
        
        # Log configurations if using versioning
        if self.training_logger:
            config = {
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': train_loader.batch_size,
                'device': str(self.device),
                'save_best': save_best,
                'patience': patience
            }
            self.training_logger.log_config(config)
            
            if dataset_info:
                self.training_logger.log_dataset_info(dataset_info)
            
            if model_info:
                self.training_logger.log_model_info(model_info)
        
        best_val_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update history
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['epochs'].append(epoch + 1)
            
            # Print results
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            # Scheduler step
            current_lr = None
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Current learning rate: {current_lr:.6f}")
            
            # Log epoch
            if self.training_logger:
                self.training_logger.log_epoch(
                    epoch + 1, train_loss, train_acc, val_loss, val_acc, current_lr
                )
            
            # Save best model
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0
                self.save_model("best_model.pth")
                print(f"New best model saved! (Acc: {val_acc:.2f}%)")
                
                if self.training_logger:
                    self.training_logger.log_best_model(epoch + 1, val_acc)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping activated after {patience} epochs without improvement")
                if self.training_logger:
                    self.training_logger.log_early_stopping(epoch + 1, patience)
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
        
        # Save final model
        self.save_model("final_model.pth")
        
        # Save history
        self.save_training_history()
        
        # Log training completion
        if self.training_logger:
            training_time = sum(self.train_history.get('epoch_times', [0]))
            self.training_logger.log_training_complete(
                len(self.train_history['epochs']), best_epoch, best_val_acc, training_time
            )
            
            # Update version status
            if self.version_manager:
                metrics = {
                    'best_val_accuracy': best_val_acc / 100,
                    'best_epoch': best_epoch,
                    'final_epoch': len(self.train_history['epochs']),
                    'final_train_loss': self.train_history['train_loss'][-1] if self.train_history['train_loss'] else 0,
                    'final_val_loss': self.train_history['val_loss'][-1] if self.train_history['val_loss'] else 0,
                    'final_train_acc': self.train_history['train_acc'][-1] if self.train_history['train_acc'] else 0,
                    'final_val_acc': self.train_history['val_acc'][-1] if self.train_history['val_acc'] else 0
                }
                
                self.version_manager.update_version_status(
                    self.version_id, 'completed', metrics
                )
                
                # Save metrics
                version_info = self.version_manager.get_version_info(self.version_id)
                if version_info:
                    self.training_logger.save_metrics(metrics, version_info['metrics_path'])
        
        return self.train_history

    def save_model(self, filename):
        """Save the model."""
        model_path = os.path.join(self.save_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    def load_model(self, filename):
        """Load the model."""
        model_path = os.path.join(self.save_dir, filename)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"Model loaded from: {model_path}")

    def save_training_history(self):
        """Save training history."""
        history_path = os.path.join(self.save_dir, "training_history.json")
        
        # Add metadata
        history_data = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'history': self.train_history
        }
        
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"History saved to: {history_path}")

    def plot_training_history(self, save_plot=True):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            epochs = self.train_history['epochs']
            
            # Plot loss
            ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train')
            ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Validation')
            ax1.set_title('Loss per Epoch')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train')
            ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Validation')
            ax2.set_title('Accuracy per Epoch')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            
            if save_plot:
                plot_path = os.path.join(self.save_dir, "training_history.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Plot saved to: {plot_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Plot will not be generated.")


def create_data_loaders(dataset, batch_size=32, train_split=0.8, num_workers=2, 
                       shuffle=True, random_seed=42):
    """
    Create training and validation DataLoaders.
    
    Args:
        dataset: Complete dataset
        batch_size: Batch size
        train_split: Training proportion
        num_workers: Number of workers
        shuffle: Whether to shuffle
        random_seed: Seed for reproducibility
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Split dataset
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Dataset split:")
    print(f"  Training: {len(train_dataset)} samples")
    print(f"  Validation: {len(val_dataset)} samples")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader
