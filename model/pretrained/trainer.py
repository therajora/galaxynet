"""
Unified trainer for pre-trained galaxy classification models.

Includes metrics, logging, early stopping and result visualization.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

from .config_manager import (
    ModelConfig, DataConfig, TrainingConfig, OptimizerConfig, 
    LossConfig, ExperimentConfig, LoggingConfig
)


class PretrainedTrainer:
    """
    Trainer for pre-trained models with advanced features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        data_config: DataConfig,
        training_config: TrainingConfig,
        optimizer_config: OptimizerConfig,
        loss_config: LossConfig,
        experiment_config: ExperimentConfig,
        logging_config: LoggingConfig,
        device: torch.device
    ):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            model_config: Model configurations
            data_config: Data configurations
            training_config: Training configurations
            optimizer_config: Optimizer configurations
            loss_config: Loss function configurations
            experiment_config: Experiment configurations
            logging_config: Logging configurations
            device: Device for training
        """
        self.model = model
        self.device = device
        self.configs = {
            'model': model_config,
            'data': data_config,
            'training': training_config,
            'optimizer': optimizer_config,
            'loss': loss_config,
            'experiment': experiment_config,
            'logging': logging_config
        }
        
        # Move model to device
        self.model.to(device)
        
        # Training configurations
        self.num_epochs = training_config.num_epochs
        self.early_stopping = training_config.early_stopping
        self.patience = training_config.patience
        self.save_best = training_config.save_best
        
        # Mixed precision
        self.use_amp = experiment_config.mixed_precision
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': [],
            'lr': []
        }
        
        # Early stopping
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Create directories
        self._create_directories()
        
        # Configure optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        print(f"Trainer initialized - Device: {device}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Save directory: {logging_config.save_dir}")
    
    def _create_directories(self):
        """Create necessary directories."""
        os.makedirs(self.configs['logging'].log_dir, exist_ok=True)
        os.makedirs(self.configs['logging'].save_dir, exist_ok=True)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_config = self.configs['optimizer']
        
        if opt_config.name.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
        elif opt_config.name.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                betas=opt_config.betas
            )
        elif opt_config.name.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=opt_config.lr,
                weight_decay=opt_config.weight_decay,
                momentum=opt_config.momentum
            )
        else:
            raise ValueError(f"Optimizer '{opt_config.name}' not supported")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create scheduler based on configuration."""
        train_config = self.configs['training']
        
        if train_config.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=train_config.scheduler_params.get('step_size', 7),
                gamma=train_config.scheduler_params.get('gamma', 0.1)
            )
        elif train_config.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config.scheduler_params.get('T_max', self.num_epochs),
                eta_min=train_config.scheduler_params.get('eta_min', 0)
            )
        elif train_config.scheduler == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=train_config.scheduler_params.get('factor', 0.5),
                patience=train_config.scheduler_params.get('patience', 3),
                verbose=True
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.configs['loss']
        
        if loss_config.name.lower() == 'crossentropy':
            if loss_config.use_class_weights:
                # Weights will be set during training
                return nn.CrossEntropyLoss(label_smoothing=loss_config.label_smoothing)
            else:
                return nn.CrossEntropyLoss(label_smoothing=loss_config.label_smoothing)
        else:
            raise ValueError(f"Loss function '{loss_config.name}' not supported")
    
    def train_epoch(self, train_loader: DataLoader, class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader
            class_weights: Weights for class balancing
            
        Returns:
            Tuple with (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Update class weights if necessary
        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        
        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader, class_weights: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation DataLoader
            class_weights: Weights for class balancing
            
        Returns:
            Tuple with (average loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Update class weights if necessary
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = self.criterion
        
        val_pbar = tqdm(val_loader, desc="Validation", leave=False)
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_pbar):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, List[float]]:
        """
        Execute complete training.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            class_weights: Weights for class balancing
            
        Returns:
            Training history
        """
        print(f"\nStarting training for {self.num_epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader, class_weights)
            
            # Validate one epoch
            val_loss, val_acc = self.validate_epoch(val_loader, class_weights)
            
            # Update scheduler
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_acc)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epochs'].append(epoch + 1)
            self.history['lr'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Epoch log
            print(f"Epoch {epoch+1}/{self.num_epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
            
            # Save best model
            if self.save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                self.save_model("best_model.pth")
                print(f"New best model saved! (Acc: {val_acc:.2f}%)")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.early_stopping and self.patience_counter >= self.patience:
                print(f"Early stopping activated after {self.patience} epochs without improvement")
                break
        
        training_time = time.time() - start_time
        
        # Save final model
        self.save_model("final_model.pth")
        
        # Save history
        self.save_training_history()
        
        # Plot results
        if self.configs['logging'].plot_training:
            self.plot_training_history()
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (Epoch {self.best_epoch})")
        print(f"Total training time: {training_time:.2f} seconds")
        
        return self.history
    
    def save_model(self, filename: str):
        """Save the model."""
        save_path = os.path.join(self.configs['logging'].save_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'configs': {k: v.__dict__ for k, v in self.configs.items()},
            'best_val_acc': self.best_val_acc,
            'best_epoch': self.best_epoch
        }, save_path)
    
    def save_training_history(self):
        """Save training history."""
        history_path = os.path.join(self.configs['logging'].save_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)
    
    def plot_training_history(self):
        """Plot training history."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            epochs = self.history['epochs']
            
            # Loss
            ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training')
            ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation')
            ax1.set_title('Loss per Epoch')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Accuracy
            ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training')
            ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation')
            ax2.set_title('Accuracy per Epoch')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.legend()
            ax2.grid(True)
            
            # Learning Rate
            ax3.plot(epochs, self.history['lr'], 'g-', label='Learning Rate')
            ax3.set_title('Learning Rate per Epoch')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.legend()
            ax3.grid(True)
            ax3.set_yscale('log')
            
            # Best epoch marker
            ax4.axvline(x=self.best_epoch, color='red', linestyle='--', 
                       label=f'Best Epoch ({self.best_epoch})')
            ax4.plot(epochs, self.history['val_acc'], 'r-', label='Validation')
            ax4.set_title('Best Epoch')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Validation Accuracy (%)')
            ax4.legend()
            ax4.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plot_path = os.path.join(self.configs['logging'].save_dir, "training_history.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {plot_path}")
            
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Plot will not be generated.")
        except Exception as e:
            print(f"Error generating plot: {e}")
