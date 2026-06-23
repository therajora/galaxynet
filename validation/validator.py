"""
Galaxy Classification Model Validator.

Implements complete validation of pre-trained models with:
- Loading trained models
- Validation on test dataset
- Complete metrics calculation
- Report and visualization generation
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt

# Add root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.pretrained.model_factory import create_pretrained_model, get_model_info
from model.pretrained.dataset import GalaxyPretrainedDataset, get_imagenet_transforms
from validation.metrics import ClassificationMetrics, plot_confusion_matrix, plot_roc_curve
from validation.validation_persistence import save_validation_outputs
from validation.validation_runner import run_validation_batches


class ModelValidator:
    """
    Class for galaxy classification model validation.
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: Union[str, Path],
        device: torch.device = None,
        class_names: List[str] = None
    ):
        """
        Initialize the validator.
        
        Args:
            model_name: Model name (e.g., 'resnet50', 'efficientnet_b0')
            model_path: Path to trained model file
            device: Device for validation (default: auto)
            class_names: Class names (default: ['Regular', 'Peculiar'])
        """
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.class_names = class_names or ['Regular', 'Peculiar']
        self.num_classes = len(self.class_names)
        
        # Configure device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Load model
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Validator initialized:")
        print(f"   Model: {self.model_name}")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.class_names}")
    
    def _load_model(self) -> nn.Module:
        """
        Load the trained model.
        
        Returns:
            Loaded PyTorch model
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Create model with correct architecture
        model = create_pretrained_model(
            model_name=self.model_name,
            num_classes=self.num_classes,
            pretrained=False  # Don't load pre-trained weights, just the architecture
        )
        
        # Load trained weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # If checkpoint contains 'model_state_dict', use it
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Assume file contains state_dict directly
            model.load_state_dict(checkpoint)
        
        return model
    
    def validate_dataset(
        self,
        test_loader: DataLoader,
        save_results: bool = True,
        output_dir: Union[str, Path] = None
    ) -> Dict:
        """
        Validate the model on a test dataset.
        
        Args:
            test_loader: DataLoader with test data
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Dictionary with metrics and results
        """
        print(f"\nStarting validation of model {self.model_name}...")

        raw_outputs = run_validation_batches(
            model=self.model,
            test_loader=test_loader,
            device=self.device,
        )
        y_true = raw_outputs["y_true"]
        y_pred = raw_outputs["y_pred"]
        y_prob = raw_outputs["y_prob"]
        
        print(f"Validation completed: {len(y_true)} samples")
        
        # Calculate metrics
        metrics_calculator = ClassificationMetrics(self.class_names)
        metrics = metrics_calculator.calculate_all_metrics(y_true, y_pred, y_prob)
        
        # Print summary
        metrics_calculator.print_summary()
        
        # Save results if requested
        if save_results:
            if output_dir is None:
                output_dir = Path("validation_results") / self.model_name
            else:
                output_dir = Path(output_dir)

            save_validation_outputs(
                output_dir=output_dir,
                model_name=self.model_name,
                model_path=str(self.model_path),
                class_names=self.class_names,
                y_true=y_true.tolist(),
                y_pred=y_pred.tolist(),
                y_prob=y_prob.tolist(),
                metrics_calculator=metrics_calculator,
                generate_visualizations=lambda **kwargs: self._generate_visualizations(
                    kwargs["metrics_calculator"],
                    kwargs["output_dir"],
                ),
            )

            print(f"Results saved to: {output_dir}")
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_true,
            'metrics_calculator': metrics_calculator
        }
    
    def _generate_visualizations(
        self, 
        metrics_calculator: ClassificationMetrics,
        output_dir: Path
    ):
        """
        Generate metric visualizations.
        
        Args:
            metrics_calculator: Metrics calculator
            output_dir: Output directory
        """
        print("Generating visualizations...")
        
        # Confusion matrix
        if metrics_calculator.confusion_matrix is not None:
            fig = plot_confusion_matrix(
                metrics_calculator.confusion_matrix,
                class_names=self.class_names,
                title=f"Confusion Matrix - {self.model_name.upper()}",
                save_path=output_dir / "confusion_matrix.png"
            )
            plt.close(fig)
        
        # ROC curve
        if metrics_calculator.roc_curve_data is not None:
            if self.num_classes == 2:
                # Binary classification
                roc_data = metrics_calculator.roc_curve_data
                fig = plot_roc_curve(
                    roc_data['fpr'],
                    roc_data['tpr'],
                    metrics_calculator.roc_auc,
                    title=f"ROC Curve - {self.model_name.upper()}",
                    save_path=output_dir / "roc_curve.png"
                )
                plt.close(fig)
            else:
                # Multiclass classification
                from validation.metrics import plot_multiclass_roc_curves
                fig = plot_multiclass_roc_curves(
                    metrics_calculator.roc_curve_data,
                    title=f"ROC Curves - {self.model_name.upper()}",
                    save_path=output_dir / "roc_curves.png"
                )
                plt.close(fig)
    
    def validate_single_image(
        self, 
        image_path: Union[str, Path],
        transform=None
    ) -> Dict:
        """
        Validate a single image.
        
        Args:
            image_path: Image path
            transform: Transformations to apply (default: ImageNet transforms)
            
        Returns:
            Dictionary with prediction and probabilities
        """
        from PIL import Image
        
        if transform is None:
            transform = get_imagenet_transforms(224, is_train=False)
        
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        # Prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)
        
        # Result
        pred_class = self.class_names[prediction.item()]
        confidence = probabilities[0][prediction.item()].item()
        
        return {
            'predicted_class': pred_class,
            'confidence': confidence,
            'probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(self.class_names, probabilities[0])
            }
        }
    
    def get_model_info(self) -> Dict:
        """
        Return model information.
        
        Returns:
            Dictionary with model information
        """
        from model.pretrained.model_factory import count_parameters
        
        params = count_parameters(self.model)
        model_info = get_model_info(self.model_name)
        
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'description': model_info['description'],
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'parameters': params,
            'device': str(self.device)
        }


def validate_model_from_checkpoint(
    model_name: str,
    checkpoint_path: Union[str, Path],
    test_loader: DataLoader,
    class_names: List[str] = None,
    device: torch.device = None,
    save_results: bool = True,
    output_dir: Union[str, Path] = None
) -> Dict:
    """
    Utility function to validate a model from a checkpoint.
    
    Args:
        model_name: Model name
        checkpoint_path: Checkpoint path
        test_loader: Test DataLoader
        class_names: Class names
        device: Device
        save_results: Whether to save results
        output_dir: Output directory
        
    Returns:
        Validation results
    """
    validator = ModelValidator(
        model_name=model_name,
        model_path=checkpoint_path,
        device=device,
        class_names=class_names
    )
    
    return validator.validate_dataset(
        test_loader=test_loader,
        save_results=save_results,
        output_dir=output_dir
    )
