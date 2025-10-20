"""
Metrics Module for Classification Model Validation.

Implements all specified metrics:
- Confusion Matrix
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve
- AUC (Area Under the Curve)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, 
    f1_score, roc_curve, auc, roc_auc_score
)
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path


class ClassificationMetrics:
    """
    Class for calculating and storing classification metrics.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            class_names: Class names (default: ['Regular', 'Peculiar'])
        """
        self.class_names = class_names or ['Regular', 'Peculiar']
        self.num_classes = len(self.class_names)
        
        # Calculated metrics
        self.confusion_matrix = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1_score = None
        self.roc_auc = None
        self.roc_curve_data = None
        
    def calculate_all_metrics(
        self, 
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_prob: Union[np.ndarray, torch.Tensor, List] = None
    ) -> Dict[str, float]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predictions (classes)
            y_prob: Prediction probabilities (optional, for ROC/AUC)
            
        Returns:
            Dictionary with all metrics
        """
        # Convert to numpy if necessary
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        
        # Calculate basic metrics
        self.confusion_matrix = calculate_confusion_matrix(y_true, y_pred)
        self.accuracy = calculate_accuracy(y_true, y_pred)
        self.precision = calculate_precision(y_true, y_pred, average='weighted')
        self.recall = calculate_recall(y_true, y_pred, average='weighted')
        self.f1_score = calculate_f1_score(y_true, y_pred, average='weighted')
        
        # Calculate ROC/AUC if probabilities provided
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
            if self.num_classes == 2:
                # For binary classification, use positive class probability
                if y_prob.ndim > 1:
                    y_prob = y_prob[:, 1]  # Probability of class 1 (Peculiar)
                self.roc_auc = calculate_roc_auc(y_true, y_prob)
                self.roc_curve_data = self._calculate_roc_curve(y_true, y_prob)
            else:
                # For multiclass, calculate ROC/AUC for each class
                self.roc_auc = {}
                self.roc_curve_data = {}
                for i in range(self.num_classes):
                    y_true_binary = (y_true == i).astype(int)
                    if y_prob.ndim > 1:
                        y_prob_class = y_prob[:, i]
                    else:
                        y_prob_class = y_prob
                    self.roc_auc[self.class_names[i]] = calculate_roc_auc(y_true_binary, y_prob_class)
                    self.roc_curve_data[self.class_names[i]] = self._calculate_roc_curve(y_true_binary, y_prob_class)
        
        return self.get_metrics_dict()
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        return data
    
    def _calculate_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
    
    def get_metrics_dict(self) -> Dict[str, Union[float, Dict]]:
        """
        Return dictionary with all calculated metrics.
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc
        }
        
        # Remove None values
        return {k: v for k, v in metrics.items() if v is not None}
    
    def save_metrics(self, filepath: Union[str, Path]):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: File path
        """
        metrics_dict = self.get_metrics_dict()
        metrics_dict['class_names'] = self.class_names
        metrics_dict['num_classes'] = self.num_classes
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, indent=4)
    
    def print_summary(self):
        """Print metrics summary."""
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS")
        print("="*60)
        
        if self.accuracy is not None:
            print(f"Accuracy: {self.accuracy:.4f}")
        if self.precision is not None:
            print(f"Precision: {self.precision:.4f}")
        if self.recall is not None:
            print(f"Recall: {self.recall:.4f}")
        if self.f1_score is not None:
            print(f"F1-Score: {self.f1_score:.4f}")
        if self.roc_auc is not None:
            if isinstance(self.roc_auc, dict):
                print("ROC-AUC by class:")
                for class_name, auc_value in self.roc_auc.items():
                    print(f"  {class_name}: {auc_value:.4f}")
            else:
                print(f"ROC-AUC: {self.roc_auc:.4f}")
        
        print("="*60)


def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        
    Returns:
        Confusion matrix
    """
    return confusion_matrix(y_true, y_pred)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        
    Returns:
        Accuracy
    """
    return accuracy_score(y_true, y_pred)


def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculate precision.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        average: Average type ('weighted', 'macro', 'micro')
        
    Returns:
        Precision
    """
    return precision_score(y_true, y_pred, average=average, zero_division=0)


def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculate recall.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        average: Average type ('weighted', 'macro', 'micro')
        
    Returns:
        Recall
    """
    return recall_score(y_true, y_pred, average=average, zero_division=0)


def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> float:
    """
    Calculate F1-Score.
    
    Args:
        y_true: True labels
        y_pred: Predictions
        average: Average type ('weighted', 'macro', 'micro')
        
    Returns:
        F1-Score
    """
    return f1_score(y_true, y_pred, average=average, zero_division=0)


def calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate ROC curve AUC.
    
    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        
    Returns:
        AUC
    """
    return roc_auc_score(y_true, y_prob)


def plot_confusion_matrix(
    cm: np.ndarray, 
    class_names: List[str] = None,
    title: str = "Confusion Matrix",
    save_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: Class names
        title: Plot title
        save_path: Path to save (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize matrix to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_title(title)
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    title: str = "ROC Curve",
    save_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        auc_score: AUC score
        title: Plot title
        save_path: Path to save (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {auc_score:.2f})')
    
    # Plot diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_multiclass_roc_curves(
    roc_data: Dict[str, Dict],
    title: str = "ROC Curves - Multiclass",
    save_path: Union[str, Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiclass classification.
    
    Args:
        roc_data: ROC data for each class
        title: Plot title
        save_path: Path to save (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))
    
    for i, (class_name, data) in enumerate(roc_data.items()):
        fpr = data['fpr']
        tpr = data['tpr']
        auc_score = data.get('auc', 0)
        
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{class_name} (AUC = {auc_score:.2f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
