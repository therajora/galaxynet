"""
Plots and visualizations for domain shift analysis.

Implements various types of visualizations to compare
results between SDSS and S-PLUS domains.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Style configuration
plt.style.use('default')
sns.set_palette("husl")


def plot_domain_comparison(
    sdss_results: Dict[str, Any],
    splus_results: Dict[str, Any],
    save_path: Path
):
    """
    Plot general comparison between domains.
    
    Args:
        sdss_results: Results in SDSS domain
        splus_results: Results in S-PLUS domain
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Domain Comparison: SDSS vs S-PLUS', fontsize=16, fontweight='bold')
    
    # 1. Performance Metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    sdss_metrics = [sdss_results['metrics'][m] for m in metrics]
    splus_metrics = [splus_results['metrics'][m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, sdss_metrics, width, label='SDSS', alpha=0.8)
    axes[0, 0].bar(x + width/2, splus_metrics, width, label='S-PLUS', alpha=0.8)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Performance Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Probability Distribution
    sdss_proba = np.array(sdss_results['predictions']['y_proba'])
    splus_proba = np.array(splus_results['predictions']['y_proba'])
    
    axes[0, 1].hist(sdss_proba, bins=30, alpha=0.7, label='SDSS', density=True)
    axes[0, 1].hist(splus_proba, bins=30, alpha=0.7, label='S-PLUS', density=True)
    axes[0, 1].set_xlabel('Predicted Probability')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Probability Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Class Distribution
    sdss_class_dist = np.bincount(sdss_results['predictions']['y_pred'], minlength=2)
    splus_class_dist = np.bincount(splus_results['predictions']['y_pred'], minlength=2)
    
    class_names = ['Regular', 'Peculiar']
    x = np.arange(len(class_names))
    
    axes[1, 0].bar(x - width/2, sdss_class_dist, width, label='SDSS', alpha=0.8)
    axes[1, 0].bar(x + width/2, splus_class_dist, width, label='S-PLUS', alpha=0.8)
    axes[1, 0].set_xlabel('Classes')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Predicted Class Distribution')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. ROC Curves
    sdss_fpr, sdss_tpr, _ = roc_curve(
        sdss_results['predictions']['y_true'],
        sdss_results['predictions']['y_proba']
    )
    splus_fpr, splus_tpr, _ = roc_curve(
        splus_results['predictions']['y_true'],
        splus_results['predictions']['y_proba']
    )
    
    sdss_auc = auc(sdss_fpr, sdss_tpr)
    splus_auc = auc(splus_fpr, splus_tpr)
    
    axes[1, 1].plot(sdss_fpr, sdss_tpr, label=f'SDSS (AUC = {sdss_auc:.3f})', linewidth=2)
    axes[1, 1].plot(splus_fpr, splus_tpr, label=f'S-PLUS (AUC = {splus_auc:.3f})', linewidth=2)
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].set_title('ROC Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_distributions(
    sdss_features: np.ndarray,
    splus_features: np.ndarray,
    save_path: Path,
    n_features: int = 6
):
    """
    Plot feature distributions between domains.
    
    Args:
        sdss_features: SDSS features
        splus_features: S-PLUS features
        save_path: Path to save plot
        n_features: Number of features to plot
    """
    # Select features with highest variance
    sdss_var = np.var(sdss_features, axis=0)
    splus_var = np.var(splus_features, axis=0)
    combined_var = sdss_var + splus_var
    top_features = np.argsort(combined_var)[-n_features:]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Feature Distributions: SDSS vs S-PLUS', fontsize=16, fontweight='bold')
    
    for i, feature_idx in enumerate(top_features):
        if i >= len(axes):
            break
            
        sdss_feature = sdss_features[:, feature_idx]
        splus_feature = splus_features[:, feature_idx]
        
        axes[i].hist(sdss_feature, bins=30, alpha=0.7, label='SDSS', density=True)
        axes[i].hist(splus_feature, bins=30, alpha=0.7, label='S-PLUS', density=True)
        axes[i].set_xlabel(f'Feature {feature_idx}')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'Feature {feature_idx} (Var: {combined_var[feature_idx]:.2f})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(top_features), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_performance_comparison(
    sdss_metrics: Dict[str, float],
    splus_metrics: Dict[str, float],
    save_path: Path
):
    """
    Plot detailed performance comparison.
    
    Args:
        sdss_metrics: SDSS metrics
        splus_metrics: S-PLUS metrics
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Detailed Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Side-by-side metrics
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
    sdss_values = [sdss_metrics[m] for m in metrics]
    splus_values = [splus_metrics[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar(x - width/2, sdss_values, width, label='SDSS', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, splus_values, width, label='S-PLUS', alpha=0.8)
    
    # Add values on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    axes[0].set_xlabel('Metrics')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Performance Metrics')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=45)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.1)
    
    # 2. Performance drop
    performance_drops = [sdss_values[i] - splus_values[i] for i in range(len(metrics))]
    colors = ['red' if drop > 0 else 'green' for drop in performance_drops]
    
    bars3 = axes[1].bar(metrics, performance_drops, color=colors, alpha=0.7)
    
    # Add values on bars
    for bar, drop in zip(bars3, performance_drops):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2.,
                    height + (0.01 if height >= 0 else -0.01),
                    f'{drop:.3f}', ha='center',
                    va='bottom' if height >= 0 else 'top', fontsize=9)
    
    axes[1].set_xlabel('Metrics')
    axes[1].set_ylabel('Performance Drop (SDSS - S-PLUS)')
    axes[1].set_title('Performance Drop in Domain Shift')
    axes[1].set_xticklabels(metrics, rotation=45)
    axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_comparison(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List],
    class_names: List[str],
    save_path: Path
):
    """
    Plot confusion matrices side by side.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        class_names: Class names
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Confusion Matrices: SDSS vs S-PLUS', fontsize=16, fontweight='bold')
    
    # SDSS confusion matrix
    cm_sdss = confusion_matrix(sdss_predictions['y_true'], sdss_predictions['y_pred'])
    sns.heatmap(cm_sdss, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('SDSS')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # S-PLUS confusion matrix
    cm_splus = confusion_matrix(splus_predictions['y_true'], splus_predictions['y_pred'])
    sns.heatmap(cm_splus, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('S-PLUS')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_distributions(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List],
    save_path: Path
):
    """
    Plot prediction confidence distributions.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confidence Analysis: SDSS vs S-PLUS', fontsize=16, fontweight='bold')
    
    sdss_proba = np.array(sdss_predictions['y_proba'])
    splus_proba = np.array(splus_predictions['y_proba'])
    
    # 1. Probability distribution
    axes[0, 0].hist(sdss_proba, bins=30, alpha=0.7, label='SDSS', density=True)
    axes[0, 0].hist(splus_proba, bins=30, alpha=0.7, label='S-PLUS', density=True)
    axes[0, 0].set_xlabel('Predicted Probability')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Probability Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Confidence (distance from 0.5 threshold)
    sdss_confidence = np.abs(sdss_proba - 0.5)
    splus_confidence = np.abs(splus_proba - 0.5)
    
    axes[0, 1].hist(sdss_confidence, bins=30, alpha=0.7, label='SDSS', density=True)
    axes[0, 1].hist(splus_confidence, bins=30, alpha=0.7, label='S-PLUS', density=True)
    axes[0, 1].set_xlabel('Confidence (|prob - 0.5|)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Confidence box plot
    confidence_data = [sdss_confidence, splus_confidence]
    axes[1, 0].boxplot(confidence_data, labels=['SDSS', 'S-PLUS'])
    axes[1, 0].set_ylabel('Confidence')
    axes[1, 0].set_title('Confidence Box Plot')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. High confidence predictions proportion
    thresholds = np.linspace(0.1, 0.4, 20)
    sdss_high_conf = [np.mean(sdss_confidence > t) for t in thresholds]
    splus_high_conf = [np.mean(splus_confidence > t) for t in thresholds]
    
    axes[1, 1].plot(thresholds, sdss_high_conf, label='SDSS', marker='o')
    axes[1, 1].plot(thresholds, splus_high_conf, label='S-PLUS', marker='s')
    axes[1, 1].set_xlabel('Confidence Threshold')
    axes[1, 1].set_ylabel('High Confidence Predictions Proportion')
    axes[1, 1].set_title('High Confidence Proportion by Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_calibration_curves(
    sdss_predictions: Dict[str, List],
    splus_predictions: Dict[str, List],
    save_path: Path,
    n_bins: int = 10
):
    """
    Plot calibration curves.
    
    Args:
        sdss_predictions: SDSS predictions
        splus_predictions: S-PLUS predictions
        save_path: Path to save plot
        n_bins: Number of bins for calibration
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Calibration Curves: SDSS vs S-PLUS', fontsize=16, fontweight='bold')
    
    def plot_calibration_curve(y_true, y_proba, ax, title, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_centers = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                
                bin_centers.append(avg_confidence_in_bin)
                bin_accuracies.append(accuracy_in_bin)
                bin_counts.append(prop_in_bin)
        
        # Plot
        ax.plot(bin_centers, bin_accuracies, 'o-', label='Calibration', linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add bin sizes
        for i, (center, acc, count) in enumerate(zip(bin_centers, bin_accuracies, bin_counts)):
            ax.annotate(f'n={count}', (center, acc), xytext=(5, 5),
                       textcoords='offset points', fontsize=8, alpha=0.7)
    
    # SDSS
    plot_calibration_curve(
        np.array(sdss_predictions['y_true']), 
        np.array(sdss_predictions['y_proba']), 
        axes[0], 'SDSS', n_bins
    )
    
    # S-PLUS
    plot_calibration_curve(
        np.array(splus_predictions['y_true']), 
        np.array(splus_predictions['y_proba']), 
        axes[1], 'S-PLUS', n_bins
    )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_space_visualization(
    sdss_features: np.ndarray,
    splus_features: np.ndarray,
    save_path: Path,
    method: str = 'pca'
):
    """
    Visualize feature space using dimensionality reduction.
    
    Args:
        sdss_features: SDSS features
        splus_features: S-PLUS features
        save_path: Path to save plot
        method: Reduction method ('pca' or 'tsne')
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'Feature Space Visualization ({method.upper()})', fontsize=16, fontweight='bold')
    
    # Combine features
    all_features = np.vstack([sdss_features, splus_features])
    n_sdss = len(sdss_features)
    n_splus = len(splus_features)
    
    # Reduce dimensionality
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        reduced_features = reducer.fit_transform(all_features)
        explained_var = reducer.explained_variance_ratio_
    elif method.lower() == 'tsne':
        # Sample for t-SNE (very slow with many data points)
        n_samples = min(1000, len(all_features))
        indices = np.random.choice(len(all_features), n_samples, replace=False)
        sampled_features = all_features[indices]
        
        reducer = TSNE(n_components=2, random_state=42)
        reduced_features = reducer.fit_transform(sampled_features)
        
        # Adjust indices
        sdss_indices = indices[indices < n_sdss]
        splus_indices = indices[indices >= n_sdss] - n_sdss
        explained_var = None
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    # Separate reduced features
    if method.lower() == 'pca':
        sdss_reduced = reduced_features[:n_sdss]
        splus_reduced = reduced_features[n_sdss:]
    else:  # t-SNE
        sdss_reduced = reduced_features[sdss_indices]
        splus_reduced = reduced_features[splus_indices]
    
    # Plot 1: Scatter plot
    axes[0].scatter(sdss_reduced[:, 0], sdss_reduced[:, 1],
                   alpha=0.6, label='SDSS', s=20)
    axes[0].scatter(splus_reduced[:, 0], splus_reduced[:, 1],
                   alpha=0.6, label='S-PLUS', s=20)
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    axes[0].set_title('Distribution in Reduced Space')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if explained_var is not None:
        axes[0].set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
        axes[0].set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    
    # Plot 2: Density
    from scipy.stats import gaussian_kde
    
    # Calculate density for SDSS
    sdss_density = gaussian_kde(sdss_reduced.T)
    splus_density = gaussian_kde(splus_reduced.T)
    
    # Create grid
    x_min, x_max = reduced_features[:, 0].min(), reduced_features[:, 0].max()
    y_min, y_max = reduced_features[:, 1].min(), reduced_features[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min -= 0.1 * x_range
    x_max += 0.1 * x_range
    y_min -= 0.1 * y_range
    y_max += 0.1 * y_range
    
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    sdss_density_values = sdss_density(positions).reshape(xx.shape)
    splus_density_values = splus_density(positions).reshape(xx.shape)
    
    # Plot SDSS density
    im1 = axes[1].contourf(xx, yy, sdss_density_values, alpha=0.5, cmap='Blues')
    axes[1].contour(xx, yy, sdss_density_values, colors='blue', alpha=0.7, linewidths=1)
    
    # Plot S-PLUS density
    im2 = axes[1].contourf(xx, yy, splus_density_values, alpha=0.5, cmap='Oranges')
    axes[1].contour(xx, yy, splus_density_values, colors='orange', alpha=0.7, linewidths=1)
    
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    axes[1].set_title('Distribution Density')
    axes[1].grid(True, alpha=0.3)
    
    if explained_var is not None:
        axes[1].set_xlabel(f'PC1 ({explained_var[0]:.1%} var)')
        axes[1].set_ylabel(f'PC2 ({explained_var[1]:.1%} var)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
