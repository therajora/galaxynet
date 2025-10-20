#!/usr/bin/env python3
"""
Plot Generator for Model Validation.

This module creates visualizations for validation results
of galaxy classification models.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import json
from datetime import datetime

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Matplotlib configuration
plt.style.use('default')
sns.set_palette("husl")


class ValidationPlotGenerator:
    """
    Plot generator for model validation.
    """
    
    def __init__(self, output_dir: str = "validation/visualization/plots"):
        """
        Initialize the plot generator.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir) / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(parents=True, exist_ok=True)
        
        # Style configurations
        self.colors = {
            'efficientnet': '#FF6B6B',
            'resnet': '#4ECDC4', 
            'vit': '#45B7D1',
            'resnext': '#96CEB4',
            'from_scratch': '#FFA07A',
            'accuracy': '#2E8B57',
            'precision': '#FF6347',
            'recall': '#4169E1',
            'f1': '#9932CC'
        }
        
        # Plot configurations
        self.figsize = (12, 8)
        self.dpi = 300
        
    def load_results(self, results_path: str) -> List[Dict]:
        """
        Load results from JSON file.
        
        Args:
            results_path: Path to results file
            
        Returns:
            List of results
        """
        with open(results_path, 'r') as f:
            results = json.load(f)
        return results
    
    def categorize_models(self, results: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Categorize models by family.
        
        Args:
            results: List of results
            
        Returns:
            Dictionary with categorized models
        """
        categories = {
            'EfficientNet': [],
            'ResNet': [],
            'ViT': [],
            'ResNeXt': [],
            'From Scratch': []
        }
        
        for result in results:
            model_name = result.get('model_name', '')
            
            if 'efficientnet' in model_name.lower():
                categories['EfficientNet'].append(result)
            elif 'resnet' in model_name.lower() and 'resnext' not in model_name.lower():
                categories['ResNet'].append(result)
            elif 'vit' in model_name.lower():
                categories['ViT'].append(result)
            elif 'resnext' in model_name.lower():
                categories['ResNeXt'].append(result)
            elif 'galaxynet' in model_name.lower() or 'from_scratch' in model_name.lower():
                categories['From Scratch'].append(result)
        
        return categories
    
    def create_accuracy_comparison_plot(self, results: List[Dict]) -> Path:
        """
        Create accuracy comparison plot.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Prepare data
        model_names = [r.get('model_name', 'Unknown') for r in results]
        accuracies = [r.get('accuracy', 0.0) for r in results]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create bar chart
        bars = ax.bar(range(len(model_names)), accuracies, 
                     color=[self._get_model_color(name) for name in model_names],
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Chart configuration
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Comparison Between Models', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add reference line
        avg_accuracy = np.mean(accuracies)
        ax.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7, 
                  label=f'Average: {avg_accuracy:.3f}')
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "plots" / "accuracy_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Accuracy plot saved to: {output_path}")
        return output_path
    
    def create_metrics_radar_chart(self, results: List[Dict]) -> Path:
        """
        Create radar chart with multiple metrics.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Colors for each model
        colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
        
        # Plot each model
        for i, result in enumerate(results):
            model_name = result.get('model_name', f'Model {i+1}')
            values = [result.get(metric, 0.0) for metric in metrics]
            
            # Close the polygon
            values += values[:1]
            angles = np.linspace(0, 2 * np.pi, len(metric_labels), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Chart configuration
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1.0)
        ax.set_title('Metrics Comparison - Radar Chart', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "plots" / "metrics_radar_chart.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Radar chart saved to: {output_path}")
        return output_path
    
    def create_family_comparison_plot(self, results: List[Dict]) -> Path:
        """
        Create comparison plot by model family.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Categorize models
        categories = self.categorize_models(results)
        
        # Prepare data
        family_names = []
        family_accuracies = []
        family_colors = []
        
        color_map = {
            'EfficientNet': self.colors['efficientnet'],
            'ResNet': self.colors['resnet'],
            'ViT': self.colors['vit'],
            'ResNeXt': self.colors['resnext'],
            'From Scratch': self.colors['from_scratch']
        }
        
        for family, models in categories.items():
            if models:
                family_names.append(family)
                accuracies = [m.get('accuracy', 0.0) for m in models]
                family_accuracies.append(accuracies)
                family_colors.append(color_map.get(family, '#CCCCCC'))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Box plot
        bp1 = ax1.boxplot(family_accuracies, labels=family_names, patch_artist=True)
        for patch, color in zip(bp1['boxes'], family_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Accuracy Distribution by Family', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean by family
        mean_accuracies = [np.mean(accs) for accs in family_accuracies]
        bars = ax2.bar(family_names, mean_accuracies, color=family_colors, alpha=0.8)
        
        ax2.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Mean Accuracy by Family', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, acc in zip(bars, mean_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "plots" / "family_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Family comparison plot saved to: {output_path}")
        return output_path
    
    def create_performance_heatmap(self, results: List[Dict]) -> Path:
        """
        Create performance heatmap.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        model_names = [r.get('model_name', f'Model {i+1}') for i, r in enumerate(results)]
        
        # Create data matrix
        data_matrix = []
        for result in results:
            row = [result.get(metric, 0.0) for metric in metrics]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Chart configuration
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        
        # Add values in cells
        for i in range(len(model_names)):
            for j in range(len(metric_labels)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Metric Value', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "plots" / "performance_heatmap.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Performance heatmap saved to: {output_path}")
        return output_path
    
    def create_confusion_matrix_grid(self, results: List[Dict]) -> Path:
        """
        Create confusion matrix grid.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Filter results that have confusion matrix
        results_with_cm = [r for r in results if 'confusion_matrix' in r]
        
        if not results_with_cm:
            print("No results with confusion matrix found")
            return None
        
        # Calculate grid size
        n_models = len(results_with_cm)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each confusion matrix
        for i, result in enumerate(results_with_cm):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            cm = np.array(result['confusion_matrix'])
            model_name = result.get('model_name', f'Model {i+1}')
            
            # Plot matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Regular', 'Peculiar'],
                       yticklabels=['Regular', 'Peculiar'])
            
            ax.set_title(f'{model_name}\nAccuracy: {result.get("accuracy", 0.0):.3f}', 
                        fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
        
        # Remove empty axes
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        # Adjust layout
        plt.suptitle('Confusion Matrices by Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "plots" / "confusion_matrix_grid.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix grid saved to: {output_path}")
        return output_path
    
    def create_roc_curves_comparison(self, results: List[Dict]) -> Path:
        """
        Create ROC curves comparison.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Filter results that have ROC data
        results_with_roc = [r for r in results if 'roc_curve' in r and r['roc_curve']]
        
        if not results_with_roc:
            print("No results with ROC curve found")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot each ROC curve
        colors = plt.cm.Set3(np.linspace(0, 1, len(results_with_roc)))
        
        for i, result in enumerate(results_with_roc):
            roc_data = result['roc_curve']
            model_name = result.get('model_name', f'Model {i+1}')
            auc = result.get('auc', 0.0)
            
            fpr = roc_data['fpr']
            tpr = roc_data['tpr']
            
            ax.plot(fpr, tpr, color=colors[i], linewidth=2,
                   label=f'{model_name} (AUC = {auc:.3f})')
        
        # Diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        # Chart configuration
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "plots" / "roc_curves_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves comparison saved to: {output_path}")
        return output_path
    
    def _get_model_color(self, model_name: str) -> str:
        """
        Return color based on model name.
        
        Args:
            model_name: Model name
            
        Returns:
            Color in hex format
        """
        model_name_lower = model_name.lower()
        
        if 'efficientnet' in model_name_lower:
            return self.colors['efficientnet']
        elif 'resnet' in model_name_lower and 'resnext' not in model_name_lower:
            return self.colors['resnet']
        elif 'vit' in model_name_lower:
            return self.colors['vit']
        elif 'resnext' in model_name_lower:
            return self.colors['resnext']
        elif 'galaxynet' in model_name_lower or 'from_scratch' in model_name_lower:
            return self.colors['from_scratch']
        else:
            return '#CCCCCC'
    
    def generate_all_plots(self, results_path: str) -> List[Path]:
        """
        Generate all validation plots.
        
        Args:
            results_path: Path to results file
            
        Returns:
            List of paths to generated plots
        """
        print("Generating validation visualizations...")
        
        # Load results
        results = self.load_results(results_path)
        
        if not results:
            print("No results found!")
            return []
        
        generated_plots = []
        
        # Generate each type of plot
        try:
            plot1 = self.create_accuracy_comparison_plot(results)
            if plot1:
                generated_plots.append(plot1)
        except Exception as e:
            print(f"Error creating accuracy plot: {e}")
        
        try:
            plot2 = self.create_metrics_radar_chart(results)
            if plot2:
                generated_plots.append(plot2)
        except Exception as e:
            print(f"Error creating radar chart: {e}")
        
        try:
            plot3 = self.create_family_comparison_plot(results)
            if plot3:
                generated_plots.append(plot3)
        except Exception as e:
            print(f"Error creating family comparison plot: {e}")
        
        try:
            plot4 = self.create_performance_heatmap(results)
            if plot4:
                generated_plots.append(plot4)
        except Exception as e:
            print(f"Error creating performance heatmap: {e}")
        
        try:
            plot5 = self.create_confusion_matrix_grid(results)
            if plot5:
                generated_plots.append(plot5)
        except Exception as e:
            print(f"Error creating confusion matrix grid: {e}")
        
        try:
            plot6 = self.create_roc_curves_comparison(results)
            if plot6:
                generated_plots.append(plot6)
        except Exception as e:
            print(f"Error creating ROC curves comparison: {e}")
        
        print(f"{len(generated_plots)} visualizations generated successfully!")
        return generated_plots


def main():
    """Main function for plot generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate validation plots")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="validation/visualization/plots",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create generator and generate plots
    generator = ValidationPlotGenerator(output_dir=args.output_dir)
    plots = generator.generate_all_plots(args.results)
    
    print(f"\nPlots generated:")
    for plot in plots:
        print(f"  - {plot}")


if __name__ == "__main__":
    main()
