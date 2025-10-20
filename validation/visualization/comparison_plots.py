#!/usr/bin/env python3
"""
Comparative Plot Generator for Model Validation.

This module creates specific comparative visualizations for
performance analysis between different models and architectures.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import json

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Matplotlib configuration
plt.style.use('default')
sns.set_palette("husl")


class ComparisonPlotGenerator:
    """
    Comparative plot generator for model validation.
    """
    
    def __init__(self, output_dir: str = "validation/visualization/comparisons"):
        """
        Initialize the comparative plot generator.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir) / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "comparisons").mkdir(parents=True, exist_ok=True)
        
        # Style configurations
        self.colors = {
            'efficientnet': '#FF6B6B',
            'resnet': '#4ECDC4', 
            'vit': '#45B7D1',
            'resnext': '#96CEB4',
            'from_scratch': '#FFA07A',
            'best': '#2E8B57',
            'worst': '#DC143C',
            'average': '#FF8C00'
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
    
    def create_model_ranking_plot(self, results: List[Dict]) -> Path:
        """
        Create model ranking plot.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Sort models by accuracy
        sorted_results = sorted(results, key=lambda x: x.get('accuracy', 0.0), reverse=True)
        
        # Prepare data
        model_names = [r.get('model_name', 'Unknown') for r in sorted_results]
        accuracies = [r.get('accuracy', 0.0) for r in sorted_results]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(model_names))
        bars = ax.barh(y_pos, accuracies, 
                      color=[self._get_model_color(name) for name in model_names],
                      alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Chart configuration
        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Ranking by Accuracy', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add values on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{acc:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
        
        # Highlight best and worst models
        if len(accuracies) > 0:
            best_acc = max(accuracies)
            worst_acc = min(accuracies)
            
            # Line for best model
            ax.axvline(x=best_acc, color=self.colors['best'], linestyle='--', alpha=0.7,
                      label=f'Best: {best_acc:.3f}')
            
            # Line for worst model
            ax.axvline(x=worst_acc, color=self.colors['worst'], linestyle='--', alpha=0.7,
                      label=f'Worst: {worst_acc:.3f}')
            
            ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comparisons" / "model_ranking.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Model ranking plot saved to: {output_path}")
        return output_path
    
    def create_metrics_comparison_plot(self, results: List[Dict]) -> Path:
        """
        Create multiple metrics comparison plot.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        model_names = [r.get('model_name', f'Model {i+1}') for i, r in enumerate(results)]
        
        # Create figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Plot each metric
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = [r.get(metric, 0.0) for r in results]
            
            # Create bar chart
            bars = ax.bar(range(len(model_names)), values,
                         color=[self._get_model_color(name) for name in model_names],
                         alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Chart configuration
            ax.set_title(f'{label}', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add values on bars
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Remove empty axis if necessary
        if len(metrics) < len(axes):
            axes[-1].set_visible(False)
        
        # Adjust layout
        plt.suptitle('Metrics Comparison by Model', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comparisons" / "metrics_comparison.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Metrics comparison plot saved to: {output_path}")
        return output_path
    
    def create_family_performance_plot(self, results: List[Dict]) -> Path:
        """
        Create performance plot by model family.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Categorize models
        categories = self._categorize_models(results)
        
        # Prepare data
        family_names = []
        family_stats = {
            'mean': [], 'std': [], 'min': [], 'max': []
        }
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
                
                family_stats['mean'].append(np.mean(accuracies))
                family_stats['std'].append(np.std(accuracies))
                family_stats['min'].append(np.min(accuracies))
                family_stats['max'].append(np.max(accuracies))
                
                family_colors.append(color_map.get(family, '#CCCCCC'))
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Mean with error bars
        x_pos = np.arange(len(family_names))
        bars = ax1.bar(x_pos, family_stats['mean'], yerr=family_stats['std'],
                      color=family_colors, alpha=0.8, capsize=5,
                      edgecolor='black', linewidth=0.5)
        
        ax1.set_xlabel('Model Family', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Accuracy', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Performance by Family', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(family_names)
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add values on bars
        for bar, mean, std in zip(bars, family_stats['mean'], family_stats['std']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        # Plot 2: Performance range
        ax2.bar(x_pos, family_stats['max'], color=family_colors, alpha=0.3, 
               label='Maximum', edgecolor='black', linewidth=0.5)
        ax2.bar(x_pos, family_stats['min'], color=family_colors, alpha=0.8,
               label='Minimum', edgecolor='black', linewidth=0.5)
        
        ax2.set_xlabel('Model Family', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Performance Range by Family', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(family_names)
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comparisons" / "family_performance.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Family performance plot saved to: {output_path}")
        return output_path
    
    def create_model_complexity_plot(self, results: List[Dict]) -> Path:
        """
        Create complexity vs performance plot.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Define complexity based on model (approximate number of parameters)
        complexity_map = {
            'efficientnet_b0': 5.3, 'efficientnet_b1': 7.8, 'efficientnet_b2': 9.2,
            'efficientnet_b3': 12.0, 'efficientnet_b4': 19.0,
            'resnet50_v1': 25.6, 'resnet50_v2': 25.6, 'resnext50': 25.0,
            'vit_b_16': 86.6, 'vit_b_32': 86.6,
            'galaxynet': 1.2, 'galaxynet_v2': 2.1
        }
        
        # Prepare data
        model_names = []
        complexities = []
        accuracies = []
        colors = []
        
        for result in results:
            model_name = result.get('model_name', '')
            model_names.append(model_name)
            
            # Search complexity
            complexity = 1.0  # Default
            for key, value in complexity_map.items():
                if key in model_name.lower():
                    complexity = value
                    break
            
            complexities.append(complexity)
            accuracies.append(result.get('accuracy', 0.0))
            colors.append(self._get_model_color(model_name))
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create scatter plot
        scatter = ax.scatter(complexities, accuracies, c=colors, s=100, alpha=0.7, 
                           edgecolors='black', linewidth=0.5)
        
        # Add model labels
        for i, model_name in enumerate(model_names):
            ax.annotate(model_name, (complexities[i], accuracies[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Chart configuration
        ax.set_xlabel('Complexity (Millions of Parameters)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Complexity vs Performance of Models', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        # Add trend line
        if len(complexities) > 1:
            z = np.polyfit(complexities, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(complexities, p(complexities), "r--", alpha=0.8, 
                   label=f'Trend (slope: {z[0]:.3f})')
            ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comparisons" / "model_complexity.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Model complexity plot saved to: {output_path}")
        return output_path
    
    def create_metric_correlation_plot(self, results: List[Dict]) -> Path:
        """
        Create correlation plot between metrics.
        
        Args:
            results: List of results
            
        Returns:
            Path to saved plot
        """
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        
        # Create data matrix
        data_matrix = []
        for result in results:
            row = [result.get(metric, 0.0) for metric in metrics]
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create correlation heatmap
        im = ax.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Chart configuration
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(metric_labels)
        ax.set_title('Correlation Between Metrics', fontsize=14, fontweight='bold')
        
        # Add values in cells
        for i in range(len(metric_labels)):
            for j in range(len(metric_labels)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation Coefficient', fontsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = self.output_dir / "comparisons" / "metric_correlation.png"
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Metric correlation plot saved to: {output_path}")
        return output_path
    
    def _categorize_models(self, results: List[Dict]) -> Dict[str, List[Dict]]:
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
    
    def _get_model_color(self, model_name: str) -> str:
        """
        Retorna cor baseada no nome do modelo.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Cor em formato hex
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
    
    def generate_all_comparison_plots(self, results_path: str) -> List[Path]:
        """
        Generate all comparison plots.
        
        Args:
            results_path: Path to results file
            
        Returns:
            List of paths to generated plots
        """
        print("Generating comparison plots...")
        
        # Load results
        results = self.load_results(results_path)
        
        if not results:
            print("No results found!")
            return []
        
        generated_plots = []
        
        # Generate each type of plot
        try:
            plot1 = self.create_model_ranking_plot(results)
            if plot1:
                generated_plots.append(plot1)
        except Exception as e:
            print(f"Error creating ranking plot: {e}")
        
        try:
            plot2 = self.create_metrics_comparison_plot(results)
            if plot2:
                generated_plots.append(plot2)
        except Exception as e:
            print(f"Error creating metrics plot: {e}")
        
        try:
            plot3 = self.create_family_performance_plot(results)
            if plot3:
                generated_plots.append(plot3)
        except Exception as e:
            print(f"Error creating family plot: {e}")
        
        try:
            plot4 = self.create_model_complexity_plot(results)
            if plot4:
                generated_plots.append(plot4)
        except Exception as e:
            print(f"Error creating complexity plot: {e}")
        
        try:
            plot5 = self.create_metric_correlation_plot(results)
            if plot5:
                generated_plots.append(plot5)
        except Exception as e:
            print(f"Error creating correlation plot: {e}")
        
        print(f"{len(generated_plots)} comparison plots generated successfully!")
        return generated_plots


def main():
    """Main function for comparison plot generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate validation comparison plots")
    parser.add_argument("--results", type=str, required=True,
                       help="Path to results JSON file")
    parser.add_argument("--output-dir", type=str, default="validation/visualization/comparisons",
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create generator and generate plots
    generator = ComparisonPlotGenerator(output_dir=args.output_dir)
    plots = generator.generate_all_comparison_plots(args.results)
    
    print(f"\nComparison plots generated:")
    for plot in plots:
        print(f"  - {plot}")


if __name__ == "__main__":
    main()
