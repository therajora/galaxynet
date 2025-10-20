"""
Benchmark Module for Classification Model Comparison.

Implements systematic comparison of different architectures:
- EfficientNet (B0 to B4)
- ResNet (50v1, 50v2, ResNeXt-50)
- Vision Transformer (ViT-Base/16, ViT-Base/32)
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.pretrained.model_factory import get_available_models, get_model_info, count_parameters
from validation.validator import ModelValidator
from validation.metrics import ClassificationMetrics


class ModelBenchmark:
    """
    Class for benchmarking multiple classification models.
    """
    
    def __init__(
        self,
        test_loader: DataLoader,
        class_names: List[str] = None,
        device: torch.device = None,
        output_dir: Union[str, Path] = "benchmark_results"
    ):
        """
        Initialize the benchmark.
        
        Args:
            test_loader: DataLoader with test data
            class_names: Class names
            device: Device for validation
            output_dir: Directory to save results
        """
        self.test_loader = test_loader
        self.class_names = class_names or ['Regular', 'Peculiar']
        self.num_classes = len(self.class_names)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store results
        self.results = {}
        self.comparison_df = None
        
        print(f"Benchmark initialized:")
        print(f"   Device: {self.device}")
        print(f"   Classes: {self.class_names}")
        print(f"   Output directory: {self.output_dir}")
    
    def run_benchmark(
        self,
        model_paths: Dict[str, Union[str, Path]],
        save_results: bool = True
    ) -> pd.DataFrame:
        """
        Execute benchmark on multiple models.
        
        Args:
            model_paths: Dictionary {model_name: model_path}
            save_results: Whether to save results
            
        Returns:
            DataFrame with metrics comparison
        """
        print(f"\nStarting benchmark of {len(model_paths)} models...")
        
        for model_name, model_path in tqdm(model_paths.items(), desc="Validating models"):
            try:
                print(f"\nValidating {model_name}...")
                
                # Validate model
                validator = ModelValidator(
                    model_name=model_name,
                    model_path=model_path,
                    device=self.device,
                    class_names=self.class_names
                )
                
                # Execute validation
                results = validator.validate_dataset(
                    test_loader=self.test_loader,
                    save_results=False  # Don't save individually
                )
                
                # Store results
                self.results[model_name] = {
                    'metrics': results['metrics'],
                    'model_info': validator.get_model_info(),
                    'validator': validator
                }
                
                print(f"{model_name} validated successfully")
                
            except Exception as e:
                print(f"Error validating {model_name}: {e}")
                self.results[model_name] = {'error': str(e)}
        
        # Create comparison DataFrame
        self.comparison_df = self._create_comparison_dataframe()
        
        if save_results:
            self._save_benchmark_results()
        
        return self.comparison_df
    
    def _create_comparison_dataframe(self) -> pd.DataFrame:
        """
        Create DataFrame with metrics comparison.
        
        Returns:
            DataFrame with metrics from all models
        """
        data = []
        
        for model_name, result in self.results.items():
            if 'error' in result:
                # Model with error
                row = {
                    'Model': model_name,
                    'Status': 'Error',
                    'Error': result['error'],
                    'Accuracy': np.nan,
                    'Precision': np.nan,
                    'Recall': np.nan,
                    'F1-Score': np.nan,
                    'ROC-AUC': np.nan,
                    'Parameters': np.nan
                }
            else:
                # Valid model
                metrics = result['metrics']
                model_info = result['model_info']
                params = model_info['parameters']
                
                # Extract AUC (can be dict for multiclass)
                roc_auc = metrics.get('roc_auc', np.nan)
                if isinstance(roc_auc, dict):
                    # For multiclass, use weighted average
                    roc_auc = np.mean(list(roc_auc.values()))
                
                row = {
                    'Model': model_name,
                    'Status': 'Success',
                    'Error': '',
                    'Accuracy': metrics.get('accuracy', np.nan),
                    'Precision': metrics.get('precision', np.nan),
                    'Recall': metrics.get('recall', np.nan),
                    'F1-Score': metrics.get('f1_score', np.nan),
                    'ROC-AUC': roc_auc,
                    'Parameters': params.get('total', np.nan)
                }
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by F1-Score (best first)
        df = df.sort_values('F1-Score', ascending=False, na_last=True)
        
        return df
    
    def _save_benchmark_results(self):
        """Save benchmark results."""
        print(f"\nSaving benchmark results...")
        
        # Save DataFrame
        self.comparison_df.to_csv(
            self.output_dir / "benchmark_comparison.csv", 
            index=False
        )
        
        # Save detailed results
        detailed_results = {}
        for model_name, result in self.results.items():
            if 'error' not in result:
                detailed_results[model_name] = {
                    'metrics': result['metrics'],
                    'model_info': result['model_info']
                }
            else:
                detailed_results[model_name] = {'error': result['error']}
        
        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=4)
        
        # Generate visualizations
        self._generate_benchmark_visualizations()
        
        print(f"Results saved to: {self.output_dir}")
    
    def _generate_benchmark_visualizations(self):
        """Generate benchmark visualizations."""
        print("Generating benchmark visualizations...")
        
        # Remove models with errors
        valid_df = self.comparison_df[self.comparison_df['Status'] == 'Success'].copy()
        
        if len(valid_df) == 0:
            print("No valid models for visualization")
            return
        
        # 1. Metrics comparison
        self._plot_metrics_comparison(valid_df)
        
        # 2. Parameters vs performance comparison
        self._plot_parameters_vs_performance(valid_df)
        
        # 3. Model ranking
        self._plot_model_ranking(valid_df)
        
        # 4. Metrics heatmap
        self._plot_metrics_heatmap(valid_df)
    
    def _plot_metrics_comparison(self, df: pd.DataFrame):
        """Plot metrics comparison."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Sort by metric
            sorted_df = df.sort_values(metric, ascending=True)
            
            bars = ax.barh(sorted_df['Model'], sorted_df[metric])
            ax.set_xlabel(metric)
            ax.set_title(f'{metric} Comparison')
            ax.grid(True, alpha=0.3)
            
            # Color bars by model family
            for j, (bar, model_name) in enumerate(zip(bars, sorted_df['Model'])):
                if 'efficientnet' in model_name.lower():
                    bar.set_color('skyblue')
                elif 'resnet' in model_name.lower() or 'resnext' in model_name.lower():
                    bar.set_color('lightcoral')
                elif 'vit' in model_name.lower():
                    bar.set_color('lightgreen')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameters_vs_performance(self, df: pd.DataFrame):
        """Plot parameters vs performance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # F1-Score vs Parameters
        scatter = ax1.scatter(df['Parameters'], df['F1-Score'], 
                             s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
        
        for i, model in enumerate(df['Model']):
            ax1.annotate(model, (df.iloc[i]['Parameters'], df.iloc[i]['F1-Score']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('F1-Score')
        ax1.set_title('F1-Score vs Parameters')
        ax1.grid(True, alpha=0.3)
        
        # ROC-AUC vs Parameters
        scatter = ax2.scatter(df['Parameters'], df['ROC-AUC'], 
                             s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
        
        for i, model in enumerate(df['Model']):
            ax2.annotate(model, (df.iloc[i]['Parameters'], df.iloc[i]['ROC-AUC']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_title('ROC-AUC vs Parameters')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "parameters_vs_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_ranking(self, df: pd.DataFrame):
        """Plot model ranking."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Use F1-Score for ranking
        sorted_df = df.sort_values('F1-Score', ascending=True)
        
        y_pos = np.arange(len(sorted_df))
        bars = ax.barh(y_pos, sorted_df['F1-Score'])
        
        # Color by family
        for i, (bar, model_name) in enumerate(zip(bars, sorted_df['Model'])):
            if 'efficientnet' in model_name.lower():
                bar.set_color('skyblue')
            elif 'resnet' in model_name.lower() or 'resnext' in model_name.lower():
                bar.set_color('lightcoral')
            elif 'vit' in model_name.lower():
                bar.set_color('lightgreen')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['Model'])
        ax.set_xlabel('F1-Score')
        ax.set_title('Model Ranking (F1-Score)')
        ax.grid(True, alpha=0.3)
        
        # Add values on bars
        for i, v in enumerate(sorted_df['F1-Score']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_ranking.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metrics_heatmap(self, df: pd.DataFrame):
        """Plot metrics heatmap."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        # Prepare data for heatmap
        heatmap_data = df[metrics].T
        heatmap_data.columns = df['Model']
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0.5,
            ax=ax,
            cbar_kws={'label': 'Score'}
        )
        
        ax.set_title('Metrics Heatmap by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Metric')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """Print benchmark summary."""
        if self.comparison_df is None:
            print("No results available. Run run_benchmark() first.")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # General statistics
        total_models = len(self.comparison_df)
        successful_models = len(self.comparison_df[self.comparison_df['Status'] == 'Success'])
        failed_models = total_models - successful_models
        
        print(f"Total models: {total_models}")
        print(f"Successful models: {successful_models}")
        print(f"Failed models: {failed_models}")
        
        if successful_models > 0:
            valid_df = self.comparison_df[self.comparison_df['Status'] == 'Success']
            
            print(f"\nTOP 3 MODELS (F1-Score):")
            top3 = valid_df.head(3)
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                print(f"  {i}. {row['Model']}: {row['F1-Score']:.4f}")
            
            print(f"\nSTATISTICS:")
            print(f"  Best F1-Score: {valid_df['F1-Score'].max():.4f}")
            print(f"  Worst F1-Score: {valid_df['F1-Score'].min():.4f}")
            print(f"  Average F1-Score: {valid_df['F1-Score'].mean():.4f}")
            
            print(f"\n  Best ROC-AUC: {valid_df['ROC-AUC'].max():.4f}")
            print(f"  Worst ROC-AUC: {valid_df['ROC-AUC'].min():.4f}")
            print(f"  Average ROC-AUC: {valid_df['ROC-AUC'].mean():.4f}")
        
        print("="*80)


def run_complete_benchmark(
    test_loader: DataLoader,
    model_paths: Dict[str, Union[str, Path]],
    class_names: List[str] = None,
    device: torch.device = None,
    output_dir: Union[str, Path] = "benchmark_results"
) -> pd.DataFrame:
    """
    Utility function to run complete benchmark.
    
    Args:
        test_loader: Test DataLoader
        model_paths: Dictionary {model_name: model_path}
        class_names: Class names
        device: Device
        output_dir: Output directory
        
    Returns:
        DataFrame with results
    """
    benchmark = ModelBenchmark(
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        output_dir=output_dir
    )
    
    results_df = benchmark.run_benchmark(model_paths, save_results=True)
    benchmark.print_summary()
    
    return results_df
