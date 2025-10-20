#!/usr/bin/env python3
"""
Main Script for Validation Visualization Execution.

This script orchestrates the complete execution of validation visualizations,
including basic plots, comparative plots, and detailed analyses.
"""

import sys
import argparse
from pathlib import Path
import json
from datetime import datetime

# Add root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validation.visualization.plot_generator import ValidationPlotGenerator
from validation.visualization.comparison_plots import ComparisonPlotGenerator
from validation.visualization.model_analysis import ModelAnalysisPlotter


def create_visualization_report(results_path: str, plots_dir: str) -> str:
    """
    Create report of generated visualizations.
    
    Args:
        results_path: Path to JSON results
        plots_dir: Directory with plots
        
    Returns:
        Path to the report
    """
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Calculate statistics
    total_models = len(results)
    avg_accuracy = sum(r.get('accuracy', 0.0) for r in results) / total_models if total_models > 0 else 0
    best_model = max(results, key=lambda x: x.get('accuracy', 0.0)) if results else None
    
    # Prepare data for report
    best_model_name = best_model.get('model_name', 'N/A') if best_model else 'N/A'
    best_model_accuracy = best_model.get('accuracy', 0.0) if best_model else 0.0
    
    # Calculate performance gap
    if results:
        accuracies = [r.get('accuracy', 0.0) for r in results]
        performance_gap = max(accuracies) - min(accuracies)
    else:
        performance_gap = 0.0
    
    # Analysis by family
    families = {}
    for result in results:
        model_name = result.get('model_name', '')
        if 'efficientnet' in model_name.lower():
            family = 'EfficientNet'
        elif 'resnet' in model_name.lower() and 'resnext' not in model_name.lower():
            family = 'ResNet'
        elif 'vit' in model_name.lower():
            family = 'ViT'
        elif 'resnext' in model_name.lower():
            family = 'ResNeXt'
        elif 'galaxynet' in model_name.lower():
            family = 'From Scratch'
        else:
            family = 'Others'
        
        if family not in families:
            families[family] = []
        families[family].append(result.get('accuracy', 0.0))
    
    # Find best family
    if families:
        best_family = max(families.items(), key=lambda x: sum(x[1])/len(x[1]))[0]
    else:
        best_family = 'N/A'
    
    # Create report
    report_content = f"""# Visualization Report - Model Validation

## 📊 Executive Summary

**Generation Date:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

### General Statistics
- **Total Models Analyzed:** {total_models}
- **Average Accuracy:** {avg_accuracy:.3f}
- **Best Model:** {best_model_name}
- **Best Accuracy:** {best_model_accuracy:.3f}

## 🎨 Generated Visualizations

### 1. Basic Plots (`plots/`)
- **Accuracy Comparison:** `accuracy_comparison.png`
- **Metrics Radar Chart:** `metrics_radar_chart.png`
- **Family Comparison:** `family_comparison.png`
- **Performance Heatmap:** `performance_heatmap.png`
- **Confusion Matrix Grid:** `confusion_matrix_grid.png`
- **ROC Curves Comparison:** `roc_curves_comparison.png`

### 2. Comparative Plots (`comparisons/`)
- **Model Ranking:** `model_ranking.png`
- **Metrics Comparison:** `metrics_comparison.png`
- **Performance by Family:** `family_performance.png`
- **Complexity vs Performance:** `model_complexity.png`
- **Metric Correlation:** `metric_correlation.png`

### 3. Detailed Analyses (`analysis/`)
- **Best Model Detailed Analysis:** `detailed_analysis_*.png`
- **Model Evolution:** `model_evolution.png`
- **Error Analysis:** `error_analysis.png`

## 📈 Results Interpretation

### Main Metrics
- **Accuracy:** Proportion of correct predictions
- **Precision:** Proportion of positive predictions that are correct
- **Recall:** Proportion of positive cases identified correctly
- **F1-Score:** Harmonic mean between precision and recall
- **AUC:** Area under the ROC curve

### Model Families
"""
    
    for family, accuracies in families.items():
        avg_acc = sum(accuracies) / len(accuracies)
        report_content += f"- **{family}:** {len(accuracies)} models, average accuracy {avg_acc:.3f}\n"
    
    report_content += f"""

## 🔍 Key Insights

### 1. General Performance
- **Best Family:** {best_family}
- **Most Efficient Model:** {best_model_name}
- **Performance Gap:** {performance_gap:.3f}

### 2. Recommendations

#### For Production
- **Recommended Model:** {best_model_name}
- **Justification:** Best balance between accuracy and stability
- **Considerations:** Evaluate computational complexity vs performance

#### For Research
- **Promising Family:** {best_family}
- **Next Steps:** Investigate variations of the most promising architecture
- **Optimizations:** Fine-tuning and data augmentation

## 📁 File Structure

```
{plots_dir}/
├── plots/                          # Basic plots
│   ├── accuracy_comparison.png
│   ├── metrics_radar_chart.png
│   ├── family_comparison.png
│   ├── performance_heatmap.png
│   ├── confusion_matrix_grid.png
│   └── roc_curves_comparison.png
├── comparisons/                    # Comparative plots
│   ├── model_ranking.png
│   ├── metrics_comparison.png
│   ├── family_performance.png
│   ├── model_complexity.png
│   └── metric_correlation.png
└── analysis/                       # Detailed analyses
    ├── detailed_analysis_*.png
    ├── model_evolution.png
    └── error_analysis.png
```

## 🚀 Next Steps

1. **Detailed Analysis:** Investigate error cases of the best model
2. **Optimization:** Apply fine-tuning techniques
3. **Cross Validation:** Confirm robustness of results
4. **Ensemble:** Combine multiple models for better performance
5. **Deploy:** Implement model in production with monitoring

## 📚 References

- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Seaborn Gallery](https://seaborn.pydata.org/examples/index.html)

---
*Report automatically generated by the validation visualization system*
"""
    
    # Save report
    report_path = Path(plots_dir) / "visualization_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Visualization report saved to: {report_path}")
    return str(report_path)


def main():
    """Main function for visualization execution."""
    parser = argparse.ArgumentParser(
        description="Execute complete model validation visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python run_visualizations.py --results validation/tests/results/test_results.json
  python run_visualizations.py --results results.json --output-dir custom/plots
  python run_visualizations.py --results results.json --skip-analysis
        """
    )
    
    parser.add_argument("--results", type=str, required=True,
                       help="Path to JSON results file")
    parser.add_argument("--output-dir", type=str, default="validation/visualization",
                       help="Output directory (default: validation/visualization)")
    parser.add_argument("--skip-basic", action="store_true",
                       help="Skip basic plots generation")
    parser.add_argument("--skip-comparisons", action="store_true",
                       help="Skip comparative plots generation")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip detailed analyses generation")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STARTING VALIDATION VISUALIZATION GENERATION")
    print("=" * 60)
    
    # Create directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plots_dir = output_dir / "plots"
    comparisons_dir = output_dir / "comparisons"
    analysis_dir = output_dir / "analysis"
    
    generated_plots = []
    
    try:
        # 1. Generate basic plots
        if not args.skip_basic:
            print("\nPHASE 1: Generating basic plots...")
            basic_generator = ValidationPlotGenerator(output_dir=str(output_dir))
            basic_plots = basic_generator.generate_all_plots(args.results)
            generated_plots.extend(basic_plots)
        else:
            print("\nPHASE 1: Skipping basic plots...")
        
        # 2. Generate comparative plots
        if not args.skip_comparisons:
            print("\nPHASE 2: Generating comparative plots...")
            comparison_generator = ComparisonPlotGenerator(output_dir=str(output_dir))
            comparison_plots = comparison_generator.generate_all_comparison_plots(args.results)
            generated_plots.extend(comparison_plots)
        else:
            print("\nPHASE 2: Skipping comparative plots...")
        
        # 3. Generate detailed analyses
        if not args.skip_analysis:
            print("\nPHASE 3: Generating detailed analyses...")
            analysis_plotter = ModelAnalysisPlotter(output_dir=str(output_dir))
            analysis_plots = analysis_plotter.generate_all_analysis_plots(args.results)
            generated_plots.extend(analysis_plots)
        else:
            print("\nPHASE 3: Skipping detailed analyses...")
        
        # 4. Create report
        print("\nPHASE 4: Creating visualization report...")
        report_path = create_visualization_report(args.results, str(output_dir))
        
        # 5. Final summary
        print("\n" + "=" * 60)
        print("VALIDATION VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\nVISUALIZATION SUMMARY:")
        print(f"  • {len(generated_plots)} plots generated")
        print(f"  • Basic plots: {len([p for p in generated_plots if 'plots' in str(p)])}")
        print(f"  • Comparative plots: {len([p for p in generated_plots if 'comparisons' in str(p)])}")
        print(f"  • Detailed analyses: {len([p for p in generated_plots if 'analysis' in str(p)])}")
        
        print(f"\nGENERATED FILES:")
        print(f"  • Report: {report_path}")
        print(f"  • Basic plots: {plots_dir}/")
        print(f"  • Comparative plots: {comparisons_dir}/")
        print(f"  • Detailed analyses: {analysis_dir}/")
        
        print(f"\nAll visualizations generated successfully!")
        
    except Exception as e:
        print(f"\nERROR during visualization generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
