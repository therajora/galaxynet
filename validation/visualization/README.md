# `galaxynet/validation/visualization/` Module

This module provides comprehensive visualization tools for model validation and analysis within the GalaxyNet project. It creates detailed plots and comparative visualizations to help understand model performance, identify patterns, and make informed decisions about model selection and optimization.

## Files:

- `__init__.py`: Initializes the `visualization` package, exposing key components.
- `plot_generator.py`: Creates general validation plots including accuracy comparisons, metrics radar charts, family performance analysis, performance heatmaps, confusion matrix grids, and ROC curve comparisons.
- `comparison_plots.py`: Generates comparative visualizations for model ranking, metrics comparison, family performance analysis, model complexity analysis, and metric correlation analysis.
- `model_analysis.py`: Provides detailed model analysis including individual model deep-dive analysis, model evolution tracking, and comprehensive error analysis.
- `README.md`: This file, summarizing the module.

## Functionality:

- **General Validation Plots**: Creates standard validation visualizations including accuracy comparisons, metrics radar charts, and performance heatmaps.
- **Comparative Analysis**: Generates side-by-side comparisons of different models and architectures, helping identify the best performing models.
- **Detailed Model Analysis**: Provides in-depth analysis of individual models, including confusion matrices, ROC curves, and error analysis.
- **Family Performance Analysis**: Groups models by architecture family (EfficientNet, ResNet, ViT, etc.) and analyzes performance patterns.
- **Error Analysis**: Creates comprehensive error analysis including false positive/negative rates, confidence distributions, and calibration analysis.
- **Model Evolution Tracking**: Simulates and visualizes model performance evolution over training epochs.

## Quick Usage Examples:

### 1. Generate all validation plots:

```python
from galaxynet.validation.visualization import ValidationPlotGenerator

generator = ValidationPlotGenerator(output_dir="validation/plots")
plots = generator.generate_all_plots("results/validation_results.json")
```

### 2. Create comparative analysis:

```python
from galaxynet.validation.visualization import ComparisonPlotGenerator

comparator = ComparisonPlotGenerator(output_dir="validation/comparisons")
plots = comparator.generate_all_comparison_plots("results/validation_results.json")
```

### 3. Perform detailed model analysis:

```python
from galaxynet.validation.visualization import ModelAnalysisPlotter

analyzer = ModelAnalysisPlotter(output_dir="validation/analysis")
plots = analyzer.generate_all_analysis_plots("results/validation_results.json")
```

### 4. Command-line usage:

```bash
# Generate all validation plots
python galaxynet/validation/visualization/plot_generator.py --results results/validation_results.json

# Generate comparative plots
python galaxynet/validation/visualization/comparison_plots.py --results results/validation_results.json

# Generate model analysis
python galaxynet/validation/visualization/model_analysis.py --results results/validation_results.json
```

## Dependencies:

- `matplotlib`
- `seaborn`
- `numpy`
- `pandas`
- `scikit-learn`
- `json`
- `pathlib`

## Output:

The module generates various types of visualizations:

1. **Accuracy Comparison Plots**: Bar charts comparing model accuracies
2. **Metrics Radar Charts**: Polar plots showing multiple metrics for each model
3. **Family Performance Plots**: Box plots and bar charts comparing model families
4. **Performance Heatmaps**: Color-coded matrices showing all metrics for all models
5. **Confusion Matrix Grids**: Side-by-side confusion matrices for multiple models
6. **ROC Curve Comparisons**: Overlaid ROC curves for model comparison
7. **Model Ranking Plots**: Horizontal bar charts ranking models by performance
8. **Complexity vs Performance Plots**: Scatter plots showing model complexity vs accuracy
9. **Error Analysis Plots**: Comprehensive error analysis including false positive/negative rates
10. **Detailed Model Analysis**: Multi-panel analysis of individual models

All plots are saved as high-resolution PNG files with consistent styling and professional appearance suitable for research publications and presentations.
