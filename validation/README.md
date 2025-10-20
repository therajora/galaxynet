# `galaxynet/validation/` Module

This module provides comprehensive validation and benchmarking tools for galaxy classification models within the GalaxyNet project. It implements systematic model evaluation, performance comparison, and detailed analysis to help researchers and practitioners make informed decisions about model selection and optimization.

## Files:

- `__init__.py`: Initializes the `validation` package, exposing key components including metrics, validators, and benchmark tools.
- `metrics.py`: Implements classification metrics including confusion matrix, accuracy, precision, recall, F1-score, ROC curve, and AUC calculations with visualization capabilities.
- `validator.py`: Provides model validation functionality for loading trained models, validating on test datasets, and generating comprehensive evaluation reports.
- `benchmark.py`: Implements systematic comparison of multiple classification models with automated benchmarking, ranking, and visualization generation.
- `visualization/`: Submodule containing comprehensive visualization tools for validation results (see `visualization/README.md` for details).
- `README.md`: This file, summarizing the module.

## Functionality:

- **Model Validation**: Load and validate trained models on test datasets with comprehensive metrics calculation.
- **Performance Metrics**: Calculate and visualize standard classification metrics including accuracy, precision, recall, F1-score, and ROC-AUC.
- **Model Benchmarking**: Systematic comparison of multiple models with automated ranking and performance analysis.
- **Comprehensive Visualization**: Generate detailed plots and comparative visualizations for model analysis.
- **Error Analysis**: Detailed analysis of model errors including confusion matrices and error patterns.
- **Model Ranking**: Automatic ranking of models based on performance metrics with statistical analysis.

## Quick Usage Examples:

### 1. Validate a single model:

```python
from galaxynet.validation import ModelValidator

validator = ModelValidator(
    model_name="efficientnet_b0",
    model_path="models/efficientnet_b0.pth",
    class_names=["Regular", "Peculiar"]
)

results = validator.validate_dataset(test_loader, save_results=True)
```

### 2. Run model benchmarking:

```python
from galaxynet.validation import ModelBenchmark

benchmark = ModelBenchmark(
    test_loader=test_loader,
    class_names=["Regular", "Peculiar"],
    output_dir="benchmark_results"
)

model_paths = {
    "efficientnet_b0": "models/efficientnet_b0.pth",
    "resnet50": "models/resnet50.pth",
    "vit_b_16": "models/vit_b_16.pth"
}

results_df = benchmark.run_benchmark(model_paths, save_results=True)
benchmark.print_summary()
```

### 3. Calculate classification metrics:

```python
from galaxynet.validation import ClassificationMetrics

metrics = ClassificationMetrics(class_names=["Regular", "Peculiar"])
results = metrics.calculate_all_metrics(y_true, y_pred, y_prob)
metrics.print_summary()
```

### 4. Generate validation visualizations:

```python
from galaxynet.validation.visualization import ValidationPlotGenerator

generator = ValidationPlotGenerator(output_dir="validation/plots")
plots = generator.generate_all_plots("results/validation_results.json")
```

## Dependencies:

- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tqdm`
- `pathlib`
- `json`

## Output:

The module generates:

1. **Validation Results**: Comprehensive evaluation reports with all classification metrics
2. **Benchmark Reports**: Comparative analysis of multiple models with rankings
3. **Visualizations**: Professional plots and charts for model analysis
4. **Error Analysis**: Detailed breakdown of model errors and performance patterns
5. **Statistical Summaries**: Performance statistics and model rankings

## Integration:

This module integrates seamlessly with the rest of the GalaxyNet project:

- **Model Training**: Validates models trained using the `model/` module
- **Data Processing**: Works with datasets prepared by the `data/` module
- **Domain Shift Analysis**: Complements the `domain_shift/` module for comprehensive analysis
- **Results Management**: Integrates with the `results/` module for versioning and logging

The validation module provides a complete solution for model evaluation and comparison, making it easy to assess model performance and make informed decisions about model selection and optimization.