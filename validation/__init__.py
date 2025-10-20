"""
Validation and Benchmarking Module for Galaxy Classification Models.

This module provides tools for:
- Pre-trained model validation
- Classification metrics calculation
- Different architecture comparison
- Benchmark report generation
"""

from .metrics import (
    ClassificationMetrics,
    calculate_confusion_matrix,
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_roc_auc,
    plot_confusion_matrix,
    plot_roc_curve
)

from .validator import ModelValidator

from .benchmark import ModelBenchmark

# Visualization imports (optional)
try:
    from .visualization import (
        ValidationPlotGenerator,
        ComparisonPlotGenerator,
        ModelAnalysisPlotter
    )
    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    # Metrics
    'ClassificationMetrics',
    'calculate_confusion_matrix',
    'calculate_accuracy', 
    'calculate_precision',
    'calculate_recall',
    'calculate_f1_score',
    'calculate_roc_auc',
    'plot_confusion_matrix',
    'plot_roc_curve',
    
    # Validator
    'ModelValidator',
    
    # Benchmark
    'ModelBenchmark'
]

# Add visualizations if available
if _visualization_available:
    __all__.extend([
        'ValidationPlotGenerator',
        'ComparisonPlotGenerator',
        'ModelAnalysisPlotter'
    ])

__version__ = "1.0.0"
__author__ = "Galaxy Classification Team"
