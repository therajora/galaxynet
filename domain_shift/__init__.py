"""
Domain Shift Module - Domain Shift Analysis between SDSS and S-PLUS

This module implements tools to analyze domain shift problems
when models trained on SDSS data are applied to S-PLUS data.

Main components:
- DomainShiftAnalyzer: Main analysis class
- metrics: Domain shift specific metrics
- visualization: Comparative plots and visualizations
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Importações condicionais para evitar erros de dependência
try:
    from .analyzer import DomainShiftAnalyzer
except ImportError:
    DomainShiftAnalyzer = None

try:
    from .metrics import (
        calculate_domain_shift_metrics,
        calculate_distribution_distance,
        calculate_feature_statistics
    )
except ImportError:
    calculate_domain_shift_metrics = None
    calculate_distribution_distance = None
    calculate_feature_statistics = None

try:
    from .visualization import (
        plot_domain_comparison,
        plot_feature_distributions,
        plot_performance_comparison,
        plot_confusion_matrix_comparison
    )
except ImportError:
    plot_domain_comparison = None
    plot_feature_distributions = None
    plot_performance_comparison = None
    plot_confusion_matrix_comparison = None

__all__ = [
    'DomainShiftAnalyzer',
    'calculate_domain_shift_metrics',
    'calculate_distribution_distance', 
    'calculate_feature_statistics',
    'plot_domain_comparison',
    'plot_feature_distributions',
    'plot_performance_comparison',
    'plot_confusion_matrix_comparison'
]
