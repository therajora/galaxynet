"""
Visualizations for domain shift analysis.

This submodule implements plots and visualizations to
analyze and compare results between SDSS and S-PLUS domains.
"""

from .domain_shift_plots import (
    plot_domain_comparison,
    plot_feature_distributions,
    plot_performance_comparison,
    plot_confusion_matrix_comparison,
    plot_confidence_distributions,
    plot_calibration_curves,
    plot_feature_space_visualization
)

__all__ = [
    'plot_domain_comparison',
    'plot_feature_distributions',
    'plot_performance_comparison',
    'plot_confusion_matrix_comparison',
    'plot_confidence_distributions',
    'plot_calibration_curves',
    'plot_feature_space_visualization'
]
