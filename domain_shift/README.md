# Domain Shift Analysis Module

Module for analyzing domain shift when models trained on SDSS are applied to S-PLUS data.

## Structure

```
domain_shift/
├── analyzer.py              - Main analyzer class
├── analyze_domain_shift.py  - CLI script for analysis
├── metrics/                 - Domain shift metrics
│   ├── __init__.py
│   └── domain_shift_metrics.py
└── visualization/           - Plotting functions
    ├── __init__.py
    └── domain_shift_plots.py
```

## Quick Usage

```python
from domain_shift import DomainShiftAnalyzer

analyzer = DomainShiftAnalyzer(
    model_name="vit_b_32",
    model_path="results/models/best_model.pth",
    device=device,
    output_dir="domain_shift_results"
)

results = analyzer.analyze_domain_shift(
    sdss_data_dir="data/complete_sdss",
    splus_data_dir="data/complete_splus"
)
```

## CLI Usage

```bash
# Find available trained models
python analyze_domain_shift.py --mode find

# Analyze single model
python analyze_domain_shift.py --mode single \
    --model-name vit_b_32 \
    --model-path results/models/best_model.pth

# Analyze multiple models
python analyze_domain_shift.py --mode multiple
```

## What It Does

1. Loads trained model
2. Evaluates on SDSS test set (source domain)
3. Evaluates on S-PLUS test set (target domain)
4. Calculates performance degradation metrics
5. Computes statistical distances between distributions
6. Generates comparative visualizations

## Metrics Calculated

- Performance drop (accuracy, F1, AUC, etc.)
- Distribution distances (Wasserstein, Jensen-Shannon, KS test)
- Confidence shift
- Calibration shift (ECE)
- Feature space distances (MMD)

## Outputs

- JSON files with detailed metrics
- Comparison plots (performance, ROC curves, confusion matrices)
- Feature distribution visualizations
- Calibration curves

