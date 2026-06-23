import sys
from types import ModuleType


def _stub_validation_module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


_stub_validation_module(
    "validation.metrics",
    ClassificationMetrics=object,
    calculate_confusion_matrix=lambda *args, **kwargs: None,
    calculate_accuracy=lambda *args, **kwargs: None,
    calculate_precision=lambda *args, **kwargs: None,
    calculate_recall=lambda *args, **kwargs: None,
    calculate_f1_score=lambda *args, **kwargs: None,
    calculate_roc_auc=lambda *args, **kwargs: None,
    plot_confusion_matrix=lambda *args, **kwargs: None,
    plot_roc_curve=lambda *args, **kwargs: None,
)
_stub_validation_module("validation.validator", ModelValidator=object)
_stub_validation_module("validation.benchmark", ModelBenchmark=object)
_stub_validation_module(
    "validation.visualization",
    ValidationPlotGenerator=object,
    ComparisonPlotGenerator=object,
    ModelAnalysisPlotter=object,
)

from validation.validation_persistence import save_validation_outputs


class FakeMetricsCalculator:
    confusion_matrix = [[1, 0], [0, 1]]
    roc_curve_data = {"fpr": [0, 1], "tpr": [0, 1]}
    roc_auc = 0.95

    def save_metrics(self, path):
        path.write_text('{"accuracy": 0.9}', encoding="utf-8")


def test_save_validation_outputs_writes_metrics_and_predictions(tmp_path):
    save_validation_outputs(
        output_dir=tmp_path,
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        class_names=["Regular", "Peculiar"],
        y_true=[1, 0],
        y_pred=[1, 0],
        y_prob=[[0.1, 0.9], [0.8, 0.2]],
        metrics_calculator=FakeMetricsCalculator(),
        generate_visualizations=lambda **kwargs: None,
    )

    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.json").exists()


def test_save_validation_outputs_calls_visualization_generator(tmp_path):
    calls = {}
    metrics_calculator = FakeMetricsCalculator()

    def fake_generate_visualizations(**kwargs):
        calls["output_dir"] = kwargs["output_dir"]
        calls["metrics_calculator"] = kwargs["metrics_calculator"]

    save_validation_outputs(
        output_dir=tmp_path,
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        class_names=["Regular", "Peculiar"],
        y_true=[1, 0],
        y_pred=[1, 0],
        y_prob=[[0.1, 0.9], [0.8, 0.2]],
        metrics_calculator=metrics_calculator,
        generate_visualizations=fake_generate_visualizations,
    )

    assert calls["output_dir"] == tmp_path
    assert calls["metrics_calculator"] is metrics_calculator
