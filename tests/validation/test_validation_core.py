import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


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
    "model.pretrained.dataset",
    GalaxyPretrainedDataset=object,
    create_data_loaders=lambda *args, **kwargs: None,
    get_imagenet_transforms=lambda *args, **kwargs: None,
)
_stub_validation_module(
    "torch",
    device=lambda value: value,
    cuda=SimpleNamespace(is_available=lambda: False),
    utils=SimpleNamespace(data=SimpleNamespace(DataLoader=object)),
)
_stub_validation_module(
    "validation.visualization",
    ValidationPlotGenerator=object,
    ComparisonPlotGenerator=object,
    ModelAnalysisPlotter=object,
)


from validation.validation_core import (
    ValidationDependencies,
    ValidationRequest,
    run_validation,
)


def test_run_validation_dispatches_find_mode():
    calls = []

    def fake_find_models(results_dir):
        calls.append(results_dir)
        return {"efficientnet_b0": "results/models/efficientnet_b0/best_model.pth"}

    request = ValidationRequest(
        mode="find",
        model_name=None,
        model_path=None,
        data_dir="data/complete_sdss",
        output_dir=None,
        device="auto",
        results_dir="results/models",
    )

    deps = ValidationDependencies(
        validate_single_runner=lambda request: None,
        benchmark_runner=lambda request: None,
        find_models_runner=fake_find_models,
        cuda_available=lambda: False,
    )

    result = run_validation(request, deps)

    assert result.success is True
    assert result.model_paths == {
        "efficientnet_b0": "results/models/efficientnet_b0/best_model.pth"
    }
    assert calls == ["results/models"]


def test_run_validation_dispatches_single_mode():
    calls = []

    def fake_single_runner(request):
        calls.append(request)
        return {
            "success": True,
            "message": "single ok",
            "output_dir": "validation_results/efficientnet_b0",
            "results_path": "validation_results/efficientnet_b0/metrics.json",
            "model_paths": None,
        }

    request = ValidationRequest(
        mode="single",
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        data_dir="data/complete_sdss",
        output_dir="validation_results/efficientnet_b0",
        device="auto",
        results_dir="results/models",
    )

    deps = ValidationDependencies(
        validate_single_runner=fake_single_runner,
        benchmark_runner=lambda request: None,
        find_models_runner=lambda results_dir: {},
        cuda_available=lambda: False,
    )

    result = run_validation(request, deps)

    assert result.success is True
    assert result.message == "single ok"
    assert calls[0].resolved_device == "cpu"


def test_run_validation_dispatches_benchmark_mode():
    calls = []

    def fake_benchmark_runner(request):
        calls.append(request)
        return {
            "success": True,
            "message": "benchmark ok",
            "output_dir": "benchmark_results",
            "results_path": "benchmark_results/benchmark_comparison.csv",
            "model_paths": {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        }

    request = ValidationRequest(
        mode="benchmark",
        model_name=None,
        model_path=None,
        data_dir="data/complete_sdss",
        output_dir="benchmark_results",
        device="cpu",
        results_dir="results/models",
    )

    deps = ValidationDependencies(
        validate_single_runner=lambda request: None,
        benchmark_runner=fake_benchmark_runner,
        find_models_runner=lambda results_dir: {
            "resnet50_v1": "results/models/resnet50_v1/best_model.pth"
        },
        cuda_available=lambda: False,
    )

    result = run_validation(request, deps)

    assert result.success is True
    assert result.message == "benchmark ok"
    assert calls[0].resolved_device == "cpu"


def test_run_validation_requires_model_identifiers_for_single():
    request = ValidationRequest(
        mode="single",
        model_name=None,
        model_path=None,
        data_dir="data/complete_sdss",
        output_dir=None,
        device="cpu",
        results_dir="results/models",
    )

    deps = ValidationDependencies(
        validate_single_runner=lambda request: None,
        benchmark_runner=lambda request: None,
        find_models_runner=lambda results_dir: {},
        cuda_available=lambda: False,
    )

    with pytest.raises(
        ValueError, match="single mode requires model_name and model_path"
    ):
        run_validation(request, deps)


def test_validate_single_entry_returns_structured_result(monkeypatch):
    import validation.validate_models as validate_models

    class FakeValidator:
        def __init__(self, model_name, model_path, device, class_names):
            self.model_name = model_name

        def validate_dataset(self, test_loader, save_results, output_dir):
            return {"metrics": {"accuracy": 0.9}}

    monkeypatch.setattr(validate_models, "ModelValidator", FakeValidator)
    monkeypatch.setattr(
        validate_models, "build_test_loader", lambda data_dir: "test_loader"
    )
    monkeypatch.setattr(validate_models, "resolve_device", lambda device: "cpu")

    result = validate_models.validate_single_entry(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        data_dir="data/complete_sdss",
        output_dir="validation_results/efficientnet_b0",
        device="auto",
    )

    assert result["success"] is True
    assert result["output_dir"] == "validation_results/efficientnet_b0"


def test_benchmark_entry_returns_structured_result(monkeypatch):
    import validation.validate_models as validate_models

    class FakeBenchmark:
        def __init__(self, test_loader, class_names, device, output_dir):
            self.output_dir = output_dir

        def run_benchmark(self, model_paths, save_results):
            return SimpleNamespace(shape=(1, 3))

        def print_summary(self):
            return None

    monkeypatch.setattr(validate_models, "ModelBenchmark", FakeBenchmark)
    monkeypatch.setattr(
        validate_models, "build_test_loader", lambda data_dir: "test_loader"
    )
    monkeypatch.setattr(validate_models, "resolve_device", lambda device: "cpu")

    result = validate_models.run_benchmark_entry(
        model_paths={"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        data_dir="data/complete_sdss",
        output_dir="benchmark_results",
        device="auto",
    )

    assert result["success"] is True
    assert result["results_path"] == "benchmark_results/benchmark_comparison.csv"
