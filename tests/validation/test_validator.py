import importlib
import sys
from types import ModuleType, SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError:
    class _ArrayLike(list):
        def tolist(self):
            return [
                item.tolist() if hasattr(item, "tolist") else item
                for item in self
            ]

    class _NumpyFallback:
        @staticmethod
        def array(values):
            if isinstance(values, list):
                return _ArrayLike(
                    [
                        _NumpyFallback.array(item) if isinstance(item, list) else item
                        for item in values
                    ]
                )
            return values

    np = _NumpyFallback()


def _stub_module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.device = None

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array(self.values)


def _load_validator_module():
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    torch_module = ModuleType("torch")
    torch_module.device = str
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    torch_module.load = lambda *args, **kwargs: {}
    torch_module.no_grad = lambda: _NoGrad()
    torch_module.softmax = lambda outputs, dim=1: outputs
    torch_module.argmax = lambda outputs, dim=1: FakeTensor(
        [row.index(max(row)) for row in outputs.values]
    )
    sys.modules["torch"] = torch_module

    _stub_module("torch.nn", Module=object)
    _stub_module("torch.utils", data=SimpleNamespace(DataLoader=object))
    _stub_module("torch.utils.data", DataLoader=object)
    _stub_module("numpy", array=np.array)

    _stub_module("matplotlib", pyplot=SimpleNamespace(close=lambda *args, **kwargs: None))
    _stub_module("matplotlib.pyplot", close=lambda *args, **kwargs: None)
    _stub_module("tqdm", tqdm=lambda iterable, **kwargs: iterable)

    _stub_module("model")
    _stub_module("model.pretrained")
    _stub_module(
        "model.pretrained.model_factory",
        create_pretrained_model=lambda **kwargs: None,
        get_model_info=lambda name: {"description": f"stub:{name}"},
    )
    _stub_module(
        "model.pretrained.dataset",
        GalaxyPretrainedDataset=object,
        create_data_loaders=lambda *args, **kwargs: None,
        get_imagenet_transforms=lambda *args, **kwargs: None,
    )

    _stub_module(
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
    _stub_module("validation.benchmark", ModelBenchmark=object)
    _stub_module(
        "validation.visualization",
        ValidationPlotGenerator=object,
        ComparisonPlotGenerator=object,
        ModelAnalysisPlotter=object,
    )

    sys.modules.pop("validation.validator", None)
    return importlib.import_module("validation.validator")


def test_validate_dataset_delegates_to_runner_and_persistence(monkeypatch, tmp_path):
    validator_module = _load_validator_module()

    calls = {}

    def fake_run_validation_batches(**kwargs):
        calls["runner_kwargs"] = kwargs
        return {
            "y_true": validator_module.np.array([1, 0]),
            "y_pred": validator_module.np.array([1, 0]),
            "y_prob": validator_module.np.array([[0.1, 0.9], [0.8, 0.2]]),
        }

    class FakeMetricsCalculator:
        confusion_matrix = None
        roc_curve_data = None

        def __init__(self, class_names):
            calls["class_names"] = class_names

        def calculate_all_metrics(self, y_true, y_pred, y_prob):
            calls["metrics_input"] = (
                y_true.tolist(),
                y_pred.tolist(),
                y_prob.tolist(),
            )
            return {"accuracy": 0.9}

        def print_summary(self):
            calls["printed"] = True

        def save_metrics(self, path):
            path.write_text('{"accuracy": 0.9}', encoding="utf-8")

    class FakeLoadedModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

        def __call__(self, images):
            return FakeTensor([[0.1, 0.9], [0.8, 0.2]])

    class FakeValidator(validator_module.ModelValidator):
        def _load_model(self):
            return FakeLoadedModel()

    monkeypatch.setattr(
        validator_module,
        "run_validation_batches",
        fake_run_validation_batches,
        raising=False,
    )
    monkeypatch.setattr(validator_module, "ClassificationMetrics", FakeMetricsCalculator)
    monkeypatch.setattr(
        validator_module,
        "save_validation_outputs",
        lambda **kwargs: calls.setdefault("saved", kwargs),
        raising=False,
    )

    validator = FakeValidator(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        device="cpu",
        class_names=["Regular", "Peculiar"],
    )

    result = validator.validate_dataset(
        test_loader=[(FakeTensor([[1], [2]]), FakeTensor([1, 0]))],
        save_results=True,
        output_dir=tmp_path,
    )

    assert result["metrics"]["accuracy"] == 0.9
    assert result["true_labels"].tolist() == [1, 0]
    assert calls["runner_kwargs"]["test_loader"]
    assert calls["runner_kwargs"]["device"] == "cpu"
    assert calls["saved"]["output_dir"] == tmp_path


def test_validate_dataset_preserves_public_result_shape(monkeypatch):
    validator_module = _load_validator_module()

    class FakeMetricsCalculator:
        confusion_matrix = None
        roc_curve_data = None

        def __init__(self, class_names):
            self.class_names = class_names

        def calculate_all_metrics(self, y_true, y_pred, y_prob):
            return {"accuracy": 0.9}

        def print_summary(self):
            return None

    class FakeLoadedModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    class FakeValidator(validator_module.ModelValidator):
        def _load_model(self):
            return FakeLoadedModel()

    monkeypatch.setattr(
        validator_module,
        "run_validation_batches",
        lambda **kwargs: {
            "y_true": validator_module.np.array([1, 0]),
            "y_pred": validator_module.np.array([1, 0]),
            "y_prob": validator_module.np.array([[0.1, 0.9], [0.8, 0.2]]),
        },
        raising=False,
    )
    monkeypatch.setattr(validator_module, "ClassificationMetrics", FakeMetricsCalculator)

    validator = FakeValidator(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        device="cpu",
        class_names=["Regular", "Peculiar"],
    )

    result = validator.validate_dataset(test_loader="loader", save_results=False)

    assert sorted(result.keys()) == [
        "metrics",
        "metrics_calculator",
        "predictions",
        "probabilities",
        "true_labels",
    ]
