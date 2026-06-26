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

from validation.validation_runner import run_validation_batches


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.device = None
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        moved = FakeTensor(self.values)
        moved.device = device
        return moved

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeModel:
    def __call__(self, images):
        return FakeTensor([[0.1, 0.9], [0.8, 0.2]])


def test_run_validation_batches_collects_predictions_probabilities_and_labels(
    monkeypatch,
):
    batches = [
        (FakeTensor([[1], [2]]), FakeTensor([1, 0])),
    ]

    monkeypatch.setattr(
        "validation.validation_runner.softmax_outputs",
        lambda outputs: FakeTensor([[0.1, 0.9], [0.8, 0.2]]),
    )
    monkeypatch.setattr(
        "validation.validation_runner.argmax_outputs",
        lambda outputs: FakeTensor([1, 0]),
    )

    result = run_validation_batches(
        model=FakeModel(),
        test_loader=batches,
        device="cpu",
    )

    assert result["y_true"].tolist() == [1, 0]
    assert result["y_pred"].tolist() == [1, 0]
    assert result["y_prob"].tolist() == [[0.1, 0.9], [0.8, 0.2]]


def test_run_validation_batches_moves_images_and_labels_to_device(monkeypatch):
    captured = {}
    images = FakeTensor([[1], [2]])
    labels = FakeTensor([1, 0])

    class RecordingModel:
        def __call__(self, batch_images):
            captured["images_device"] = batch_images.device
            return FakeTensor([[0.1, 0.9], [0.8, 0.2]])

    monkeypatch.setattr(
        "validation.validation_runner.softmax_outputs",
        lambda outputs: FakeTensor([[0.1, 0.9], [0.8, 0.2]]),
    )
    monkeypatch.setattr(
        "validation.validation_runner.argmax_outputs",
        lambda outputs: FakeTensor([1, 0]),
    )

    run_validation_batches(
        model=RecordingModel(),
        test_loader=[(images, labels)],
        device="cuda",
    )

    assert images.to_calls == ["cuda"]
    assert labels.to_calls == ["cuda"]
    assert captured["images_device"] == "cuda"
