import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from model.training_core import (
    TrainingDependencies,
    TrainingRequest,
    run_training,
)


def _stub_module(monkeypatch, name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _load_module_from_path(monkeypatch, module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_train_pretrained_module(monkeypatch):
    torch_module = _stub_module(
        monkeypatch,
        "torch",
        manual_seed=lambda seed: None,
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False, manual_seed=lambda seed: None),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(deterministic=False, benchmark=False)
        ),
    )
    _stub_module(monkeypatch, "numpy", random=SimpleNamespace(seed=lambda seed: None))
    _stub_module(monkeypatch, "datasets", load_dataset=lambda *args, **kwargs: None)
    _stub_module(
        monkeypatch,
        "PIL",
        Image=SimpleNamespace(
            Image=type("FakeImage", (), {}),
            fromarray=lambda image: image,
        ),
    )
    _stub_module(monkeypatch, "torch.utils")
    _stub_module(monkeypatch, "torch.utils.data", Dataset=object)
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.utils", sys.modules["torch.utils"])
    monkeypatch.setitem(sys.modules, "torch.utils.data", sys.modules["torch.utils.data"])

    pretrained_package = _stub_module(monkeypatch, "model.pretrained")
    pretrained_package.__path__ = []
    _stub_module(monkeypatch, "model.datasets", create_huggingface_dataset=lambda *a, **k: None)
    _stub_module(monkeypatch, "model.pretrained.config_manager", ConfigManager=object)
    _stub_module(
        monkeypatch,
        "model.pretrained.dataset",
        GalaxyPretrainedDataset=object,
        create_data_loaders=lambda **kwargs: None,
        get_imagenet_transforms=lambda image_size: None,
    )
    _stub_module(
        monkeypatch,
        "model.pretrained.model_factory",
        create_pretrained_model=lambda **kwargs: None,
        print_model_summary=lambda *args, **kwargs: None,
        count_parameters=lambda model: None,
    )
    _stub_module(monkeypatch, "model.pretrained.trainer", PretrainedTrainer=object)

    return _load_module_from_path(
        monkeypatch,
        "tests.fake_train_pretrained",
        "/workspace/model/pretrained/train_pretrained.py",
    )


def _load_train_from_scratch_module(monkeypatch):
    torch_module = _stub_module(
        monkeypatch,
        "torch",
        manual_seed=lambda seed: None,
        device=lambda name: name,
        save=lambda payload, path: None,
        cuda=SimpleNamespace(
            is_available=lambda: False,
            manual_seed=lambda seed: None,
            manual_seed_all=lambda seed: None,
        ),
    )
    _stub_module(monkeypatch, "numpy", random=SimpleNamespace(seed=lambda seed: None))
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    from_scratch_package = _stub_module(monkeypatch, "model.from_scratch")
    from_scratch_package.__path__ = []
    _stub_module(
        monkeypatch,
        "model.from_scratch.dataset",
        GalaxyDataset=object,
        calculate_mean_std=lambda dataset: None,
        create_transforms=lambda **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "model.from_scratch.model",
        create_model=lambda **kwargs: None,
        count_parameters=lambda model: None,
    )
    _stub_module(
        monkeypatch,
        "model.from_scratch.trainer",
        GalaxyTrainer=object,
        create_data_loaders=lambda **kwargs: None,
    )
    _stub_module(
        monkeypatch,
        "model.from_scratch.predictor",
        GalaxyPredictor=object,
    )

    return _load_module_from_path(
        monkeypatch,
        "tests.fake_train_from_scratch",
        "/workspace/model/from_scratch/train_from_scratch.py",
    )


def test_run_training_dispatches_pretrained_flow():
    calls = []

    def fake_pretrained_runner(request):
        calls.append(request)
        return {
            "success": True,
            "message": "ok",
            "output_dir": "results/models/pretrained",
            "artifacts_path": None,
            "metrics_path": None,
        }

    request = TrainingRequest(
        mode="pretrained",
        config_path="model/pretrained/configs/efficientnet_b0.json",
        model_name=None,
        device="auto",
        seed=42,
        validate_only=False,
        data_dir=None,
        batch_size=None,
        num_epochs=None,
        learning_rate=None,
        predict_samples=None,
        evaluate=False,
    )

    deps = TrainingDependencies(
        pretrained_runner=fake_pretrained_runner,
        from_scratch_runner=lambda request: None,
        cuda_available=lambda: False,
    )

    result = run_training(request, deps)

    assert result.success is True
    assert result.output_dir == "results/models/pretrained"
    assert calls[0].resolved_device == "cpu"


def test_run_training_dispatches_from_scratch_flow():
    calls = []

    def fake_from_scratch_runner(request):
        calls.append(request)
        return {
            "success": True,
            "message": "from scratch ok",
            "output_dir": "results/models/from-scratch",
            "artifacts_path": "results/models/from-scratch/artifacts.pth",
            "metrics_path": None,
        }

    request = TrainingRequest(
        mode="from-scratch",
        config_path=None,
        model_name="galaxy_cnn",
        device="cpu",
        seed=7,
        validate_only=False,
        data_dir="data/complete_sdss",
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        predict_samples=3,
        evaluate=False,
    )

    deps = TrainingDependencies(
        pretrained_runner=lambda request: None,
        from_scratch_runner=fake_from_scratch_runner,
        cuda_available=lambda: False,
    )

    result = run_training(request, deps)

    assert result.success is True
    assert calls[0].resolved_device == "cpu"


def test_run_training_rejects_invalid_mode():
    request = TrainingRequest(
        mode="benchmark",
        config_path=None,
        model_name=None,
        device="cpu",
        seed=1,
        validate_only=False,
        data_dir=None,
        batch_size=None,
        num_epochs=None,
        learning_rate=None,
        predict_samples=None,
        evaluate=False,
    )

    deps = TrainingDependencies(
        pretrained_runner=lambda request: None,
        from_scratch_runner=lambda request: None,
        cuda_available=lambda: False,
    )

    with pytest.raises(ValueError, match="Unsupported training mode"):
        run_training(request, deps)


def test_run_training_requires_pretrained_source():
    request = TrainingRequest(
        mode="pretrained",
        config_path=None,
        model_name=None,
        device="cpu",
        seed=1,
        validate_only=False,
        data_dir=None,
        batch_size=None,
        num_epochs=None,
        learning_rate=None,
        predict_samples=None,
        evaluate=False,
    )

    deps = TrainingDependencies(
        pretrained_runner=lambda request: None,
        from_scratch_runner=lambda request: None,
        cuda_available=lambda: False,
    )

    with pytest.raises(
        ValueError, match="pretrained mode requires config_path or model_name"
    ):
        run_training(request, deps)


def test_run_training_requires_data_dir_for_from_scratch():
    request = TrainingRequest(
        mode="from-scratch",
        config_path=None,
        model_name="galaxy_cnn",
        device="cpu",
        seed=1,
        validate_only=False,
        data_dir=None,
        batch_size=32,
        num_epochs=10,
        learning_rate=0.001,
        predict_samples=3,
        evaluate=False,
    )

    deps = TrainingDependencies(
        pretrained_runner=lambda request: None,
        from_scratch_runner=lambda request: None,
        cuda_available=lambda: False,
    )

    with pytest.raises(ValueError, match="from-scratch mode requires data_dir"):
        run_training(request, deps)


def test_run_pretrained_entry_returns_validation_result(monkeypatch):
    train_pretrained = _load_train_pretrained_module(monkeypatch)

    calls = {"seed": None, "config_path": None, "printed": False}

    class FakeConfigManager:
        def __init__(self, config_path):
            calls["config_path"] = config_path

        def validate_config(self):
            return []

        def print_config(self):
            calls["printed"] = True

    monkeypatch.setattr(
        train_pretrained, "setup_seeds", lambda seed: calls.__setitem__("seed", seed)
    )
    monkeypatch.setattr(train_pretrained, "ConfigManager", FakeConfigManager)

    result = train_pretrained.run_pretrained_entry(
        config_path="model/pretrained/configs/efficientnet_b0.json",
        device="cpu",
        seed=123,
        validate_only=True,
    )

    assert result == {
        "success": True,
        "message": "Validation completed successfully!",
        "output_dir": None,
        "artifacts_path": None,
        "metrics_path": None,
    }
    assert calls["seed"] == 123
    assert calls["config_path"] == "model/pretrained/configs/efficientnet_b0.json"
    assert calls["printed"] is True


def test_run_from_scratch_entry_returns_training_summary(monkeypatch):
    train_from_scratch = _load_train_from_scratch_module(monkeypatch)

    calls = {"seed": None, "torch_save_path": None, "predictor_used": False}

    class FakeDataset:
        classes = ["spiral", "elliptical"]

        def __init__(self, img_dir, metadata_path, transform):
            self.img_dir = img_dir
            self.metadata_path = metadata_path
            self.transform = transform

        def __len__(self):
            return 4

        def get_class_distribution(self):
            return {"spiral": 2, "elliptical": 2}

    class FakeTrainer:
        def __init__(self, *args, **kwargs):
            self.save_dir = "model_artifacts/version_001"

        def train(self, **kwargs):
            return {"loss": [0.1]}

        def plot_training_history(self, save_plot=True):
            return None

    class FakePredictor:
        def __init__(self, *args, **kwargs):
            calls["predictor_used"] = True

    monkeypatch.setattr(
        train_from_scratch, "setup_seeds", lambda seed: calls.__setitem__("seed", seed)
    )
    monkeypatch.setattr(train_from_scratch.os.path, "exists", lambda path: True)
    monkeypatch.setattr(train_from_scratch.os, "makedirs", lambda path, exist_ok: None)
    monkeypatch.setattr(
        train_from_scratch.torch.cuda, "is_available", lambda: False
    )
    monkeypatch.setattr(
        train_from_scratch,
        "create_transforms",
        lambda **kwargs: ("train_transform", "viz_transform"),
    )
    monkeypatch.setattr(train_from_scratch, "GalaxyDataset", FakeDataset)
    monkeypatch.setattr(
        train_from_scratch, "calculate_mean_std", lambda dataset: ([0.1], [0.2])
    )
    monkeypatch.setattr(
        train_from_scratch,
        "create_data_loaders",
        lambda **kwargs: ("train_loader", "val_loader"),
    )
    monkeypatch.setattr(
        train_from_scratch, "create_model", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(
        train_from_scratch,
        "count_parameters",
        lambda model: {"total": 10, "trainable": 8},
    )
    monkeypatch.setattr(train_from_scratch, "GalaxyTrainer", FakeTrainer)
    monkeypatch.setattr(train_from_scratch, "GalaxyPredictor", FakePredictor)
    monkeypatch.setattr(
        train_from_scratch.torch,
        "save",
        lambda payload, path: calls.__setitem__("torch_save_path", path),
    )

    result = train_from_scratch.run_from_scratch_entry(
        data_dir="data/complete_sdss",
        device="cpu",
        seed=7,
        batch_size=16,
        num_epochs=2,
        learning_rate=0.01,
        model_name="galaxy_cnn",
        predict_samples=0,
        evaluate=False,
    )

    assert result == {
        "success": True,
        "message": "Training completed successfully!",
        "output_dir": "model_artifacts/version_001",
        "artifacts_path": "model_artifacts/version_001/galaxy_cnn_artifacts.pth",
        "metrics_path": None,
    }
    assert calls["seed"] == 7
    assert calls["torch_save_path"] == "model_artifacts/version_001/galaxy_cnn_artifacts.pth"
    assert calls["predictor_used"] is False
