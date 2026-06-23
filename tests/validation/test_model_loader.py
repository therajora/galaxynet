import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest


class FakeModel:
    def __init__(self):
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


def _stub_module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    module_names = [
        "validation",
        "validation.model_loader",
        "model",
        "model.pretrained",
        "model.pretrained.model_factory",
        "torch",
    ]
    original_modules = {
        name: sys.modules.get(name)
        for name in module_names
    }

    yield

    for name, module in original_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _import_model_loader_module():
    validation_package = ModuleType("validation")
    validation_package.__path__ = [
        str(Path(__file__).resolve().parents[2] / "validation")
    ]
    sys.modules["validation"] = validation_package

    _stub_module("torch", load=lambda *args, **kwargs: {})
    _stub_module("model")
    _stub_module("model.pretrained")
    _stub_module(
        "model.pretrained.model_factory",
        create_pretrained_model=lambda **kwargs: None,
    )

    sys.modules.pop("validation.model_loader", None)
    return importlib.import_module("validation.model_loader")


def test_load_trained_model_uses_model_state_dict(monkeypatch, tmp_path):
    model_loader = _import_model_loader_module()
    model_path = tmp_path / "best_model.pth"
    model_path.write_text("checkpoint", encoding="utf-8")

    fake_model = FakeModel()
    calls = {}

    def fake_create_pretrained_model(**kwargs):
        calls["model_kwargs"] = kwargs
        return fake_model

    def fake_torch_load(path, map_location):
        calls["torch_load"] = {"path": path, "map_location": map_location}
        return {"model_state_dict": {"weight": [1, 2, 3]}}

    monkeypatch.setattr(
        model_loader,
        "create_pretrained_model",
        fake_create_pretrained_model,
    )
    monkeypatch.setattr(
        model_loader.torch,
        "load",
        fake_torch_load,
    )

    model = model_loader.load_trained_model(
        model_name="efficientnet_b0",
        model_path=model_path,
        num_classes=2,
        device="cpu",
    )

    assert model is fake_model
    assert fake_model.loaded_state_dict == {"weight": [1, 2, 3]}
    assert calls["torch_load"]["path"] == model_path
    assert calls["torch_load"]["map_location"] == "cpu"


def test_load_trained_model_accepts_direct_state_dict(monkeypatch, tmp_path):
    model_loader = _import_model_loader_module()
    model_path = tmp_path / "best_model.pth"
    model_path.write_text("checkpoint", encoding="utf-8")

    fake_model = FakeModel()

    monkeypatch.setattr(
        model_loader,
        "create_pretrained_model",
        lambda **kwargs: fake_model,
    )
    monkeypatch.setattr(
        model_loader.torch,
        "load",
        lambda path, map_location: {"weight": [9, 8, 7]},
    )

    model = model_loader.load_trained_model(
        model_name="efficientnet_b0",
        model_path=model_path,
        num_classes=2,
        device="cpu",
    )

    assert model is fake_model
    assert fake_model.loaded_state_dict == {"weight": [9, 8, 7]}


def test_load_trained_model_raises_when_file_does_not_exist(tmp_path):
    model_loader = _import_model_loader_module()
    missing_path = tmp_path / "missing_model.pth"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        model_loader.load_trained_model(
            model_name="efficientnet_b0",
            model_path=missing_path,
            num_classes=2,
            device="cpu",
        )


def test_load_trained_model_creates_architecture_with_pretrained_disabled(
    monkeypatch,
    tmp_path,
):
    model_loader = _import_model_loader_module()
    model_path = tmp_path / "best_model.pth"
    model_path.write_text("checkpoint", encoding="utf-8")

    fake_model = FakeModel()
    calls = {}

    def fake_create_pretrained_model(**kwargs):
        calls["model_kwargs"] = kwargs
        return fake_model

    monkeypatch.setattr(
        model_loader,
        "create_pretrained_model",
        fake_create_pretrained_model,
    )
    monkeypatch.setattr(
        model_loader.torch,
        "load",
        lambda path, map_location: {"model_state_dict": {"weight": [1]}},
    )

    model_loader.load_trained_model(
        model_name="resnet50",
        model_path=model_path,
        num_classes=3,
        device="cuda",
    )

    assert calls["model_kwargs"] == {
        "model_name": "resnet50",
        "num_classes": 3,
        "pretrained": False,
    }
