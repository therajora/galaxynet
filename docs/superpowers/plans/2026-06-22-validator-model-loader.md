# Validator Model Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extrair a carga de modelo de `validation/validator.py` para um loader reutilizavel, preservando a interface publica do `ModelValidator`.

**Architecture:** A implementacao cria `validation/model_loader.py` para validar o caminho do checkpoint, criar a arquitetura e restaurar o `state_dict`. `validation/validator.py` passa a delegar `_load_model()` ao novo loader, mantendo `to(device)` e `eval()` na inicializacao.

**Tech Stack:** Python 3.11, pytest, monkeypatch, pathlib, torch, stubs para model factory e checkpoint

---

### Task 1: Criar o loader de modelo com checkpoint usando `model_state_dict`

**Files:**
- Create: `validation/model_loader.py`
- Create: `tests/validation/test_model_loader.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from validation.model_loader import load_trained_model


class FakeModel:
    def __init__(self):
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = state_dict


def test_load_trained_model_uses_model_state_dict(monkeypatch, tmp_path):
    model_path = tmp_path / "best_model.pth"
    model_path.write_text("checkpoint", encoding="utf-8")

    fake_model = FakeModel()
    calls = {}

    monkeypatch.setattr(
        "validation.model_loader.create_pretrained_model",
        lambda **kwargs: calls.setdefault("model_kwargs", kwargs) or fake_model,
    )
    monkeypatch.setattr(
        "validation.model_loader.torch.load",
        lambda path, map_location: calls.setdefault(
            "torch_load", {"path": path, "map_location": map_location}
        ) or {"model_state_dict": {"weight": [1, 2, 3]}},
    )

    model = load_trained_model(
        model_name="efficientnet_b0",
        model_path=model_path,
        num_classes=2,
        device="cpu",
    )

    assert model is fake_model
    assert fake_model.loaded_state_dict == {"weight": [1, 2, 3]}
    assert calls["torch_load"]["path"] == model_path
    assert calls["torch_load"]["map_location"] == "cpu"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_model_loader.py::test_load_trained_model_uses_model_state_dict -v`
Expected: FAIL com `ModuleNotFoundError` para `validation.model_loader`

- [ ] **Step 3: Write minimal implementation**

```python
from pathlib import Path

import torch

from model.pretrained.model_factory import create_pretrained_model


def load_trained_model(*, model_name, model_path, num_classes, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = create_pretrained_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict)
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_model_loader.py::test_load_trained_model_uses_model_state_dict -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/model_loader.py tests/validation/test_model_loader.py
git commit -m "refactor: add trained model loader"
```

### Task 2: Cobrir checkpoint com `state_dict` direto e arquivo ausente

**Files:**
- Modify: `validation/model_loader.py`
- Modify: `tests/validation/test_model_loader.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest


def test_load_trained_model_accepts_direct_state_dict(monkeypatch, tmp_path):
    model_path = tmp_path / "best_model.pth"
    model_path.write_text("checkpoint", encoding="utf-8")

    fake_model = FakeModel()
    monkeypatch.setattr(
        "validation.model_loader.create_pretrained_model",
        lambda **kwargs: fake_model,
    )
    monkeypatch.setattr(
        "validation.model_loader.torch.load",
        lambda path, map_location: {"weight": [9, 8, 7]},
    )

    model = load_trained_model(
        model_name="efficientnet_b0",
        model_path=model_path,
        num_classes=2,
        device="cpu",
    )

    assert model.loaded_state_dict == {"weight": [9, 8, 7]}


def test_load_trained_model_raises_when_file_does_not_exist(tmp_path):
    missing_path = tmp_path / "missing_model.pth"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_trained_model(
            model_name="efficientnet_b0",
            model_path=missing_path,
            num_classes=2,
            device="cpu",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_model_loader.py -v`
Expected: FAIL porque o loader ainda exige apenas `model_state_dict`

- [ ] **Step 3: Write minimal implementation**

```python
def load_trained_model(*, model_name, model_path, num_classes, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = create_pretrained_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_model_loader.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/model_loader.py tests/validation/test_model_loader.py
git commit -m "test: cover validator model loader checkpoint formats"
```

### Task 3: Cobrir criação da arquitetura com os argumentos corretos

**Files:**
- Modify: `tests/validation/test_model_loader.py`

- [ ] **Step 1: Write the failing test**

```python
def test_load_trained_model_creates_architecture_with_pretrained_disabled(monkeypatch, tmp_path):
    model_path = tmp_path / "best_model.pth"
    model_path.write_text("checkpoint", encoding="utf-8")

    fake_model = FakeModel()
    calls = {}

    monkeypatch.setattr(
        "validation.model_loader.create_pretrained_model",
        lambda **kwargs: calls.setdefault("model_kwargs", kwargs) or fake_model,
    )
    monkeypatch.setattr(
        "validation.model_loader.torch.load",
        lambda path, map_location: {"model_state_dict": {"weight": [1]}},
    )

    load_trained_model(
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_model_loader.py::test_load_trained_model_creates_architecture_with_pretrained_disabled -v`
Expected: FAIL se o loader ainda nao repassar corretamente os argumentos

- [ ] **Step 3: Write minimal implementation**

```python
model = create_pretrained_model(
    model_name=model_name,
    num_classes=num_classes,
    pretrained=False,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_model_loader.py::test_load_trained_model_creates_architecture_with_pretrained_disabled -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_model_loader.py validation/model_loader.py
git commit -m "test: cover model loader architecture creation"
```

### Task 4: Adaptar `validator.py` para delegar a carga ao loader

**Files:**
- Modify: `validation/validator.py`
- Modify: `tests/validation/test_validator.py`

- [ ] **Step 1: Write the failing test**

```python
def test_model_validator_loads_model_via_model_loader(monkeypatch, tmp_path):
    validator_module = _load_validator_module()

    calls = {}

    class FakeLoadedModel:
        def to(self, device):
            calls["to_device"] = device
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    monkeypatch.setattr(
        validator_module,
        "load_trained_model",
        lambda **kwargs: calls.setdefault("loader_kwargs", kwargs) or FakeLoadedModel(),
        raising=False,
    )

    validator = validator_module.ModelValidator(
        model_name="efficientnet_b0",
        model_path=tmp_path / "best_model.pth",
        device="cpu",
        class_names=["Regular", "Peculiar"],
    )

    assert calls["loader_kwargs"]["model_name"] == "efficientnet_b0"
    assert calls["loader_kwargs"]["num_classes"] == 2
    assert calls["to_device"] == "cpu"
    assert calls["eval_called"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validator.py::test_model_validator_loads_model_via_model_loader -v`
Expected: FAIL porque `_load_model()` ainda usa a lógica inline

- [ ] **Step 3: Write minimal implementation**

```python
from validation.model_loader import load_trained_model


def _load_model(self) -> nn.Module:
    return load_trained_model(
        model_name=self.model_name,
        model_path=self.model_path,
        num_classes=self.num_classes,
        device=self.device,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validator.py::test_model_validator_loads_model_via_model_loader -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validator.py tests/validation/test_validator.py validation/model_loader.py tests/validation/test_model_loader.py
git commit -m "refactor: delegate validator model loading"
```

### Task 5: Fechar regressao focal da carga de modelo

**Files:**
- Modify: `tests/validation/test_model_loader.py`
- Modify: `tests/validation/test_validator.py`

- [ ] **Step 1: Write the failing regression test**

```python
def test_model_validator_preserves_public_initialization_flow(monkeypatch, tmp_path):
    validator_module = _load_validator_module()

    calls = {}

    class FakeLoadedModel:
        def to(self, device):
            calls["to_device"] = device
            return self

        def eval(self):
            calls["eval_called"] = True
            return self

    monkeypatch.setattr(
        validator_module,
        "load_trained_model",
        lambda **kwargs: FakeLoadedModel(),
        raising=False,
    )

    validator = validator_module.ModelValidator(
        model_name="efficientnet_b0",
        model_path=tmp_path / "best_model.pth",
        device="cpu",
        class_names=["Regular", "Peculiar"],
    )

    assert validator.model_name == "efficientnet_b0"
    assert validator.class_names == ["Regular", "Peculiar"]
    assert calls["to_device"] == "cpu"
    assert calls["eval_called"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validator.py::test_model_validator_preserves_public_initialization_flow -v`
Expected: FAIL se a extracao tiver mudado o fluxo de inicializacao

- [ ] **Step 3: Write minimal implementation**

```python
self.model = self._load_model()
self.model.to(self.device)
self.model.eval()
```

- [ ] **Step 4: Run final checks**

Run: `pytest tests/validation/test_model_loader.py tests/validation/test_validator.py tests/validation/test_validation_runner.py tests/validation/test_validation_persistence.py tests/validation/test_validation_core.py tests/validation/test_validation_cli.py -v`
Expected: PASS

Run: `python validation/validation_cli.py benchmark --help`
Expected: ajuda do subcomando `benchmark`

- [ ] **Step 5: Commit**

```bash
git add validation/model_loader.py validation/validator.py tests/validation/test_model_loader.py tests/validation/test_validator.py tests/validation/test_validation_runner.py tests/validation/test_validation_persistence.py tests/validation/test_validation_core.py tests/validation/test_validation_cli.py
git commit -m "test: preserve validator initialization flow"
```
