# Validator Runner + Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separar inferencia em lote e persistencia de artefatos do `ModelValidator`, mantendo a interface publica atual de `validate_dataset()`.

**Architecture:** A implementacao cria `validation/validation_runner.py` para executar a inferencia em lote e `validation/validation_persistence.py` para salvar metricas, previsoes e visualizacoes. `validation/validator.py` continua como fachada publica, mas passa a delegar a essas duas unidades internas.

**Tech Stack:** Python 3.11, pytest, monkeypatch, tmp_path, pathlib, json, numpy, torch, matplotlib

---

### Task 1: Criar o runner de inferencia em lote

**Files:**
- Create: `validation/validation_runner.py`
- Create: `tests/validation/test_validation_runner.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np

from validation.validation_runner import run_validation_batches


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


class FakeModel:
    def __call__(self, images):
        return FakeTensor([[0.1, 0.9], [0.8, 0.2]])


def test_run_validation_batches_collects_predictions_probabilities_and_labels(monkeypatch):
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_runner.py::test_run_validation_batches_collects_predictions_probabilities_and_labels -v`
Expected: FAIL com `ModuleNotFoundError` para `validation.validation_runner`

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np


def softmax_outputs(outputs):
    import torch
    return torch.softmax(outputs, dim=1)


def argmax_outputs(outputs):
    import torch
    return torch.argmax(outputs, dim=1)


def run_validation_batches(*, model, test_loader, device):
    all_predictions = []
    all_probabilities = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = softmax_outputs(outputs)
        predictions = argmax_outputs(outputs)

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_predictions),
        "y_prob": np.array(all_probabilities),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validation_runner.py::test_run_validation_batches_collects_predictions_probabilities_and_labels -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_runner.py tests/validation/test_validation_runner.py
git commit -m "refactor: add validation batch runner"
```

### Task 2: Cobrir uso de device no runner

**Files:**
- Modify: `validation/validation_runner.py`
- Modify: `tests/validation/test_validation_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_validation_batches_moves_images_and_labels_to_device(monkeypatch):
    images = FakeTensor([[1], [2]])
    labels = FakeTensor([1, 0])

    monkeypatch.setattr(
        "validation.validation_runner.softmax_outputs",
        lambda outputs: FakeTensor([[0.1, 0.9], [0.8, 0.2]]),
    )
    monkeypatch.setattr(
        "validation.validation_runner.argmax_outputs",
        lambda outputs: FakeTensor([1, 0]),
    )

    run_validation_batches(
        model=FakeModel(),
        test_loader=[(images, labels)],
        device="cuda",
    )

    assert images.device == "cuda"
    assert labels.device == "cuda"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_runner.py -v`
Expected: FAIL se o runner ainda nao mover corretamente os objetos para o device esperado

- [ ] **Step 3: Write minimal implementation**

```python
def run_validation_batches(*, model, test_loader, device):
    all_predictions = []
    all_probabilities = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = softmax_outputs(outputs)
        predictions = argmax_outputs(outputs)

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_predictions),
        "y_prob": np.array(all_probabilities),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_validation_runner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_runner.py tests/validation/test_validation_runner.py
git commit -m "test: cover validation runner device handling"
```

### Task 3: Criar a persistencia de resultados

**Files:**
- Create: `validation/validation_persistence.py`
- Create: `tests/validation/test_validation_persistence.py`

- [ ] **Step 1: Write the failing tests**

```python
import json

from validation.validation_persistence import save_validation_outputs


class FakeMetricsCalculator:
    confusion_matrix = [[1, 0], [0, 1]]
    roc_curve_data = {"fpr": [0, 1], "tpr": [0, 1]}
    roc_auc = 0.95

    def save_metrics(self, path):
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"accuracy": 0.9}, file)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_validation_persistence.py::test_save_validation_outputs_writes_metrics_and_predictions -v`
Expected: FAIL com `ModuleNotFoundError` para `validation.validation_persistence`

- [ ] **Step 3: Write minimal implementation**

```python
import json
from pathlib import Path


def save_validation_outputs(
    *,
    output_dir,
    model_name,
    model_path,
    class_names,
    y_true,
    y_pred,
    y_prob,
    metrics_calculator,
    generate_visualizations,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_calculator.save_metrics(output_dir / "metrics.json")

    payload = {
        "model_name": model_name,
        "model_path": model_path,
        "num_samples": len(y_true),
        "predictions": list(y_pred),
        "probabilities": list(y_prob),
        "true_labels": list(y_true),
        "class_names": class_names,
    }

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    generate_visualizations(output_dir=output_dir, metrics_calculator=metrics_calculator)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validation_persistence.py::test_save_validation_outputs_writes_metrics_and_predictions -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_persistence.py tests/validation/test_validation_persistence.py
git commit -m "refactor: add validation result persistence"
```

### Task 4: Cobrir chamada de visualizacao na persistencia

**Files:**
- Modify: `validation/validation_persistence.py`
- Modify: `tests/validation/test_validation_persistence.py`

- [ ] **Step 1: Write the failing test**

```python
def test_save_validation_outputs_calls_visualization_generator(tmp_path):
    calls = {}

    class FakeMetricsCalculator:
        confusion_matrix = [[1, 0], [0, 1]]
        roc_curve_data = {"fpr": [0, 1], "tpr": [0, 1]}
        roc_auc = 0.95

        def save_metrics(self, path):
            path.write_text('{"accuracy": 0.9}', encoding="utf-8")

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
        metrics_calculator=FakeMetricsCalculator(),
        generate_visualizations=fake_generate_visualizations,
    )

    assert calls["output_dir"] == tmp_path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_persistence.py -v`
Expected: FAIL se a persistencia ainda nao chamar o gerador com os argumentos corretos

- [ ] **Step 3: Write minimal implementation**

```python
def save_validation_outputs(
    *,
    output_dir,
    model_name,
    model_path,
    class_names,
    y_true,
    y_pred,
    y_prob,
    metrics_calculator,
    generate_visualizations,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_calculator.save_metrics(output_dir / "metrics.json")

    payload = {
        "model_name": model_name,
        "model_path": model_path,
        "num_samples": len(y_true),
        "predictions": list(y_pred),
        "probabilities": list(y_prob),
        "true_labels": list(y_true),
        "class_names": class_names,
    }

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    generate_visualizations(output_dir=output_dir, metrics_calculator=metrics_calculator)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_validation_persistence.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_persistence.py tests/validation/test_validation_persistence.py
git commit -m "test: cover validation persistence visualization hook"
```

### Task 5: Adaptar `validator.py` para delegar ao runner e a persistencia

**Files:**
- Modify: `validation/validator.py`
- Create: `tests/validation/test_validator.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_dataset_delegates_to_runner_and_persistence(monkeypatch, tmp_path):
    import validation.validator as validator_module

    calls = {}

    class FakeMetricsCalculator:
        def __init__(self, class_names):
            calls["class_names"] = class_names

        def calculate_all_metrics(self, y_true, y_pred, y_prob):
            calls["metrics_input"] = (y_true.tolist(), y_pred.tolist(), y_prob.tolist())
            return {"accuracy": 0.9}

        def print_summary(self):
            calls["printed"] = True

    class FakeValidator(validator_module.ModelValidator):
        def _load_model(self):
            return "fake-model"

    monkeypatch.setattr(
        validator_module,
        "run_validation_batches",
        lambda **kwargs: {
            "y_true": validator_module.np.array([1, 0]),
            "y_pred": validator_module.np.array([1, 0]),
            "y_prob": validator_module.np.array([[0.1, 0.9], [0.8, 0.2]]),
        },
    )
    monkeypatch.setattr(validator_module, "ClassificationMetrics", FakeMetricsCalculator)
    monkeypatch.setattr(
        validator_module,
        "save_validation_outputs",
        lambda **kwargs: calls.setdefault("saved", kwargs["output_dir"]),
    )

    validator = FakeValidator(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        device="cpu",
        class_names=["Regular", "Peculiar"],
    )

    result = validator.validate_dataset(test_loader="loader", save_results=True, output_dir=tmp_path)

    assert result["metrics"]["accuracy"] == 0.9
    assert calls["saved"] == tmp_path
    assert result["true_labels"].tolist() == [1, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validator.py::test_validate_dataset_delegates_to_runner_and_persistence -v`
Expected: FAIL porque `validator.py` ainda nao delega para `run_validation_batches` e `save_validation_outputs`

- [ ] **Step 3: Write minimal implementation**

```python
from validation.validation_runner import run_validation_batches
from validation.validation_persistence import save_validation_outputs


def validate_dataset(self, test_loader, save_results=True, output_dir=None):
    raw_outputs = run_validation_batches(
        model=self.model,
        test_loader=test_loader,
        device=self.device,
    )

    y_true = raw_outputs["y_true"]
    y_pred = raw_outputs["y_pred"]
    y_prob = raw_outputs["y_prob"]

    metrics_calculator = ClassificationMetrics(self.class_names)
    metrics = metrics_calculator.calculate_all_metrics(y_true, y_pred, y_prob)
    metrics_calculator.print_summary()

    if save_results:
        if output_dir is None:
            output_dir = Path("validation_results") / self.model_name
        else:
            output_dir = Path(output_dir)

        save_validation_outputs(
            output_dir=output_dir,
            model_name=self.model_name,
            model_path=str(self.model_path),
            class_names=self.class_names,
            y_true=y_true.tolist(),
            y_pred=y_pred.tolist(),
            y_prob=y_prob.tolist(),
            metrics_calculator=metrics_calculator,
            generate_visualizations=lambda **kwargs: self._generate_visualizations(
                kwargs["metrics_calculator"],
                kwargs["output_dir"],
            ),
        )

    return {
        "metrics": metrics,
        "predictions": y_pred,
        "probabilities": y_prob,
        "true_labels": y_true,
        "metrics_calculator": metrics_calculator,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validator.py::test_validate_dataset_delegates_to_runner_and_persistence -v`
Expected: PASS

- [ ] **Step 5: Run focused regression**

Run: `pytest tests/validation/test_validation_runner.py tests/validation/test_validation_persistence.py tests/validation/test_validator.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add validation/validator.py validation/validation_runner.py validation/validation_persistence.py tests/validation/test_validation_runner.py tests/validation/test_validation_persistence.py tests/validation/test_validator.py
git commit -m "refactor: split validator inference and persistence"
```

### Task 6: Fechar com regressao do modulo Validation

**Files:**
- Modify: `tests/validation/test_validator.py`
- Modify: `tests/validation/test_validation_core.py`
- Modify: `tests/validation/test_validation_cli.py`

- [ ] **Step 1: Write the failing regression test**

```python
def test_validate_dataset_preserves_public_result_shape(monkeypatch):
    import validation.validator as validator_module

    class FakeMetricsCalculator:
        def __init__(self, class_names):
            pass

        def calculate_all_metrics(self, y_true, y_pred, y_prob):
            return {"accuracy": 0.9}

        def print_summary(self):
            return None

    class FakeValidator(validator_module.ModelValidator):
        def _load_model(self):
            return "fake-model"

    monkeypatch.setattr(
        validator_module,
        "run_validation_batches",
        lambda **kwargs: {
            "y_true": validator_module.np.array([1, 0]),
            "y_pred": validator_module.np.array([1, 0]),
            "y_prob": validator_module.np.array([[0.1, 0.9], [0.8, 0.2]]),
        },
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validator.py::test_validate_dataset_preserves_public_result_shape -v`
Expected: FAIL se a extracao tiver alterado o contrato publico de retorno

- [ ] **Step 3: Write minimal implementation**

```python
return {
    "metrics": metrics,
    "predictions": y_pred,
    "probabilities": y_prob,
    "true_labels": y_true,
    "metrics_calculator": metrics_calculator,
}
```

- [ ] **Step 4: Run final checks**

Run: `pytest tests/validation/test_validation_runner.py tests/validation/test_validation_persistence.py tests/validation/test_validator.py tests/validation/test_validation_core.py tests/validation/test_validation_cli.py -v`
Expected: PASS

Run: `python validation/validation_cli.py benchmark --help`
Expected: ajuda do subcomando `benchmark`

- [ ] **Step 5: Commit**

```bash
git add validation/validator.py validation/validation_runner.py validation/validation_persistence.py tests/validation/test_validation_runner.py tests/validation/test_validation_persistence.py tests/validation/test_validator.py tests/validation/test_validation_core.py tests/validation/test_validation_cli.py
git commit -m "test: preserve validator public contract"
```
