# Treino/CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduzir uma CLI Python unica para treino e um nucleo reutilizavel que torne `Treino/CLI` mais simples e testavel sem depender de dataset real ou GPU.

**Architecture:** A implementacao cria `model/training_core.py` para normalizacao e despacho de fluxo, e `model/training_cli.py` para parsing de subcomandos e codigos de saida. Os entrypoints Python existentes passam a expor funcoes reutilizaveis, e os wrappers shell viram delegadores finos para a nova CLI.

**Tech Stack:** Python 3.11, argparse, dataclasses, pytest, monkeypatch, shell wrappers Bash existentes

---

### Task 1: Criar o nucleo testavel de orquestracao

**Files:**
- Create: `model/training_core.py`
- Test: `tests/model/test_training_core.py`

- [ ] **Step 1: Write the failing test**

```python
from dataclasses import replace

from model.training_core import (
    TrainingDependencies,
    TrainingRequest,
    run_training,
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_training_core.py::test_run_training_dispatches_pretrained_flow -v`
Expected: FAIL com `ModuleNotFoundError` ou simbolos ausentes em `model.training_core`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class TrainingRequest:
    mode: str
    config_path: Optional[str]
    model_name: Optional[str]
    device: str
    seed: int
    validate_only: bool
    data_dir: Optional[str]
    batch_size: Optional[int]
    num_epochs: Optional[int]
    learning_rate: Optional[float]
    predict_samples: Optional[int]
    evaluate: bool
    resolved_device: Optional[str] = None


@dataclass(frozen=True)
class TrainingResult:
    success: bool
    message: str
    output_dir: Optional[str] = None
    artifacts_path: Optional[str] = None
    metrics_path: Optional[str] = None


@dataclass(frozen=True)
class TrainingDependencies:
    pretrained_runner: Callable[[TrainingRequest], dict]
    from_scratch_runner: Callable[[TrainingRequest], dict]
    cuda_available: Callable[[], bool]


def _resolve_device(device: str, cuda_available: Callable[[], bool]) -> str:
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    return device


def run_training(request: TrainingRequest, deps: TrainingDependencies) -> TrainingResult:
    normalized_request = TrainingRequest(
        **{
            **request.__dict__,
            "resolved_device": _resolve_device(request.device, deps.cuda_available),
        }
    )

    if normalized_request.mode == "pretrained":
        payload = deps.pretrained_runner(normalized_request)
    elif normalized_request.mode == "from-scratch":
        payload = deps.from_scratch_runner(normalized_request)
    else:
        raise ValueError(f"Unsupported training mode: {normalized_request.mode}")

    return TrainingResult(**payload)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/model/test_training_core.py::test_run_training_dispatches_pretrained_flow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/training_core.py tests/model/test_training_core.py
git commit -m "refactor: add training core dispatcher"
```

### Task 2: Cobrir validacao e fluxo `from-scratch` no core

**Files:**
- Modify: `model/training_core.py`
- Modify: `tests/model/test_training_core.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from model.training_core import (
    TrainingDependencies,
    TrainingRequest,
    run_training,
)


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/model/test_training_core.py -v`
Expected: FAIL porque `run_training()` ainda nao cobre o novo caso ou nao protege o modo invalido adequadamente

- [ ] **Step 3: Write minimal implementation**

```python
def _validate_request(request: TrainingRequest) -> None:
    valid_modes = {"pretrained", "from-scratch"}
    if request.mode not in valid_modes:
        raise ValueError(f"Unsupported training mode: {request.mode}")

    if request.mode == "pretrained" and not (request.config_path or request.model_name):
        raise ValueError("pretrained mode requires config_path or model_name")

    if request.mode == "from-scratch" and not request.data_dir:
        raise ValueError("from-scratch mode requires data_dir")


def run_training(request: TrainingRequest, deps: TrainingDependencies) -> TrainingResult:
    _validate_request(request)

    normalized_request = TrainingRequest(
        **{
            **request.__dict__,
            "resolved_device": _resolve_device(request.device, deps.cuda_available),
        }
    )

    if normalized_request.mode == "pretrained":
        payload = deps.pretrained_runner(normalized_request)
    else:
        payload = deps.from_scratch_runner(normalized_request)

    return TrainingResult(**payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/model/test_training_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/training_core.py tests/model/test_training_core.py
git commit -m "test: cover training core validation paths"
```

### Task 3: Extrair funcoes reutilizaveis dos entrypoints Python

**Files:**
- Modify: `model/pretrained/train_pretrained.py`
- Modify: `model/from_scratch/train_from_scratch.py`
- Modify: `tests/model/test_training_core.py`

- [ ] **Step 1: Write the failing tests**

```python
from model.training_core import (
    TrainingDependencies,
    TrainingRequest,
    run_training,
)


def test_run_training_uses_pretrained_entrypoint_adapter(monkeypatch):
    imported = {}

    def fake_pretrained_runner(request):
        imported["mode"] = request.mode
        return {
            "success": True,
            "message": "delegated",
            "output_dir": "results/models/pretrained",
            "artifacts_path": None,
            "metrics_path": None,
        }

    request = TrainingRequest(
        mode="pretrained",
        config_path="model/pretrained/configs/efficientnet_b0.json",
        model_name=None,
        device="cpu",
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

    run_training(request, deps)

    assert imported["mode"] == "pretrained"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/model/test_training_core.py::test_run_training_uses_pretrained_entrypoint_adapter -v`
Expected: FAIL se os entrypoints ainda nao expuserem adaptadores reutilizaveis ou se o core ainda depender de `main()`

- [ ] **Step 3: Write minimal implementation**

```python
# model/pretrained/train_pretrained.py
def run_pretrained_entry(
    *,
    config_path: str,
    device: str,
    seed: int,
    validate_only: bool,
) -> dict:
    setup_seeds(seed)
    resolved_device = setup_device(device)

    return {
        "success": True,
        "message": "pretrained flow executed",
        "output_dir": "results/models",
        "artifacts_path": None,
        "metrics_path": None,
    }


def main():
    args = parse_args()
    result = run_pretrained_entry(
        config_path=args.config,
        device=args.device,
        seed=args.seed,
        validate_only=args.validate_only,
    )
    print(result["message"])
```

```python
# model/from_scratch/train_from_scratch.py
def run_from_scratch_entry(
    *,
    data_dir: str,
    device: str,
    seed: int,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    model_name: str,
    predict_samples: int,
    evaluate: bool,
) -> dict:
    setup_seeds(seed)

    return {
        "success": True,
        "message": "from scratch flow executed",
        "output_dir": "model_artifacts",
        "artifacts_path": "model_artifacts/galaxy_cnn_artifacts.pth",
        "metrics_path": None,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/model/test_training_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/pretrained/train_pretrained.py model/from_scratch/train_from_scratch.py tests/model/test_training_core.py
git commit -m "refactor: expose reusable training entrypoints"
```

### Task 4: Criar a nova CLI Python

**Files:**
- Create: `model/training_cli.py`
- Test: `tests/model/test_training_cli.py`

- [ ] **Step 1: Write the failing tests**

```python
from model.training_cli import build_parser, main


def test_build_parser_accepts_pretrained_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        ["pretrained", "--config", "model/pretrained/configs/efficientnet_b0.json"]
    )

    assert args.command == "pretrained"
    assert args.config == "model/pretrained/configs/efficientnet_b0.json"


def test_main_returns_zero_on_success(monkeypatch, capsys):
    monkeypatch.setattr(
        "model.training_cli.run_training",
        lambda request, deps: type(
            "Result",
            (),
            {"success": True, "message": "ok", "output_dir": "results/models"},
        )(),
    )

    exit_code = main(
        ["pretrained", "--config", "model/pretrained/configs/efficientnet_b0.json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "ok" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/model/test_training_cli.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'model.training_cli'`

- [ ] **Step 3: Write minimal implementation**

```python
import argparse
import torch

from model.from_scratch.train_from_scratch import run_from_scratch_entry
from model.pretrained.train_pretrained import run_pretrained_entry
from model.training_core import TrainingDependencies, TrainingRequest, run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified training CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pretrained = subparsers.add_parser("pretrained")
    pretrained.add_argument("--config", required=True)
    pretrained.add_argument("--device", default="auto")
    pretrained.add_argument("--seed", type=int, default=42)
    pretrained.add_argument("--validate-only", action="store_true")

    from_scratch = subparsers.add_parser("from-scratch")
    from_scratch.add_argument("--data_dir", required=True)
    from_scratch.add_argument("--device", default="auto")
    from_scratch.add_argument("--seed", type=int, default=42)
    from_scratch.add_argument("--batch_size", type=int, default=32)
    from_scratch.add_argument("--num_epochs", type=int, default=10)
    from_scratch.add_argument("--learning_rate", type=float, default=0.001)
    from_scratch.add_argument("--model_name", default="galaxy_cnn")
    from_scratch.add_argument("--predict_samples", type=int, default=5)
    from_scratch.add_argument("--evaluate", action="store_true")

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    request = TrainingRequest(
        mode=args.command,
        config_path=getattr(args, "config", None),
        model_name=getattr(args, "model_name", None),
        device=args.device,
        seed=args.seed,
        validate_only=getattr(args, "validate_only", False),
        data_dir=getattr(args, "data_dir", None),
        batch_size=getattr(args, "batch_size", None),
        num_epochs=getattr(args, "num_epochs", None),
        learning_rate=getattr(args, "learning_rate", None),
        predict_samples=getattr(args, "predict_samples", None),
        evaluate=getattr(args, "evaluate", False),
    )

    deps = TrainingDependencies(
        pretrained_runner=lambda request: run_pretrained_entry(
            config_path=request.config_path,
            device=request.resolved_device or request.device,
            seed=request.seed,
            validate_only=request.validate_only,
        ),
        from_scratch_runner=lambda request: run_from_scratch_entry(
            data_dir=request.data_dir,
            device=request.resolved_device or request.device,
            seed=request.seed,
            batch_size=request.batch_size,
            num_epochs=request.num_epochs,
            learning_rate=request.learning_rate,
            model_name=request.model_name,
            predict_samples=request.predict_samples,
            evaluate=request.evaluate,
        ),
        cuda_available=torch.cuda.is_available,
    )

    result = run_training(request, deps)
    print(result.message)
    return 0 if result.success else 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/model/test_training_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add model/training_cli.py tests/model/test_training_cli.py
git commit -m "feat: add unified training cli"
```

### Task 5: Reduzir os wrappers shell a delegadores finos

**Files:**
- Modify: `utils/scripts/train_pretrained.sh`
- Modify: `utils/scripts/from_scratch.sh`
- Test: `tests/model/test_training_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_shell_wrappers_delegate_to_python_cli():
    pretrained_wrapper = Path("utils/scripts/train_pretrained.sh").read_text()
    from_scratch_wrapper = Path("utils/scripts/from_scratch.sh").read_text()

    assert "python model/training_cli.py pretrained" in pretrained_wrapper
    assert "python model/training_cli.py from-scratch" in from_scratch_wrapper
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_training_cli.py::test_shell_wrappers_delegate_to_python_cli -v`
Expected: FAIL porque os wrappers ainda contem validacao, prompts e montagem manual de comando

- [ ] **Step 3: Write minimal implementation**

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

python model/training_cli.py pretrained "$@"
```

```bash
#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

python model/training_cli.py from-scratch "$@"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/model/test_training_cli.py::test_shell_wrappers_delegate_to_python_cli -v`
Expected: PASS

- [ ] **Step 5: Run focused regression checks**

Run: `pytest tests/model/test_training_core.py tests/model/test_training_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add utils/scripts/train_pretrained.sh utils/scripts/from_scratch.sh tests/model/test_training_core.py tests/model/test_training_cli.py model/training_core.py model/training_cli.py model/pretrained/train_pretrained.py model/from_scratch/train_from_scratch.py
git commit -m "refactor: route training scripts through unified cli"
```

### Task 6: Fazer limpeza final e validar a entrada principal

**Files:**
- Modify: `model/training_cli.py`
- Modify: `model/training_core.py`
- Modify: `tests/model/test_training_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from model.training_cli import main


def test_main_returns_one_and_prints_error(monkeypatch, capsys):
    def fake_run_training(request, deps):
        return type(
            "Result",
            (),
            {"success": False, "message": "invalid configuration", "output_dir": None},
        )()

    monkeypatch.setattr("model.training_cli.run_training", fake_run_training)

    exit_code = main(
        ["pretrained", "--config", "model/pretrained/configs/efficientnet_b0.json"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid configuration" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/model/test_training_cli.py::test_main_returns_one_and_prints_error -v`
Expected: FAIL se a CLI ainda sempre retornar `0`

- [ ] **Step 3: Write minimal implementation**

```python
def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        request = _build_request(args)
        deps = _build_dependencies()
        result = run_training(request, deps)
    except ValueError as error:
        print(str(error))
        return 1

    print(result.message)
    return 0 if result.success else 1
```

- [ ] **Step 4: Run final checks**

Run: `pytest tests/model/test_training_core.py tests/model/test_training_cli.py -v`
Expected: PASS

Run: `python model/training_cli.py pretrained --help`
Expected: ajuda do subcomando `pretrained`

Run: `python model/training_cli.py from-scratch --help`
Expected: ajuda do subcomando `from-scratch`

- [ ] **Step 5: Commit**

```bash
git add model/training_cli.py model/training_core.py tests/model/test_training_cli.py
git commit -m "test: finalize training cli exit behavior"
```
