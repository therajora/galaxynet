# Validation CLI + Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduzir uma CLI Python unica para `Validation` e um core reutilizavel que torne os fluxos `single`, `benchmark` e `find` mais simples e testaveis sem depender de dados ou modelos reais.

**Architecture:** A implementacao cria `validation/validation_core.py` para validacao, normalizacao e despacho dos modos, e `validation/validation_cli.py` para parsing e codigos de saida. `validation/validate_models.py` vira um delegador fino, enquanto `validation/benchmark.py` e `validation/validator.py` permanecem como componentes reutilizados pelo core.

**Tech Stack:** Python 3.11, argparse, dataclasses, pathlib, pytest, monkeypatch, subprocess, torch

---

### Task 1: Criar o core de validacao reutilizavel

**Files:**
- Create: `validation/validation_core.py`
- Create: `tests/validation/test_validation_core.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_core.py::test_run_validation_dispatches_find_mode -v`
Expected: FAIL com `ModuleNotFoundError` para `validation.validation_core`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ValidationRequest:
    mode: str
    model_name: Optional[str]
    model_path: Optional[str]
    data_dir: str
    output_dir: Optional[str]
    device: str
    results_dir: str
    resolved_device: Optional[str] = None


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    message: str
    output_dir: Optional[str] = None
    results_path: Optional[str] = None
    model_paths: Optional[dict[str, str]] = None


@dataclass(frozen=True)
class ValidationDependencies:
    validate_single_runner: Callable[[ValidationRequest], dict]
    benchmark_runner: Callable[[ValidationRequest], dict]
    find_models_runner: Callable[[str], dict[str, str]]
    cuda_available: Callable[[], bool]


def _resolve_device(device: str, cuda_available: Callable[[], bool]) -> str:
    if device == "auto":
        return "cuda" if cuda_available() else "cpu"
    return device


def run_validation(request: ValidationRequest, deps: ValidationDependencies) -> ValidationResult:
    normalized_request = ValidationRequest(
        **{
            **request.__dict__,
            "resolved_device": _resolve_device(request.device, deps.cuda_available),
        }
    )

    if normalized_request.mode == "find":
        model_paths = deps.find_models_runner(normalized_request.results_dir)
        return ValidationResult(
            success=True,
            message="Models found",
            output_dir=None,
            results_path=None,
            model_paths=model_paths,
        )

    raise ValueError(f"Unsupported validation mode: {normalized_request.mode}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validation_core.py::test_run_validation_dispatches_find_mode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_core.py tests/validation/test_validation_core.py
git commit -m "refactor: add validation core dispatcher"
```

### Task 2: Cobrir validacao de argumentos e os modos `single` e `benchmark`

**Files:**
- Modify: `validation/validation_core.py`
- Modify: `tests/validation/test_validation_core.py`

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from validation.validation_core import (
    ValidationDependencies,
    ValidationRequest,
    run_validation,
)


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
        find_models_runner=lambda results_dir: {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        cuda_available=lambda: False,
    )

    result = run_validation(request, deps)

    assert result.success is True
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

    with pytest.raises(ValueError, match="single mode requires model_name and model_path"):
        run_validation(request, deps)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_validation_core.py -v`
Expected: FAIL porque o core ainda nao valida os modos nem despacha `single` e `benchmark`

- [ ] **Step 3: Write minimal implementation**

```python
def _validate_request(request: ValidationRequest) -> None:
    valid_modes = {"single", "benchmark", "find"}
    if request.mode not in valid_modes:
        raise ValueError(f"Unsupported validation mode: {request.mode}")

    if request.mode == "single" and not (request.model_name and request.model_path):
        raise ValueError("single mode requires model_name and model_path")


def run_validation(request: ValidationRequest, deps: ValidationDependencies) -> ValidationResult:
    _validate_request(request)

    normalized_request = ValidationRequest(
        **{
            **request.__dict__,
            "resolved_device": _resolve_device(request.device, deps.cuda_available),
        }
    )

    if normalized_request.mode == "single":
        payload = deps.validate_single_runner(normalized_request)
    elif normalized_request.mode == "benchmark":
        payload = deps.benchmark_runner(normalized_request)
    else:
        model_paths = deps.find_models_runner(normalized_request.results_dir)
        payload = {
            "success": True,
            "message": "Models found",
            "output_dir": None,
            "results_path": None,
            "model_paths": model_paths,
        }

    return ValidationResult(**payload)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_validation_core.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_core.py tests/validation/test_validation_core.py
git commit -m "test: cover validation core modes"
```

### Task 3: Expor funcoes reutilizaveis de validacao e benchmark

**Files:**
- Modify: `validation/validate_models.py`
- Test: `tests/validation/test_validation_core.py`

- [ ] **Step 1: Write the failing tests**

```python
from types import SimpleNamespace


def test_validate_single_entry_returns_structured_result(monkeypatch):
    import validation.validate_models as validate_models

    class FakeValidator:
        def __init__(self, model_name, model_path, device, class_names):
            self.model_name = model_name

        def validate_dataset(self, test_loader, save_results, output_dir):
            return {"metrics": {"accuracy": 0.9}}

    monkeypatch.setattr(validate_models, "ModelValidator", FakeValidator)
    monkeypatch.setattr(validate_models, "build_test_loader", lambda data_dir: "test_loader")
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
    monkeypatch.setattr(validate_models, "build_test_loader", lambda data_dir: "test_loader")
    monkeypatch.setattr(validate_models, "resolve_device", lambda device: "cpu")

    result = validate_models.run_benchmark_entry(
        model_paths={"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        data_dir="data/complete_sdss",
        output_dir="benchmark_results",
        device="auto",
    )

    assert result["success"] is True
    assert result["results_path"] == "benchmark_results/benchmark_comparison.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_validation_core.py::test_validate_single_entry_returns_structured_result tests/validation/test_validation_core.py::test_benchmark_entry_returns_structured_result -v`
Expected: FAIL porque as funcoes reutilizaveis ainda nao existem

- [ ] **Step 3: Write minimal implementation**

```python
def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_test_loader(data_dir: str):
    transforms = get_imagenet_transforms(224, is_train=False)
    test_dataset = GalaxyPretrainedDataset(
        img_dir=data_dir,
        metadata_path=None,
        transform=transforms,
    )
    return torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )


def validate_single_entry(*, model_name: str, model_path: str, data_dir: str, output_dir: str, device: str) -> dict:
    resolved_device = resolve_device(device)
    test_loader = build_test_loader(data_dir)
    validator = ModelValidator(
        model_name=model_name,
        model_path=model_path,
        device=resolved_device,
        class_names=["Regular", "Peculiar"],
    )
    validator.validate_dataset(test_loader=test_loader, save_results=True, output_dir=output_dir)
    return {
        "success": True,
        "message": "Validation completed successfully!",
        "output_dir": output_dir,
        "results_path": f"{output_dir}/metrics.json",
        "model_paths": None,
    }


def run_benchmark_entry(*, model_paths: dict[str, str], data_dir: str, output_dir: str, device: str) -> dict:
    resolved_device = resolve_device(device)
    test_loader = build_test_loader(data_dir)
    benchmark = ModelBenchmark(
        test_loader=test_loader,
        class_names=["Regular", "Peculiar"],
        device=resolved_device,
        output_dir=output_dir,
    )
    benchmark.run_benchmark(model_paths, save_results=True)
    benchmark.print_summary()
    return {
        "success": True,
        "message": "Benchmark completed successfully!",
        "output_dir": output_dir,
        "results_path": f"{output_dir}/benchmark_comparison.csv",
        "model_paths": model_paths,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_validation_core.py::test_validate_single_entry_returns_structured_result tests/validation/test_validation_core.py::test_benchmark_entry_returns_structured_result -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validate_models.py tests/validation/test_validation_core.py
git commit -m "refactor: expose validation entrypoints"
```

### Task 4: Criar a nova CLI de Validation

**Files:**
- Create: `validation/validation_cli.py`
- Create: `tests/validation/test_validation_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from validation.validation_cli import build_parser, main


def test_build_parser_accepts_benchmark_mode():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--data-dir", "data/complete_sdss"])

    assert args.command == "benchmark"
    assert args.data_dir == "data/complete_sdss"


def test_main_returns_zero_on_success(monkeypatch, capsys):
    monkeypatch.setattr(
        "validation.validation_cli.run_validation",
        lambda request, deps: type(
            "Result",
            (),
            {
                "success": True,
                "message": "benchmark ok",
                "output_dir": "benchmark_results",
                "results_path": "benchmark_results/benchmark_comparison.csv",
            },
        )(),
    )
    monkeypatch.setattr("validation.validation_cli.build_dependencies", lambda: "deps-sentinel")

    exit_code = main(["benchmark", "--data-dir", "data/complete_sdss"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "benchmark ok" in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_validation_cli.py -v`
Expected: FAIL com `ModuleNotFoundError` para `validation.validation_cli`

- [ ] **Step 3: Write minimal implementation**

```python
import argparse
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from validation.validation_core import ValidationDependencies, ValidationRequest, run_validation
from validation.validate_models import (
    find_trained_models,
    run_benchmark_entry,
    validate_single_entry,
)


def cuda_is_available() -> bool:
    import torch
    return torch.cuda.is_available()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified validation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--model-name", required=True)
    single.add_argument("--model-path", required=True)
    single.add_argument("--data-dir", default="data/complete_sdss")
    single.add_argument("--output-dir")
    single.add_argument("--device", default="auto")
    single.add_argument("--results-dir", default="results/models")

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--data-dir", default="data/complete_sdss")
    benchmark.add_argument("--output-dir")
    benchmark.add_argument("--device", default="auto")
    benchmark.add_argument("--results-dir", default="results/models")

    find_cmd = subparsers.add_parser("find")
    find_cmd.add_argument("--data-dir", default="data/complete_sdss")
    find_cmd.add_argument("--output-dir")
    find_cmd.add_argument("--device", default="auto")
    find_cmd.add_argument("--results-dir", default="results/models")

    return parser


def build_request(args: argparse.Namespace) -> ValidationRequest:
    return ValidationRequest(
        mode=args.command,
        model_name=getattr(args, "model_name", None),
        model_path=getattr(args, "model_path", None),
        data_dir=args.data_dir,
        output_dir=getattr(args, "output_dir", None),
        device=args.device,
        results_dir=args.results_dir,
    )


def build_dependencies() -> ValidationDependencies:
    return ValidationDependencies(
        validate_single_runner=lambda request: validate_single_entry(
            model_name=request.model_name,
            model_path=request.model_path,
            data_dir=request.data_dir,
            output_dir=request.output_dir or f"validation_results/{request.model_name}",
            device=request.resolved_device or request.device,
        ),
        benchmark_runner=lambda request: run_benchmark_entry(
            model_paths=find_trained_models(request.results_dir),
            data_dir=request.data_dir,
            output_dir=request.output_dir or "benchmark_results",
            device=request.resolved_device or request.device,
        ),
        find_models_runner=find_trained_models,
        cuda_available=cuda_is_available,
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_validation(build_request(args), build_dependencies())
    print(result.message)
    return 0 if result.success else 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_validation_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add validation/validation_cli.py tests/validation/test_validation_cli.py
git commit -m "feat: add unified validation cli"
```

### Task 5: Transformar `validate_models.py` em delegador fino

**Files:**
- Modify: `validation/validate_models.py`
- Modify: `tests/validation/test_validation_cli.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path


def test_validate_models_script_delegates_to_validation_cli():
    script = Path("validation/validate_models.py").read_text()
    assert "from validation.validation_cli import main as validation_main" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_cli.py::test_validate_models_script_delegates_to_validation_cli -v`
Expected: FAIL porque `validate_models.py` ainda concentra a CLI antiga

- [ ] **Step 3: Write minimal implementation**

```python
from validation.validation_cli import main as validation_main


if __name__ == "__main__":
    raise SystemExit(validation_main())
```

```python
def main():
    return validation_main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_validation_cli.py::test_validate_models_script_delegates_to_validation_cli -v`
Expected: PASS

- [ ] **Step 5: Run focused regression**

Run: `pytest tests/validation/test_validation_core.py tests/validation/test_validation_cli.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add validation/validate_models.py tests/validation/test_validation_core.py tests/validation/test_validation_cli.py validation/validation_core.py validation/validation_cli.py
git commit -m "refactor: route validation through unified cli"
```

### Task 6: Finalizar tratamento de erro e execucao direta

**Files:**
- Modify: `validation/validation_cli.py`
- Modify: `tests/validation/test_validation_cli.py`

- [ ] **Step 1: Write the failing test**

```python
import subprocess
import sys
from pathlib import Path

from validation.validation_cli import main


def test_main_returns_one_and_prints_validation_error(monkeypatch, capsys):
    def fake_run_validation(request, deps):
        raise ValueError("invalid validation configuration")

    monkeypatch.setattr("validation.validation_cli.run_validation", fake_run_validation)
    monkeypatch.setattr("validation.validation_cli.build_dependencies", lambda: "deps")

    exit_code = main(["benchmark", "--data-dir", "data/complete_sdss"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "invalid validation configuration" in captured.out


def test_validation_cli_runs_as_script_from_repo_root():
    result = subprocess.run(
        [sys.executable, "validation/validation_cli.py", "benchmark", "--help"],
        cwd=Path("/workspace"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_validation_cli.py::test_main_returns_one_and_prints_validation_error tests/validation/test_validation_cli.py::test_validation_cli_runs_as_script_from_repo_root -v`
Expected: FAIL porque a CLI ainda nao captura `ValueError` ou ainda nao sobe corretamente como script direto

- [ ] **Step 3: Write minimal implementation**

```python
def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_validation(build_request(args), build_dependencies())
    except ValueError as error:
        print(str(error))
        return 1

    print(result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run final checks**

Run: `pytest tests/validation/test_validation_core.py tests/validation/test_validation_cli.py -v`
Expected: PASS

Run: `python validation/validation_cli.py single --help`
Expected: ajuda do subcomando `single`

Run: `python validation/validation_cli.py benchmark --help`
Expected: ajuda do subcomando `benchmark`

Run: `python validation/validation_cli.py find --help`
Expected: ajuda do subcomando `find`

- [ ] **Step 5: Commit**

```bash
git add validation/validation_cli.py tests/validation/test_validation_cli.py
git commit -m "test: finalize validation cli error handling"
```
