# Benchmark Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplificar a execucao de `ModelBenchmark.run_benchmark()` isolando a validacao de um unico modelo em um helper interno testavel.

**Architecture:** A implementacao mantem `validation/benchmark.py` como modulo publico, mas extrai a execucao de um modelo para um helper interno pequeno que devolve payload de sucesso ou erro. `run_benchmark()` continua publico e responsavel por iterar, preencher `self.results`, construir `self.comparison_df` e acionar a persistencia existente sem tocar ainda em CSV/JSON/plots.

**Tech Stack:** Python 3.11, pytest, monkeypatch, importlib, sys.modules stubs, pandas, torch

---

## File Structure

- `validation/benchmark.py`
  - Mantem `ModelBenchmark` como API publica.
  - Ganha um helper interno para executar benchmark de um unico modelo.
  - `run_benchmark()` passa a delegar a esse helper e manter o resto do contrato.

- `tests/validation/test_benchmark.py`
  - Novo arquivo de testes focado na execucao do benchmark.
  - Isola imports pesados com stubs de `torch`, `pandas`, `validation.validator` e dependencias relacionadas.

### Task 1: Criar harness de testes e cobrir sucesso por modelo

**Files:**
- Create: `tests/validation/test_benchmark.py`
- Modify: `validation/benchmark.py`

- [ ] **Step 1: Write the failing tests**

```python
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _stub_module(name: str, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse=True)
def _restore_modules():
    module_names = [
        "torch",
        "torch.utils",
        "torch.utils.data",
        "pandas",
        "numpy",
        "matplotlib",
        "matplotlib.pyplot",
        "seaborn",
        "tqdm",
        "model",
        "model.pretrained",
        "model.pretrained.model_factory",
        "validation.validator",
        "validation.metrics",
        "validation.benchmark",
    ]
    original = {name: sys.modules.get(name) for name in module_names}
    yield
    for name, module in original.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_benchmark_module():
    torch_module = ModuleType("torch")
    torch_module.device = str
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_module
    _stub_module("torch.utils", data=SimpleNamespace(DataLoader=object))
    _stub_module("torch.utils.data", DataLoader=object)
    _stub_module(
        "pandas",
        DataFrame=type("FakeDataFrame", (), {}),
    )
    _stub_module(
        "numpy",
        nan=float("nan"),
        mean=lambda values: sum(values) / len(values),
    )
    _stub_module("matplotlib", pyplot=SimpleNamespace(close=lambda *args, **kwargs: None))
    _stub_module("matplotlib.pyplot", close=lambda *args, **kwargs: None)
    _stub_module("seaborn")
    _stub_module("tqdm", tqdm=lambda iterable, **kwargs: iterable)
    _stub_module("model")
    _stub_module("model.pretrained")
    _stub_module(
        "model.pretrained.model_factory",
        get_available_models=lambda: [],
        get_model_info=lambda name: {"description": name},
        count_parameters=lambda model: {"total": 10},
    )
    _stub_module("validation.metrics", ClassificationMetrics=object)
    _stub_module("validation.validator", ModelValidator=object)

    sys.modules.pop("validation.benchmark", None)
    return importlib.import_module("validation.benchmark")


def test_run_single_model_benchmark_returns_success_payload(monkeypatch):
    benchmark_module = _load_benchmark_module()

    calls = {}

    class FakeValidator:
        def __init__(self, **kwargs):
            calls["validator_kwargs"] = kwargs

        def validate_dataset(self, *, test_loader, save_results):
            calls["validate_kwargs"] = {
                "test_loader": test_loader,
                "save_results": save_results,
            }
            return {"metrics": {"accuracy": 0.9}}

        def get_model_info(self):
            return {"parameters": {"total": 123}}

    monkeypatch.setattr(benchmark_module, "ModelValidator", FakeValidator)

    benchmark = benchmark_module.ModelBenchmark(
        test_loader="loader",
        class_names=["Regular", "Peculiar"],
        device="cpu",
        output_dir="benchmark_results",
    )

    result = benchmark._run_single_model_benchmark(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
    )

    assert result == {
        "metrics": {"accuracy": 0.9},
        "model_info": {"parameters": {"total": 123}},
    }
    assert calls["validator_kwargs"] == {
        "model_name": "efficientnet_b0",
        "model_path": "results/models/efficientnet_b0/best_model.pth",
        "device": "cpu",
        "class_names": ["Regular", "Peculiar"],
    }
    assert calls["validate_kwargs"] == {
        "test_loader": "loader",
        "save_results": False,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_benchmark.py::test_run_single_model_benchmark_returns_success_payload -v`
Expected: FAIL because `_run_single_model_benchmark()` does not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
def _run_single_model_benchmark(
    self,
    *,
    model_name: str,
    model_path: Union[str, Path],
) -> Dict:
    validator = ModelValidator(
        model_name=model_name,
        model_path=model_path,
        device=self.device,
        class_names=self.class_names,
    )
    results = validator.validate_dataset(
        test_loader=self.test_loader,
        save_results=False,
    )
    return {
        "metrics": results["metrics"],
        "model_info": validator.get_model_info(),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_benchmark.py::test_run_single_model_benchmark_returns_success_payload -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_benchmark.py validation/benchmark.py
git commit -m "test: cover benchmark single-model execution"
```

### Task 2: Cobrir erro por modelo no helper interno

**Files:**
- Modify: `tests/validation/test_benchmark.py`
- Modify: `validation/benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_single_model_benchmark_returns_error_payload(monkeypatch):
    benchmark_module = _load_benchmark_module()

    class FailingValidator:
        def __init__(self, **kwargs):
            raise RuntimeError("checkpoint corrupted")

    monkeypatch.setattr(benchmark_module, "ModelValidator", FailingValidator)

    benchmark = benchmark_module.ModelBenchmark(
        test_loader="loader",
        class_names=["Regular", "Peculiar"],
        device="cpu",
        output_dir="benchmark_results",
    )

    result = benchmark._run_single_model_benchmark(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
    )

    assert result == {"error": "checkpoint corrupted"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_benchmark.py::test_run_single_model_benchmark_returns_error_payload -v`
Expected: FAIL because the helper still lets the exception escape

- [ ] **Step 3: Write minimal implementation**

```python
def _run_single_model_benchmark(
    self,
    *,
    model_name: str,
    model_path: Union[str, Path],
) -> Dict:
    try:
        validator = ModelValidator(
            model_name=model_name,
            model_path=model_path,
            device=self.device,
            class_names=self.class_names,
        )
        results = validator.validate_dataset(
            test_loader=self.test_loader,
            save_results=False,
        )
        return {
            "metrics": results["metrics"],
            "model_info": validator.get_model_info(),
        }
    except Exception as exc:
        return {"error": str(exc)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_benchmark.py::test_run_single_model_benchmark_returns_error_payload -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_benchmark.py validation/benchmark.py
git commit -m "test: cover benchmark model execution errors"
```

### Task 3: Adaptar `run_benchmark()` para delegar ao helper

**Files:**
- Modify: `tests/validation/test_benchmark.py`
- Modify: `validation/benchmark.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_run_benchmark_populates_results_via_shared_helper(monkeypatch):
    benchmark_module = _load_benchmark_module()

    calls = []

    monkeypatch.setattr(
        benchmark_module.ModelBenchmark,
        "_run_single_model_benchmark",
        lambda self, *, model_name, model_path: calls.append(
            {"model_name": model_name, "model_path": model_path}
        ) or {"metrics": {"accuracy": 0.9}, "model_info": {"parameters": {"total": 1}}},
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module.ModelBenchmark,
        "_create_comparison_dataframe",
        lambda self: "comparison-df",
        raising=False,
    )

    benchmark = benchmark_module.ModelBenchmark(
        test_loader="loader",
        class_names=["Regular", "Peculiar"],
        device="cpu",
        output_dir="benchmark_results",
    )

    result = benchmark.run_benchmark(
        {"efficientnet_b0": "results/models/efficientnet_b0/best_model.pth"},
        save_results=False,
    )

    assert result == "comparison-df"
    assert benchmark.results == {
        "efficientnet_b0": {
            "metrics": {"accuracy": 0.9},
            "model_info": {"parameters": {"total": 1}},
        }
    }
    assert benchmark.comparison_df == "comparison-df"
    assert calls == [
        {
            "model_name": "efficientnet_b0",
            "model_path": "results/models/efficientnet_b0/best_model.pth",
        }
    ]


def test_run_benchmark_skips_save_when_save_results_is_false(monkeypatch):
    benchmark_module = _load_benchmark_module()

    saved = {}

    monkeypatch.setattr(
        benchmark_module.ModelBenchmark,
        "_run_single_model_benchmark",
        lambda self, *, model_name, model_path: {"error": "failed"},
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module.ModelBenchmark,
        "_create_comparison_dataframe",
        lambda self: "comparison-df",
        raising=False,
    )
    monkeypatch.setattr(
        benchmark_module.ModelBenchmark,
        "_save_benchmark_results",
        lambda self: saved.setdefault("called", True),
        raising=False,
    )

    benchmark = benchmark_module.ModelBenchmark(
        test_loader="loader",
        class_names=["Regular", "Peculiar"],
        device="cpu",
        output_dir="benchmark_results",
    )

    benchmark.run_benchmark(
        {"broken_model": "results/models/broken_model/best_model.pth"},
        save_results=False,
    )

    assert benchmark.results == {
        "broken_model": {"error": "failed"}
    }
    assert "called" not in saved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/validation/test_benchmark.py::test_run_benchmark_populates_results_via_shared_helper tests/validation/test_benchmark.py::test_run_benchmark_skips_save_when_save_results_is_false -v`
Expected: FAIL because `run_benchmark()` still contains inline execution instead of delegating

- [ ] **Step 3: Write minimal implementation**

```python
def run_benchmark(
    self,
    model_paths: Dict[str, Union[str, Path]],
    save_results: bool = True
) -> pd.DataFrame:
    print(f"\nStarting benchmark of {len(model_paths)} models...")

    for model_name, model_path in tqdm(model_paths.items(), desc="Validating models"):
        print(f"\nValidating {model_name}...")
        self.results[model_name] = self._run_single_model_benchmark(
            model_name=model_name,
            model_path=model_path,
        )
        if "error" in self.results[model_name]:
            print(f"Error validating {model_name}: {self.results[model_name]['error']}")
        else:
            print(f"{model_name} validated successfully")

    self.comparison_df = self._create_comparison_dataframe()

    if save_results:
        self._save_benchmark_results()

    return self.comparison_df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/validation/test_benchmark.py::test_run_benchmark_populates_results_via_shared_helper tests/validation/test_benchmark.py::test_run_benchmark_skips_save_when_save_results_is_false -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_benchmark.py validation/benchmark.py
git commit -m "refactor: share benchmark execution flow"
```

### Task 4: Fechar a validacao focal do benchmark

**Files:**
- Modify: `tests/validation/test_benchmark.py`
- Modify: `validation/benchmark.py`

- [ ] **Step 1: Keep persistence and visualizations untouched**

```python
def _save_benchmark_results(self):
    print(f"\nSaving benchmark results...")
    self.comparison_df.to_csv(
        self.output_dir / "benchmark_comparison.csv",
        index=False
    )
    detailed_results = {}
    for model_name, result in self.results.items():
        if 'error' not in result:
            detailed_results[model_name] = {
                'metrics': result['metrics'],
                'model_info': result['model_info']
            }
        else:
            detailed_results[model_name] = {'error': result['error']}
    with open(self.output_dir / "detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=4)
    self._generate_benchmark_visualizations()
```

- [ ] **Step 2: Run final checks**

Run: `pytest tests/validation/test_benchmark.py tests/validation/test_validation_cli.py tests/validation/test_validation_core.py tests/validation/test_validator.py tests/validation/test_model_loader.py -v`
Expected: PASS

Run: `python -m py_compile validation/benchmark.py tests/validation/test_benchmark.py`
Expected: no output and exit code 0

- [ ] **Step 3: Commit**

```bash
git add tests/validation/test_benchmark.py validation/benchmark.py
git commit -m "test: preserve benchmark execution behavior"
```
