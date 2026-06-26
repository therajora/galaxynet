# Validation Legacy Compat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplificar `validation/validate_models.py` removendo duplicacao entre os fluxos legados `single` e `benchmark`, preservando compatibilidade publica.

**Architecture:** A implementacao mantem `validation/validate_models.py` como ponte legada publica, mas extrai a execucao real para dois helpers internos pequenos: um para `single` e outro para `benchmark`. As funcoes publicas existentes continuam com as mesmas assinaturas e passam a ser wrappers finos, enquanto `main()` continua delegando para `validation_cli.py`.

**Tech Stack:** Python 3.11, pytest, monkeypatch, pathlib, torch, ModelValidator, ModelBenchmark

---

## File Structure

- `validation/validate_models.py`
  - Continua como modulo publico legado.
  - Ganha dois helpers internos: `_run_single_validation()` e `_run_benchmark_validation()`.
  - Mantem `validate_single_entry()`, `validate_single_model()`, `run_benchmark_entry()`, `run_benchmark()`, `find_trained_models()` e `main()`.

- `tests/validation/test_validation_cli.py`
  - Ganha testes focais para garantir que os pares publicos legados delegam para os mesmos helpers internos.
  - Mantem os testes atuais de CLI e compatibilidade.

### Task 1: Consolidar o fluxo legado `single`

**Files:**
- Modify: `tests/validation/test_validation_cli.py`
- Modify: `validation/validate_models.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_validate_single_public_functions_delegate_to_shared_helper(monkeypatch):
    import validation.validate_models as validate_models

    calls = []

    def fake_run_single_validation(**kwargs):
        calls.append(kwargs)
        return {"metrics": {"accuracy": 0.9}, "predictions": [1, 0]}

    monkeypatch.setattr(
        validate_models,
        "_run_single_validation",
        fake_run_single_validation,
        raising=False,
    )

    entry_result = validate_models.validate_single_entry(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        data_dir="data/complete_sdss",
        output_dir="validation_results/efficientnet_b0",
        device="cpu",
    )
    legacy_result = validate_models.validate_single_model(
        model_name="efficientnet_b0",
        model_path="results/models/efficientnet_b0/best_model.pth",
        data_dir="data/complete_sdss",
        output_dir="validation_results/efficientnet_b0",
        device="cpu",
    )

    assert entry_result == {
        "success": True,
        "message": "Validation completed successfully!",
        "output_dir": "validation_results/efficientnet_b0",
        "results_path": "validation_results/efficientnet_b0/metrics.json",
        "model_paths": None,
    }
    assert legacy_result == {"metrics": {"accuracy": 0.9}, "predictions": [1, 0]}
    assert calls == [
        {
            "model_name": "efficientnet_b0",
            "model_path": "results/models/efficientnet_b0/best_model.pth",
            "data_dir": "data/complete_sdss",
            "output_dir": "validation_results/efficientnet_b0",
            "device": "cpu",
        },
        {
            "model_name": "efficientnet_b0",
            "model_path": "results/models/efficientnet_b0/best_model.pth",
            "data_dir": "data/complete_sdss",
            "output_dir": "validation_results/efficientnet_b0",
            "device": "cpu",
        },
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_cli.py::test_validate_single_public_functions_delegate_to_shared_helper -v`
Expected: FAIL because `validate_single_entry()` e `validate_single_model()` ainda nao delegam para `_run_single_validation()`

- [ ] **Step 3: Write minimal implementation**

```python
def _run_single_validation(
    *,
    model_name: str,
    model_path: str,
    data_dir: str,
    output_dir: str,
    device: str,
):
    print(f"Validating model: {model_name}")

    resolved_device = resolve_device(device)
    test_loader = build_test_loader(data_dir)
    validator = ModelValidator(
        model_name=model_name,
        model_path=model_path,
        device=resolved_device,
        class_names=CLASS_NAMES,
    )
    return validator.validate_dataset(
        test_loader=test_loader,
        save_results=True,
        output_dir=output_dir,
    )


def validate_single_entry(*, model_name: str, model_path: str, data_dir: str, output_dir: str, device: str) -> Dict:
    _run_single_validation(
        model_name=model_name,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
    return {
        "success": True,
        "message": "Validation completed successfully!",
        "output_dir": output_dir,
        "results_path": f"{output_dir}/metrics.json",
        "model_paths": None,
    }


def validate_single_model(
    model_name: str,
    model_path: str,
    data_dir: str,
    output_dir: str = None,
    device: str = "auto",
) -> Dict:
    return _run_single_validation(
        model_name=model_name,
        model_path=model_path,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validation_cli.py::test_validate_single_public_functions_delegate_to_shared_helper -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_validation_cli.py validation/validate_models.py
git commit -m "refactor: share legacy single validation flow"
```

### Task 2: Consolidar o fluxo legado `benchmark`

**Files:**
- Modify: `tests/validation/test_validation_cli.py`
- Modify: `validation/validate_models.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_benchmark_public_functions_delegate_to_shared_helper(monkeypatch):
    import validation.validate_models as validate_models

    calls = []

    def fake_run_benchmark_validation(**kwargs):
        calls.append(kwargs)
        return "dataframe-sentinel"

    monkeypatch.setattr(
        validate_models,
        "_run_benchmark_validation",
        fake_run_benchmark_validation,
        raising=False,
    )

    entry_result = validate_models.run_benchmark_entry(
        model_paths={"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        data_dir="data/complete_sdss",
        output_dir="benchmark_results",
        device="cpu",
    )
    legacy_result = validate_models.run_benchmark(
        model_paths={"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        data_dir="data/complete_sdss",
        output_dir="benchmark_results",
        device="cpu",
    )

    assert entry_result == {
        "success": True,
        "message": "Benchmark completed successfully!",
        "output_dir": "benchmark_results",
        "results_path": "benchmark_results/benchmark_comparison.csv",
        "model_paths": {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
    }
    assert legacy_result == "dataframe-sentinel"
    assert calls == [
        {
            "model_paths": {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
            "data_dir": "data/complete_sdss",
            "output_dir": "benchmark_results",
            "device": "cpu",
        },
        {
            "model_paths": {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
            "data_dir": "data/complete_sdss",
            "output_dir": "benchmark_results",
            "device": "cpu",
        },
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/validation/test_validation_cli.py::test_benchmark_public_functions_delegate_to_shared_helper -v`
Expected: FAIL because `run_benchmark_entry()` e `run_benchmark()` ainda nao delegam para `_run_benchmark_validation()`

- [ ] **Step 3: Write minimal implementation**

```python
def _run_benchmark_validation(
    *,
    model_paths: Dict[str, str],
    data_dir: str,
    output_dir: str,
    device: str,
):
    print(f"Running benchmark of {len(model_paths)} models...")

    resolved_device = resolve_device(device)
    test_loader = build_test_loader(data_dir)
    benchmark = ModelBenchmark(
        test_loader=test_loader,
        class_names=CLASS_NAMES,
        device=resolved_device,
        output_dir=output_dir,
    )
    results = benchmark.run_benchmark(model_paths, save_results=True)
    benchmark.print_summary()
    return results


def run_benchmark_entry(*, model_paths: Dict[str, str], data_dir: str, output_dir: str, device: str) -> Dict:
    _run_benchmark_validation(
        model_paths=model_paths,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
    return {
        "success": True,
        "message": "Benchmark completed successfully!",
        "output_dir": output_dir,
        "results_path": f"{output_dir}/benchmark_comparison.csv",
        "model_paths": model_paths,
    }


def run_benchmark(
    model_paths: Dict[str, str],
    data_dir: str,
    output_dir: str = "benchmark_results",
    device: str = "auto",
):
    return _run_benchmark_validation(
        model_paths=model_paths,
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/validation/test_validation_cli.py::test_benchmark_public_functions_delegate_to_shared_helper -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/validation/test_validation_cli.py validation/validate_models.py
git commit -m "refactor: share legacy benchmark flow"
```

### Task 3: Fechar validacao focal da compatibilidade legada

**Files:**
- Modify: `tests/validation/test_validation_cli.py`
- Modify: `validation/validate_models.py`

- [ ] **Step 1: Keep the thin `main()` delegator untouched**

```python
def main():
    return validation_main()
```

- [ ] **Step 2: Run final checks**

Run: `pytest tests/validation/test_validation_cli.py tests/validation/test_validation_core.py tests/validation/test_validator.py tests/validation/test_model_loader.py -v`
Expected: PASS, including the existing `test_validate_models_main_delegates_to_validation_cli`

Run: `python validation/validation_cli.py benchmark --help`
Expected: help output for `benchmark`

- [ ] **Step 3: Commit**

```bash
git add tests/validation/test_validation_cli.py validation/validate_models.py
git commit -m "test: preserve validation legacy compatibility"
```
