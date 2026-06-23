import sys
import subprocess
from pathlib import Path
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
    "model.pretrained.dataset",
    GalaxyPretrainedDataset=object,
    create_data_loaders=lambda *args, **kwargs: None,
    get_imagenet_transforms=lambda *args, **kwargs: None,
)
_stub_validation_module(
    "torch",
    device=lambda value: value,
    cuda=type("CudaModule", (), {"is_available": staticmethod(lambda: False)})(),
    utils=type(
        "UtilsModule",
        (),
        {"data": type("DataModule", (), {"DataLoader": object})()},
    )(),
)
_stub_validation_module(
    "validation.visualization",
    ValidationPlotGenerator=object,
    ComparisonPlotGenerator=object,
    ModelAnalysisPlotter=object,
)


from validation.validation_cli import build_dependencies, build_parser, main


def test_build_parser_accepts_benchmark_mode():
    parser = build_parser()
    args = parser.parse_args(["benchmark", "--data-dir", "data/complete_sdss"])

    assert args.command == "benchmark"
    assert args.data_dir == "data/complete_sdss"


def test_build_dependencies_delegate_to_reusable_entrypoints(monkeypatch):
    calls = {}

    def fake_validate_single_entry(**kwargs):
        calls["single"] = kwargs
        return {
            "success": True,
            "message": "single ok",
            "output_dir": "validation_results/efficientnet_b0",
            "results_path": "validation_results/efficientnet_b0/metrics.json",
            "model_paths": None,
        }

    def fake_run_benchmark_entry(**kwargs):
        calls["benchmark"] = kwargs
        return {
            "success": True,
            "message": "benchmark ok",
            "output_dir": "benchmark_results",
            "results_path": "benchmark_results/benchmark_comparison.csv",
            "model_paths": {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        }

    monkeypatch.setattr(
        "validation.validation_cli.validate_single_entry", fake_validate_single_entry
    )
    monkeypatch.setattr(
        "validation.validation_cli.run_benchmark_entry", fake_run_benchmark_entry
    )
    monkeypatch.setattr(
        "validation.validation_cli.find_trained_models",
        lambda results_dir: {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
    )

    deps = build_dependencies()

    single_result = deps.validate_single_runner(
        type(
            "Request",
            (),
            {
                "model_name": "efficientnet_b0",
                "model_path": "results/models/efficientnet_b0/best_model.pth",
                "data_dir": "data/complete_sdss",
                "output_dir": None,
                "resolved_device": "cpu",
                "device": "auto",
            },
        )()
    )
    benchmark_result = deps.benchmark_runner(
        type(
            "Request",
            (),
            {
                "results_dir": "results/models",
                "data_dir": "data/complete_sdss",
                "output_dir": None,
                "resolved_device": "cpu",
                "device": "auto",
            },
        )()
    )

    assert single_result["message"] == "single ok"
    assert calls["single"] == {
        "model_name": "efficientnet_b0",
        "model_path": "results/models/efficientnet_b0/best_model.pth",
        "data_dir": "data/complete_sdss",
        "output_dir": "validation_results/efficientnet_b0",
        "device": "cpu",
    }
    assert benchmark_result["message"] == "benchmark ok"
    assert calls["benchmark"] == {
        "model_paths": {"resnet50_v1": "results/models/resnet50_v1/best_model.pth"},
        "data_dir": "data/complete_sdss",
        "output_dir": "benchmark_results",
        "device": "cpu",
    }


def test_main_returns_zero_on_success(monkeypatch, capsys):
    captured = {}

    def fake_run_validation(request, deps):
        captured["request"] = request
        captured["deps"] = deps
        return type(
            "Result",
            (),
            {
                "success": True,
                "message": "benchmark ok",
                "output_dir": "benchmark_results",
                "results_path": "benchmark_results/benchmark_comparison.csv",
            },
        )()

    monkeypatch.setattr("validation.validation_cli.run_validation", fake_run_validation)
    monkeypatch.setattr(
        "validation.validation_cli.build_dependencies", lambda: "deps-sentinel"
    )

    exit_code = main(["benchmark", "--data-dir", "data/complete_sdss"])

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert "benchmark ok" in captured_output.out
    assert captured["request"].mode == "benchmark"
    assert captured["request"].data_dir == "data/complete_sdss"
    assert captured["deps"] == "deps-sentinel"


def test_main_returns_one_and_prints_validation_error(monkeypatch, capsys):
    def fake_run_validation(request, deps):
        raise ValueError("invalid validation configuration")

    monkeypatch.setattr("validation.validation_cli.run_validation", fake_run_validation)
    monkeypatch.setattr("validation.validation_cli.build_dependencies", lambda: "deps")

    exit_code = main(["benchmark", "--data-dir", "data/complete_sdss"])

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "invalid validation configuration" in captured_output.out


def test_validate_models_main_delegates_to_validation_cli(monkeypatch):
    import validation.validate_models as validate_models

    calls = {}

    monkeypatch.setattr(validate_models.sys, "argv", ["validate_models.py", "--mode", "find"])
    monkeypatch.setattr(
        validate_models,
        "find_trained_models",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("main antigo executado")
        ),
    )
    monkeypatch.setattr(
        validate_models,
        "validation_main",
        lambda argv=None: calls.setdefault("argv", argv) or 7,
        raising=False,
    )

    assert validate_models.main() == 7
    assert calls["argv"] is None


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
