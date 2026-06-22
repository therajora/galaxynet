import subprocess
import sys
from pathlib import Path

from model.training_cli import build_dependencies, build_parser, main


def test_build_parser_accepts_pretrained_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        ["pretrained", "--config", "model/pretrained/configs/efficientnet_b0.json"]
    )

    assert args.command == "pretrained"
    assert args.config == "model/pretrained/configs/efficientnet_b0.json"


def test_build_dependencies_delegate_to_reusable_entrypoints(monkeypatch):
    calls = {}

    def fake_pretrained_entry(**kwargs):
        calls["pretrained"] = kwargs
        return {
            "success": True,
            "message": "pretrained ok",
            "output_dir": "results/models/pretrained",
            "artifacts_path": None,
            "metrics_path": None,
        }

    def fake_from_scratch_entry(**kwargs):
        calls["from_scratch"] = kwargs
        return {
            "success": True,
            "message": "from scratch ok",
            "output_dir": "model_artifacts",
            "artifacts_path": "model_artifacts/galaxy_cnn_artifacts.pth",
            "metrics_path": None,
        }

    monkeypatch.setattr("model.training_cli.run_pretrained_entry", fake_pretrained_entry)
    monkeypatch.setattr(
        "model.training_cli.run_from_scratch_entry", fake_from_scratch_entry
    )

    deps = build_dependencies()

    pretrained_result = deps.pretrained_runner(
        type(
            "Request",
            (),
            {
                "config_path": "model/pretrained/configs/efficientnet_b0.json",
                "resolved_device": "cpu",
                "device": "auto",
                "seed": 42,
                "validate_only": True,
            },
        )()
    )
    from_scratch_result = deps.from_scratch_runner(
        type(
            "Request",
            (),
            {
                "data_dir": "data/complete_sdss",
                "resolved_device": "cpu",
                "device": "auto",
                "seed": 7,
                "batch_size": 16,
                "num_epochs": 2,
                "learning_rate": 0.01,
                "model_name": "galaxy_cnn",
                "predict_samples": 0,
                "evaluate": False,
            },
        )()
    )

    assert pretrained_result["message"] == "pretrained ok"
    assert calls["pretrained"] == {
        "config_path": "model/pretrained/configs/efficientnet_b0.json",
        "device": "cpu",
        "seed": 42,
        "validate_only": True,
    }
    assert from_scratch_result["message"] == "from scratch ok"
    assert calls["from_scratch"] == {
        "data_dir": "data/complete_sdss",
        "device": "cpu",
        "seed": 7,
        "batch_size": 16,
        "num_epochs": 2,
        "learning_rate": 0.01,
        "model_name": "galaxy_cnn",
        "predict_samples": 0,
        "evaluate": False,
    }


def test_main_returns_zero_on_success(monkeypatch, capsys):
    captured = {}

    def fake_run_training(request, deps):
        captured["request"] = request
        captured["deps"] = deps
        return type(
            "Result",
            (),
            {"success": True, "message": "ok", "output_dir": "results/models"},
        )()

    monkeypatch.setattr("model.training_cli.run_training", fake_run_training)
    monkeypatch.setattr(
        "model.training_cli.build_dependencies", lambda: "deps-sentinel"
    )

    exit_code = main(
        ["pretrained", "--config", "model/pretrained/configs/efficientnet_b0.json"]
    )

    captured_output = capsys.readouterr()
    assert exit_code == 0
    assert "ok" in captured_output.out
    assert captured["request"].mode == "pretrained"
    assert (
        captured["request"].config_path
        == "model/pretrained/configs/efficientnet_b0.json"
    )
    assert captured["deps"] == "deps-sentinel"


def test_main_returns_one_and_prints_validation_error(monkeypatch, capsys):
    def fake_run_training(request, deps):
        raise ValueError("invalid configuration")

    monkeypatch.setattr("model.training_cli.run_training", fake_run_training)
    monkeypatch.setattr(
        "model.training_cli.build_dependencies", lambda: "deps-sentinel"
    )

    exit_code = main(
        ["pretrained", "--config", "model/pretrained/configs/efficientnet_b0.json"]
    )

    captured_output = capsys.readouterr()
    assert exit_code == 1
    assert "invalid configuration" in captured_output.out


def test_training_cli_runs_as_script_from_repo_root():
    result = subprocess.run(
        [sys.executable, "model/training_cli.py", "pretrained", "--help"],
        cwd=Path("/workspace"),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "usage:" in result.stdout


def test_shell_wrappers_delegate_to_python_cli():
    pretrained_wrapper = Path("utils/scripts/train_pretrained.sh").read_text()
    from_scratch_wrapper = Path("utils/scripts/from_scratch.sh").read_text()

    assert "python model/training_cli.py pretrained" in pretrained_wrapper
    assert "python model/training_cli.py from-scratch" in from_scratch_wrapper
