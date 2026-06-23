import argparse
import importlib
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _load_validation_core():
    if __package__ in (None, ""):
        from validation_core import (
            ValidationDependencies,
            ValidationRequest,
            run_validation,
        )
    else:
        from validation.validation_core import (
            ValidationDependencies,
            ValidationRequest,
            run_validation,
        )

    return ValidationDependencies, ValidationRequest, run_validation


def _load_validate_models_module():
    if __package__ in (None, ""):
        import validate_models as module
    else:
        module = importlib.import_module("validation.validate_models")

    return module


def run_validation(request, deps):
    _, _, runner = _load_validation_core()
    return runner(request, deps)


def validate_single_entry(**kwargs):
    return _load_validate_models_module().validate_single_entry(**kwargs)


def run_benchmark_entry(**kwargs):
    return _load_validate_models_module().run_benchmark_entry(**kwargs)


def find_trained_models(results_dir: str):
    return _load_validate_models_module().find_trained_models(results_dir)


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


def build_request(args: argparse.Namespace):
    _, ValidationRequest, _ = _load_validation_core()
    return ValidationRequest(
        mode=args.command,
        model_name=getattr(args, "model_name", None),
        model_path=getattr(args, "model_path", None),
        data_dir=args.data_dir,
        output_dir=getattr(args, "output_dir", None),
        device=args.device,
        results_dir=args.results_dir,
    )


def build_dependencies():
    ValidationDependencies, _, _ = _load_validation_core()
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
    try:
        result = run_validation(build_request(args), build_dependencies())
    except ValueError as error:
        print(str(error))
        return 1

    print(result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
