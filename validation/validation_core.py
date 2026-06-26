from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ValidationRequest:
    mode: str
    model_name: str | None
    model_path: str | None
    data_dir: str
    output_dir: str | None
    device: str
    results_dir: str
    resolved_device: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    success: bool
    message: str
    output_dir: str | None = None
    results_path: str | None = None
    model_paths: dict[str, str] | None = None


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


def _validate_request(request: ValidationRequest) -> None:
    valid_modes = {"single", "benchmark", "find"}
    if request.mode not in valid_modes:
        raise ValueError(f"Unsupported validation mode: {request.mode}")

    if request.mode == "single" and not (request.model_name and request.model_path):
        raise ValueError("single mode requires model_name and model_path")


def run_validation(
    request: ValidationRequest, deps: ValidationDependencies
) -> ValidationResult:
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
