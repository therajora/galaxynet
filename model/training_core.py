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
