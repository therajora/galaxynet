import argparse
import sys
from pathlib import Path
from typing import Optional

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from model.training_core import TrainingDependencies, TrainingRequest, run_training


def run_pretrained_entry(**kwargs):
    from model.pretrained.train_pretrained import run_pretrained_entry as entry

    return entry(**kwargs)


def run_from_scratch_entry(**kwargs):
    from model.from_scratch.train_from_scratch import run_from_scratch_entry as entry

    return entry(**kwargs)


def cuda_is_available() -> bool:
    import torch

    return torch.cuda.is_available()


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


def build_dependencies() -> TrainingDependencies:
    return TrainingDependencies(
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
        cuda_available=cuda_is_available,
    )


def build_request(args: argparse.Namespace) -> TrainingRequest:
    return TrainingRequest(
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


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_training(build_request(args), build_dependencies())
    except ValueError as error:
        print(str(error))
        return 1

    print(result.message)
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
