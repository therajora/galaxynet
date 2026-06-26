from pathlib import Path

import torch

from model.pretrained.model_factory import create_pretrained_model


def load_trained_model(*, model_name, model_path, num_classes, device):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = create_pretrained_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,
    )

    checkpoint = torch.load(model_path, map_location=device)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model
