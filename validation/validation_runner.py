try:
    import numpy as np
except ModuleNotFoundError:
    class _ArrayLike(list):
        def tolist(self):
            return [
                item.tolist() if hasattr(item, "tolist") else item
                for item in self
            ]

    class _NumpyFallback:
        @staticmethod
        def array(values):
            if isinstance(values, list):
                return _ArrayLike(
                    [
                        _NumpyFallback.array(item) if isinstance(item, list) else item
                        for item in values
                    ]
                )
            return values

    np = _NumpyFallback()


def softmax_outputs(outputs):
    import torch

    return torch.softmax(outputs, dim=1)


def argmax_outputs(outputs):
    import torch

    return torch.argmax(outputs, dim=1)


def run_validation_batches(*, model, test_loader, device):
    all_predictions = []
    all_probabilities = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probabilities = softmax_outputs(outputs)
        predictions = argmax_outputs(outputs)

        all_predictions.extend(predictions.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return {
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_predictions),
        "y_prob": np.array(all_probabilities),
    }
