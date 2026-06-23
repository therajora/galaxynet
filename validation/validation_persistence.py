import json
from pathlib import Path


def save_validation_outputs(
    *,
    output_dir,
    model_name,
    model_path,
    class_names,
    y_true,
    y_pred,
    y_prob,
    metrics_calculator,
    generate_visualizations,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_calculator.save_metrics(output_path / "metrics.json")

    payload = {
        "model_name": model_name,
        "model_path": model_path,
        "num_samples": len(y_true),
        "predictions": list(y_pred),
        "probabilities": list(y_prob),
        "true_labels": list(y_true),
        "class_names": class_names,
    }

    with (output_path / "predictions.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)

    generate_visualizations(
        output_dir=output_path,
        metrics_calculator=metrics_calculator,
    )
