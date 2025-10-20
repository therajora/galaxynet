"""
Module for uploading trained models to Hugging Face Hub.

Includes functionality for:
- PyTorch model upload
- Metrics and results upload
- Automatic README creation
- Model versioning
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from huggingface_hub import HfApi, login, create_repo
    from huggingface_hub.utils import RepositoryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("huggingface_hub not available. Install with: pip install huggingface_hub")


class ModelUploader:
    """Class for uploading models to Hugging Face Hub."""
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the uploader.
        
        Args:
            token: Hugging Face token (optional, can use environment variable)
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is not available")
        
        self.api = HfApi()
        self.token = token or os.environ.get('HUGGINGFACE_TOKEN')
        
        if self.token:
            login(token=self.token)
        else:
            print("Hugging Face token not provided")
    
    def create_model_card(self, 
                         model_name: str,
                         metrics: Dict[str, Any],
                         description: str = "",
                         dataset_info: str = "rajora/galaxy-classification-sdss") -> str:
        """
        Create a README.md for the model.
        
        Args:
            model_name: Model name
            metrics: Model metrics
            description: Model description
            dataset_info: Dataset information
            
        Returns:
            README.md content
        """
        # Extract main metrics
        accuracy = metrics.get('accuracy', 0)
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        f1_score = metrics.get('f1_score', 0)
        roc_auc = metrics.get('roc_auc', 0)
        confusion_matrix = metrics.get('confusion_matrix', [])
        
        # Create metrics table
        metrics_table = f"""
| Metric | Value |
|---------|-------|
| Accuracy | {accuracy:.4f} |
| Precision | {precision:.4f} |
| Recall | {recall:.4f} |
| F1-Score | {f1_score:.4f} |
| ROC-AUC | {roc_auc:.4f} |
"""
        
        # Create confusion matrix if available
        confusion_section = ""
        if confusion_matrix:
            confusion_section = f"""
## Confusion Matrix

```
{confusion_matrix[0]}
{confusion_matrix[1]}
```

- **True Negatives**: {confusion_matrix[0][0]}
- **False Positives**: {confusion_matrix[0][1]}
- **False Negatives**: {confusion_matrix[1][0]}
- **True Positives**: {confusion_matrix[1][1]}
"""
        
        # README template
        readme_content = f"""---
library_name: pytorch
tags:
- galaxy-classification
- computer-vision
- pytorch
- transfer-learning
license: mit
datasets:
- {dataset_info}
metrics:
- accuracy
- precision
- recall
- f1-score
- roc-auc
---

# {model_name} - Galaxy Classification

{description}

## Performance Metrics

{metrics_table}
{confusion_section}

## Usage

```python
import torch
from transformers import AutoModel, AutoTokenizer

# Load model
model = torch.load('pytorch_model.bin')
model.eval()

# Inference example
# (implement as needed)
```

## Dataset

This model was trained on the dataset: [{dataset_info}](https://huggingface.co/datasets/{dataset_info})

## Technical Information

- **Architecture**: {model_name}
- **Framework**: PyTorch
- **Problem Type**: Binary Classification
- **Classes**: Regular (0), Peculiar (1)
- **Image Size**: 224x224
- **Training Date**: {datetime.now().strftime('%Y-%m-%d')}

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{galaxynet_{model_name.lower()},
  title={{GalaxyNet: {model_name} for Galaxy Classification}},
  author={{GalaxyNet Team}},
  year={{2024}},
  url={{https://huggingface.co/models/your-username/galaxy-classification-{model_name.lower()}}}
}}
```
"""
        
        return readme_content
    
    def save_metrics_in_test_format(self, 
                                   metrics: Dict[str, Any], 
                                   model_name: str,
                                   output_path: str) -> Dict[str, Any]:
        """
        Save metrics in test_results.json format.
        
        Args:
            metrics: Model metrics
            model_name: Model name
            output_path: Path to save
            
        Returns:
            Metrics in standardized format
        """
        # Standardized format based on test_results.json
        formatted_metrics = {
            "confusion_matrix": metrics.get('confusion_matrix', [[0, 0], [0, 0]]),
            "accuracy": float(metrics.get('accuracy', 0)),
            "precision": float(metrics.get('precision', 0)),
            "recall": float(metrics.get('recall', 0)),
            "f1_score": float(metrics.get('f1_score', 0)),
            "roc_auc": float(metrics.get('roc_auc', 0)),
            "model_name": model_name,
            "num_samples": metrics.get('num_samples', 0),
            "training_date": datetime.now().isoformat(),
            "model_path": metrics.get('model_path', ''),
            "config": metrics.get('config', {})
        }
        
        # Save file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump([formatted_metrics], f, indent=2)
        
        return formatted_metrics
    
    def upload_model(self,
                    model_path: str,
                    metrics: Dict[str, Any],
                    repo_name: str,
                    model_name: str,
                    description: str = "",
                    private: bool = False) -> str:
        """
        Upload model to Hugging Face Hub.
        
        Args:
            model_path: Path to model file (.pth)
            metrics: Model metrics
            repo_name: Repository name (e.g., "username/model-name")
            model_name: Model name
            description: Model description
            private: Whether repository should be private
            
        Returns:
            Model URL on Hugging Face Hub
        """
        if not self.token:
            raise ValueError("Hugging Face token is required for upload")
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create repository if it doesn't exist
        try:
            self.api.repo_info(repo_name)
        except RepositoryNotFoundError:
            print(f"Creating repository: {repo_name}")
            create_repo(repo_name, private=private, token=self.token)
        
        # Prepare files for upload
        files_to_upload = {}
        
        # 1. Main model
        files_to_upload['pytorch_model.bin'] = str(model_path)
        
        # 2. Metrics in standardized format
        metrics_formatted = self.save_metrics_in_test_format(
            metrics, model_name, f"/tmp/{model_name}_metrics.json"
        )
        files_to_upload['test_results.json'] = f"/tmp/{model_name}_metrics.json"
        
        # 3. Detailed metrics
        with open(f"/tmp/{model_name}_detailed_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        files_to_upload['detailed_metrics.json'] = f"/tmp/{model_name}_detailed_metrics.json"
        
        # 4. README.md
        readme_content = self.create_model_card(model_name, metrics, description)
        with open(f"/tmp/{model_name}_README.md", 'w') as f:
            f.write(readme_content)
        files_to_upload['README.md'] = f"/tmp/{model_name}_README.md"
        
        # 5. Configuration (if available)
        if 'config' in metrics:
            with open(f"/tmp/{model_name}_config.json", 'w') as f:
                json.dump(metrics['config'], f, indent=2)
            files_to_upload['config.json'] = f"/tmp/{model_name}_config.json"
        
        # Upload files
        print(f"Uploading to: {repo_name}")
        for filename, filepath in files_to_upload.items():
            print(f"  - {filename}")
            self.api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filename,
                repo_id=repo_name,
                token=self.token
            )
        
        # Model URL
        model_url = f"https://huggingface.co/{repo_name}"
        print(f"Upload completed: {model_url}")
        
        return model_url


def upload_model_to_huggingface(model_path: str,
                               metrics: Dict[str, Any],
                               repo_name: str,
                               model_name: str,
                               description: str = "",
                               private: bool = False,
                               token: Optional[str] = None) -> str:
    """
    Convenient function for model upload.
    
    Args:
        model_path: Path to model file
        metrics: Model metrics
        repo_name: Repository name
        model_name: Model name
        description: Model description
        private: Whether repository should be private
        token: Hugging Face token
        
    Returns:
        Model URL on Hugging Face Hub
    """
    uploader = ModelUploader(token=token)
    return uploader.upload_model(
        model_path=model_path,
        metrics=metrics,
        repo_name=repo_name,
        model_name=model_name,
        description=description,
        private=private
    )


def save_training_metrics(model_name: str,
                         metrics: Dict[str, Any],
                         output_dir: str = "results/metrics") -> str:
    """
    Save training metrics in standardized format.
    
    Args:
        model_name: Model name
        metrics: Model metrics
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}_metrics.json"
    output_path = os.path.join(output_dir, filename)
    
    uploader = ModelUploader()
    formatted_metrics = uploader.save_metrics_in_test_format(
        metrics, model_name, output_path
    )
    
    print(f"Metrics saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Usage example
    example_metrics = {
        "accuracy": 0.85,
        "precision": 0.84,
        "recall": 0.86,
        "f1_score": 0.85,
        "roc_auc": 0.92,
        "confusion_matrix": [[200, 30], [25, 195]],
        "num_samples": 450
    }
    
    # Save metrics
    save_training_metrics("efficientnet_b0", example_metrics)
    
    print("Example executed successfully!")
