"""
DomainShiftAnalyzer - Main class for domain shift analysis

This class implements methods to compare performance of models
trained on SDSS when applied to S-PLUS data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.pretrained.model_factory import create_pretrained_model, get_model_info
from model.pretrained.dataset import GalaxyPretrainedDataset, get_imagenet_transforms
from model.datasets.galaxy_dataset import GalaxyDataset
from validation.metrics import calculate_metrics
from .metrics import calculate_domain_shift_metrics, calculate_distribution_distance
from .visualization import (
    plot_domain_comparison, plot_feature_distributions, 
    plot_performance_comparison, plot_confusion_matrix_comparison
)


class DomainShiftAnalyzer:
    """
    Domain shift analyzer between SDSS and S-PLUS.
    
    This class allows comparing performance of models trained on SDSS
    when applied to S-PLUS data, identifying domain shift problems.
    """
    
    def __init__(self, model_name: str, model_path: Path, device: torch.device, output_dir: Path):
        """
        Initialize domain shift analyzer.
        
        Args:
            model_name: Model name (e.g., 'resnet50_v1')
            model_path: Path to trained model (.pth)
            device: Computing device
            output_dir: Directory to save results
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = self._load_model()
        self.model_info = get_model_info(model_name)
        
        print(f"DomainShiftAnalyzer initialized for model: {model_name}")
        print(f"Model loaded from: {model_path}")
    
    def _load_model(self) -> nn.Module:
        """Load trained model."""
        model = create_pretrained_model(self.model_name, num_classes=2, pretrained=False)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model
    
    def analyze_domain_shift(
        self,
        sdss_data_dir: Path,
        splus_data_dir: Path,
        batch_size: int = 32,
        num_workers: int = 2
    ) -> Dict[str, Any]:
        """
        Execute complete domain shift analysis.
        
        Args:
            sdss_data_dir: Directory with SDSS data
            splus_data_dir: Directory with S-PLUS data
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            
        Returns:
            Dictionary with analysis results
        """
        print("Starting domain shift analysis...")
        
        # Load datasets
        sdss_results = self._evaluate_on_domain(sdss_data_dir, "SDSS", batch_size, num_workers)
        splus_results = self._evaluate_on_domain(splus_data_dir, "S-PLUS", batch_size, num_workers)
        
        # Calculate domain shift metrics
        domain_shift_metrics = calculate_domain_shift_metrics(
            sdss_results['predictions'],
            splus_results['predictions'],
            sdss_results['features'],
            splus_results['features']
        )
        
        # Compile results
        results = {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
            'sdss_results': sdss_results,
            'splus_results': splus_results,
            'domain_shift_metrics': domain_shift_metrics,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        print(f"Domain shift analysis complete. Results in: {self.output_dir}")
        return results
    
    def _evaluate_on_domain(
        self,
        data_dir: Path,
        domain_name: str,
        batch_size: int,
        num_workers: int
    ) -> Dict[str, Any]:
        """
        Evaluate model on a specific domain.
        
        Args:
            data_dir: Directory with domain data
            domain_name: Domain name (SDSS or S-PLUS)
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            Dictionary with evaluation results
        """
        print(f"Evaluating model on {domain_name} data...")
        
        # Load dataset
        transforms = get_imagenet_transforms(
            image_size=self.model_info['input_size'],
            is_train=False
        )
        dataset = GalaxyPretrainedDataset(img_dir=str(data_dir), transform=transforms)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        # Get class names
        class_names = dataset.classes if dataset.classes else ["Regular", "Peculiar"]
        
        # Run inference
        y_true = []
        y_pred = []
        y_proba = []
        features = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc=f"Processing {domain_name}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Extract features (before classification layer)
                features_batch = self._extract_features(inputs)
                features.extend(features_batch.cpu().numpy())
                
                # Calculate probabilities and predictions
                probabilities = torch.softmax(outputs, dim=1)[:, 1]
                _, predicted = torch.max(outputs.data, 1)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_proba.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        return {
            'domain_name': domain_name,
            'data_dir': str(data_dir),
            'num_samples': len(dataset),
            'class_names': class_names,
            'predictions': {
                'y_true': y_true,
                'y_pred': y_pred,
                'y_proba': y_proba
            },
            'features': np.array(features),
            'metrics': metrics
        }
    
    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Extract features from model before classification layer.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Extracted features
        """
        # Remove classification layer and extract features
        if self.model_name.startswith('efficientnet_'):
            # EfficientNet: use features before classifier
            features = self.model.features(inputs)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
        elif self.model_name in ['resnet50_v1', 'resnet50_v2', 'resnext50']:
            # ResNet: use features before fc
            features = self.model.conv1(inputs)
            features = self.model.bn1(features)
            features = self.model.relu(features)
            features = self.model.maxpool(features)
            features = self.model.layer1(features)
            features = self.model.layer2(features)
            features = self.model.layer3(features)
            features = self.model.layer4(features)
            features = self.model.avgpool(features)
            features = torch.flatten(features, 1)
        elif self.model_name.startswith('vit_'):
            # ViT: use features before heads
            features = self.model.encoder(inputs)
            features = features[:, 0]  # CLS token
        else:
            # Fallback: use last layer before classifier
            features = inputs
        
        return features
    
    def _save_results(self, results: Dict[str, Any]):
        """Save results to JSON files."""
        import json
        
        # Save complete results
        results_path = self.output_dir / "domain_shift_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # Convert numpy arrays to lists for JSON
            json_results = self._prepare_for_json(results)
            json.dump(json_results, f, indent=4, ensure_ascii=False)
        
        # Save metrics summary
        summary_path = self.output_dir / "domain_shift_summary.json"
        summary = {
            'model_name': results['model_name'],
            'analysis_timestamp': results['analysis_timestamp'],
            'sdss_metrics': results['sdss_results']['metrics'],
            'splus_metrics': results['splus_results']['metrics'],
            'domain_shift_metrics': results['domain_shift_metrics']
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)
    
    def _prepare_for_json(self, obj: Any) -> Any:
        """Prepare data for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _generate_visualizations(self, results: Dict[str, Any]):
        """Generate result visualizations."""
        print("Generating visualizations...")
        
        # Domain comparison plot
        plot_domain_comparison(
            results['sdss_results'],
            results['splus_results'],
            self.output_dir / "domain_comparison.png"
        )
        
        # Feature distributions plot
        plot_feature_distributions(
            results['sdss_results']['features'],
            results['splus_results']['features'],
            self.output_dir / "feature_distributions.png"
        )
        
        # Performance comparison plot
        plot_performance_comparison(
            results['sdss_results']['metrics'],
            results['splus_results']['metrics'],
            self.output_dir / "performance_comparison.png"
        )
        
        # Confusion matrices plot
        plot_confusion_matrix_comparison(
            results['sdss_results']['predictions'],
            results['splus_results']['predictions'],
            results['sdss_results']['class_names'],
            self.output_dir / "confusion_matrices.png"
        )
    
    def compare_multiple_models(
        self,
        models_config: List[Dict[str, Any]],
        sdss_data_dir: Path,
        splus_data_dir: Path
    ) -> Dict[str, Any]:
        """
        Compare domain shift across multiple models.
        
        Args:
            models_config: List of model configurations
            sdss_data_dir: Directory with SDSS data
            splus_data_dir: Directory with S-PLUS data
            
        Returns:
            Comparative results
        """
        print(f"Comparing domain shift across {len(models_config)} models...")
        
        all_results = {}
        
        for config in models_config:
            model_name = config['model_name']
            model_path = config['model_path']
            
            print(f"Analyzing model: {model_name}")
            
            # Create analyzer for this model
            model_output_dir = self.output_dir / model_name
            analyzer = DomainShiftAnalyzer(
                model_name=model_name,
                model_path=Path(model_path),
                device=self.device,
                output_dir=model_output_dir
            )
            
            # Run analysis
            results = analyzer.analyze_domain_shift(sdss_data_dir, splus_data_dir)
            all_results[model_name] = results
        
        # Save general comparison
        comparison_path = self.output_dir / "models_comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(self._prepare_for_json(all_results), f, indent=4, ensure_ascii=False)
        
        return all_results
