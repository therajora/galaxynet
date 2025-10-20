"""
Version manager for trained models.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Optional


class ModelVersionManager:
    """
    Manages versions of trained models and their logs.
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize the version manager.
        
        Args:
            results_dir: Base results directory
        """
        self.results_dir = Path(results_dir)
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.metrics_dir = self.results_dir / "metrics"
        
        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Version registry file
        self.versions_file = self.results_dir / "model_versions.json"
        
        # Load existing versions
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict:
        """Load existing versions from file."""
        if self.versions_file.exists():
            with open(self.versions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def _save_versions(self):
        """Save versions to file."""
        with open(self.versions_file, 'w', encoding='utf-8') as f:
            json.dump(self.versions, f, indent=2, ensure_ascii=False)

    def create_new_version(self, model_name: str, model_type: str, 
                          dataset_name: str, description: str = "") -> str:
        """
        Create a new model version.
        
        Args:
            model_name: Model name
            model_type: Model type (simple, v2, etc.)
            dataset_name: Dataset name used
            description: Version description
            
        Returns:
            str: Created version ID
        """
        # Generate version ID based on timestamp
        timestamp = datetime.datetime.now()
        version_id = f"{model_name}_{model_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create version directory
        version_dir = self.models_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Version information
        version_info = {
            'version_id': version_id,
            'model_name': model_name,
            'model_type': model_type,
            'dataset_name': dataset_name,
            'description': description,
            'created_at': timestamp.isoformat(),
            'status': 'training',
            'model_path': str(version_dir / "model.pth"),
            'artifacts_path': str(version_dir / "artifacts.pth"),
            'logs_path': str(self.logs_dir / f"{version_id}.log"),
            'metrics_path': str(self.metrics_dir / f"{version_id}_metrics.json")
        }
        
        # Add to versions list
        if model_name not in self.versions:
            self.versions[model_name] = []
        
        self.versions[model_name].append(version_info)
        
        # Save versions
        self._save_versions()
        
        print(f"New version created: {version_id}")
        print(f"Directory: {version_dir}")
        
        return version_id

    def update_version_status(self, version_id: str, status: str, 
                            metrics: Optional[Dict] = None):
        """
        Update version status.
        
        Args:
            version_id: Version ID
            status: New status (training, completed, failed)
            metrics: Model metrics (optional)
        """
        for model_name, versions in self.versions.items():
            for version in versions:
                if version['version_id'] == version_id:
                    version['status'] = status
                    version['updated_at'] = datetime.datetime.now().isoformat()
                    
                    if metrics:
                        version['metrics'] = metrics
                        # Save metrics in separate file
                        metrics_file = Path(version['metrics_path'])
                        with open(metrics_file, 'w', encoding='utf-8') as f:
                            json.dump(metrics, f, indent=2, ensure_ascii=False)
                    
                    self._save_versions()
                    print(f"Status updated for {version_id}: {status}")
                    return
        
        print(f"Version not found: {version_id}")

    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """
        Get information for a specific version.
        
        Args:
            version_id: Version ID
            
        Returns:
            dict: Version information or None if not found
        """
        for model_name, versions in self.versions.items():
            for version in versions:
                if version['version_id'] == version_id:
                    return version
        return None

    def list_versions(self, model_name: Optional[str] = None) -> List[Dict]:
        """
        List available versions.
        
        Args:
            model_name: Model name (optional, lists all if None)
            
        Returns:
            list: List of versions
        """
        if model_name:
            return self.versions.get(model_name, [])
        
        all_versions = []
        for versions in self.versions.values():
            all_versions.extend(versions)
        
        # Sort by creation date (most recent first)
        all_versions.sort(key=lambda x: x['created_at'], reverse=True)
        return all_versions

    def get_latest_version(self, model_name: str) -> Optional[Dict]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Model name
            
        Returns:
            dict: Latest version or None if not found
        """
        versions = self.versions.get(model_name, [])
        if not versions:
            return None
        
        # Return the most recent version
        return max(versions, key=lambda x: x['created_at'])

    def print_versions_summary(self, model_name: Optional[str] = None):
        """
        Print versions summary.
        
        Args:
            model_name: Model name (optional)
        """
        versions = self.list_versions(model_name)
        
        if not versions:
            print("No versions found.")
            return
        
        print("\n" + "=" * 80)
        print("MODEL VERSIONS SUMMARY")
        print("=" * 80)
        
        for version in versions:
            status_emoji = {
                'training': '[TRAINING]',
                'completed': '[COMPLETED]',
                'failed': '[FAILED]'
            }.get(version['status'], '[UNKNOWN]')
            
            print(f"{status_emoji} {version['version_id']}")
            print(f"   Model: {version['model_name']} ({version['model_type']})")
            print(f"   Dataset: {version['dataset_name']}")
            print(f"   Status: {version['status']}")
            print(f"   Created: {version['created_at']}")
            
            if 'description' in version and version['description']:
                print(f"   Description: {version['description']}")
            
            if 'metrics' in version:
                metrics = version['metrics']
                if 'best_val_accuracy' in metrics:
                    print(f"   Best Accuracy: {metrics['best_val_accuracy']:.2%}")
                if 'final_epoch' in metrics:
                    print(f"   Epochs: {metrics['final_epoch']}")
            
            print()

    def cleanup_old_versions(self, model_name: str, keep_count: int = 5):
        """
        Remove old versions, keeping only the most recent ones.
        
        Args:
            model_name: Model name
            keep_count: Number of versions to keep
        """
        if model_name not in self.versions:
            print(f"Model not found: {model_name}")
            return
        
        versions = self.versions[model_name]
        if len(versions) <= keep_count:
            print(f"No versions to remove. Total: {len(versions)}")
            return
        
        # Sort by creation date
        versions.sort(key=lambda x: x['created_at'], reverse=True)
        
        # Remove old versions
        versions_to_remove = versions[keep_count:]
        for version in versions_to_remove:
            version_id = version['version_id']
            
            # Remove files
            model_path = Path(version['model_path'])
            artifacts_path = Path(version['artifacts_path'])
            logs_path = Path(version['logs_path'])
            metrics_path = Path(version['metrics_path'])
            
            if model_path.exists():
                model_path.unlink()
            if artifacts_path.exists():
                artifacts_path.unlink()
            if logs_path.exists():
                logs_path.unlink()
            if metrics_path.exists():
                metrics_path.unlink()
            
            # Remove version directory
            version_dir = model_path.parent
            if version_dir.exists() and not any(version_dir.iterdir()):
                version_dir.rmdir()
            
            print(f"Version removed: {version_id}")
        
        # Update versions list
        self.versions[model_name] = versions[:keep_count]
        self._save_versions()
        
        print(f"Cleanup completed. Kept {keep_count} most recent versions.")


def get_version_manager() -> ModelVersionManager:
    """Return version manager instance."""
    return ModelVersionManager()
