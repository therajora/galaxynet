"""
Prediction module for galaxy classification models.
"""

import os
import torch
import cv2
from torchvision import transforms
from typing import Dict, List
import random


class GalaxyPredictor:
    """
    Class for making predictions with trained galaxy classification models.
    """
    
    def __init__(self, model, class_names, mean, std, device=None):
        """
        Initialize the predictor.
        
        Args:
            model: Trained PyTorch model
            class_names: List with class names
            mean: Mean for normalization
            std: Standard deviation for normalization
            device: Device for prediction
        """
        self.model = model
        self.class_names = class_names
        self.mean = mean
        self.std = std
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Create transformation for prediction
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        print(f"Predictor initialized - Device: {self.device}")
        print(f"Classes: {self.class_names}")

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess an image for prediction.
        
        Args:
            image_path: Path to the image
            
        Returns:
            torch.Tensor: Preprocessed image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transformations
        image_tensor = self.transform(image)
        
        # Add batch dimension
        input_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return input_tensor

    def predict_single(self, image_path: str) -> Dict:
        """
        Make prediction for a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Prediction results
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image_path)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get main prediction
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            
            # Get all probabilities
            all_probabilities = {
                self.class_names[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        
        return {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'confidence': confidence_score,
            'all_probabilities': all_probabilities,
            'predicted_idx': predicted_idx.item()
        }

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """
        Make prediction for a batch of images.
        
        Args:
            image_paths: List of paths to images
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results

    def predict_random_sample(self, data_dir: str, num_samples: int = 1) -> List[Dict]:
        """
        Make prediction on random samples from the dataset.
        
        Args:
            data_dir: Dataset directory
            num_samples: Number of random samples
            
        Returns:
            list: List of prediction results
        """
        # Collect all images
        all_images = []
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_images.append(os.path.join(class_dir, file_name))
        
        # Select random samples
        if len(all_images) < num_samples:
            num_samples = len(all_images)
        
        random_samples = random.sample(all_images, num_samples)
        
        # Make predictions
        results = self.predict_batch(random_samples)
        
        return results

    def print_prediction_result(self, result: Dict, show_all_probs: bool = True):
        """
        Print prediction result in formatted way.
        
        Args:
            result: Prediction result
            show_all_probs: Whether to show all probabilities
        """
        if 'error' in result:
            print(f"Error: {result['error']}")
            return
        
        print("\n" + "=" * 50)
        print("PREDICTION RESULT")
        print("=" * 50)
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if show_all_probs:
            print("\nProbabilities for all classes:")
            sorted_probs = sorted(
                result['all_probabilities'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            for class_name, prob in sorted_probs:
                marker = ">" if class_name == result['predicted_class'] else " "
                print(f"{marker} {class_name}: {prob:.2%}")
        
        print("=" * 50)

    def evaluate_accuracy(self, data_dir: str, num_samples: int = 100) -> Dict:
        """
        Evaluate model accuracy on dataset samples.
        
        Args:
            data_dir: Dataset directory
            num_samples: Number of samples for evaluation
            
        Returns:
            dict: Evaluation metrics
        """
        # Collect balanced samples
        samples_per_class = num_samples // len(self.class_names)
        all_samples = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                class_images = [
                    os.path.join(class_dir, f) 
                    for f in os.listdir(class_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                ]
                
                if len(class_images) >= samples_per_class:
                    selected = random.sample(class_images, samples_per_class)
                else:
                    selected = class_images
                
                for img_path in selected:
                    all_samples.append((img_path, class_name))
        
        # Make predictions
        correct = 0
        total = len(all_samples)
        class_correct = {cls: 0 for cls in self.class_names}
        class_total = {cls: 0 for cls in self.class_names}
        
        print(f"Evaluating model on {total} samples...")
        
        for img_path, true_class in all_samples:
            try:
                result = self.predict_single(img_path)
                predicted_class = result['predicted_class']
                
                class_total[true_class] += 1
                
                if predicted_class == true_class:
                    correct += 1
                    class_correct[true_class] += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        # Calculate metrics
        overall_accuracy = correct / total if total > 0 else 0
        class_accuracies = {
            cls: class_correct[cls] / class_total[cls] if class_total[cls] > 0 else 0
            for cls in self.class_names
        }
        
        metrics = {
            'overall_accuracy': overall_accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total,
            'correct_predictions': correct,
            'class_totals': class_total,
            'class_correct': class_correct
        }
        
        return metrics

    def print_evaluation_results(self, metrics: Dict):
        """
        Print evaluation results in formatted way.
        
        Args:
            metrics: Evaluation metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Overall Accuracy: {metrics['overall_accuracy']:.2%}")
        print(f"Total Samples: {metrics['total_samples']}")
        print(f"Correct Predictions: {metrics['correct_predictions']}")
        
        print("\nAccuracy by Class:")
        for class_name, accuracy in metrics['class_accuracies'].items():
            correct = metrics['class_correct'][class_name]
            total = metrics['class_totals'][class_name]
            print(f"  {class_name}: {accuracy:.2%} ({correct}/{total})")
        
        print("=" * 60)


def load_predictor_from_artifacts(model_path: str, artifacts_path: str, 
                                 device=None) -> GalaxyPredictor:
    """
    Load a predictor from saved artifacts.
    
    Args:
        model_path: Path to the model
        artifacts_path: Path to the artifacts
        device: Device for prediction
        
    Returns:
        GalaxyPredictor: Loaded predictor
    """
    from .model import GalaxyNetCNN
    
    # Load artifacts
    artifacts = torch.load(artifacts_path, map_location='cpu')
    class_names = artifacts['class_names']
    mean = artifacts['mean']
    std = artifacts['std']
    
    # Create model
    num_classes = len(class_names)
    model = GalaxyNetCNN(num_classes=num_classes)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Create predictor
    predictor = GalaxyPredictor(model, class_names, mean, std, device)
    
    print(f"Predictor loaded from:")
    print(f"  Model: {model_path}")
    print(f"  Artifacts: {artifacts_path}")
    
    return predictor
