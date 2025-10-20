"""
Factory for creating pre-trained galaxy classification models.

Supports ResNet50, EfficientNet and Vision Transformer (ViT).
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any, Optional


# Available models mapping
AVAILABLE_MODELS = {
    # EfficientNet family (B0 to B4)
    'efficientnet_b0': {
        'class': models.efficientnet_b0,
        'weights': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'classifier_layer': 'classifier.1',
        'description': 'EfficientNet-B0 - Efficient and compact CNN'
    },
    'efficientnet_b1': {
        'class': models.efficientnet_b1,
        'weights': models.EfficientNet_B1_Weights.IMAGENET1K_V2,
        'classifier_layer': 'classifier.1',
        'description': 'EfficientNet-B1 - Efficient CNN with more parameters'
    },
    'efficientnet_b2': {
        'class': models.efficientnet_b2,
        'weights': models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        'classifier_layer': 'classifier.1',
        'description': 'EfficientNet-B2 - Medium-sized efficient CNN'
    },
    'efficientnet_b3': {
        'class': models.efficientnet_b3,
        'weights': models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        'classifier_layer': 'classifier.1',
        'description': 'EfficientNet-B3 - Large-sized efficient CNN'
    },
    'efficientnet_b4': {
        'class': models.efficientnet_b4,
        'weights': models.EfficientNet_B4_Weights.IMAGENET1K_V1,
        'classifier_layer': 'classifier.1',
        'description': 'EfficientNet-B4 - Very large-sized efficient CNN'
    },
    
    # ResNet family
    'resnet50': {
        'class': models.resnet50,
        'weights': models.ResNet50_Weights.IMAGENET1K_V2,
        'classifier_layer': 'fc',
        'description': 'ResNet-50v1 - Classic CNN with 50 layers'
    },
    'resnet50_v2': {
        'class': models.resnet50,
        'weights': models.ResNet50_Weights.IMAGENET1K_V2,
        'classifier_layer': 'fc',
        'description': 'ResNet-50v2 - Classic CNN with 50 layers (version 2)'
    },
    'resnext50': {
        'class': models.resnext50_32x4d,
        'weights': models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2,
        'classifier_layer': 'fc',
        'description': 'ResNeXt-50 - CNN with grouped convolutions'
    },
    
    # Vision Transformer family
    'vit_b_16': {
        'class': models.vit_b_16,
        'weights': models.ViT_B_16_Weights.IMAGENET1K_V1,
        'classifier_layer': 'heads.head',
        'description': 'ViT-Base/16 - Vision Transformer with patch size 16'
    },
    'vit_b_32': {
        'class': models.vit_b_32,
        'weights': models.ViT_B_32_Weights.IMAGENET1K_V1,
        'classifier_layer': 'heads.head',
        'description': 'ViT-Base/32 - Vision Transformer with patch size 32'
    }
}


def get_available_models() -> Dict[str, Dict[str, Any]]:
    """
    Return information about all available models.
    
    Returns:
        Dictionary with model information
    """
    return AVAILABLE_MODELS.copy()


def create_pretrained_model(
    model_name: str,
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_rate: float = 0.0
) -> nn.Module:
    """
    Create a pre-trained model for galaxy classification.
    
    Args:
        model_name: Model name ('resnet50', 'efficientnet_b0', 'vit_b_16')
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Whether to use pre-trained weights
        freeze_backbone: Whether to freeze backbone layers
        dropout_rate: Dropout rate for the final layer
        
    Returns:
        Configured PyTorch model
        
    Raises:
        ValueError: If model name is not supported
    """
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
    
    model_info = AVAILABLE_MODELS[model_name]
    
    # Create the model
    if pretrained:
        model = model_info['class'](weights=model_info['weights'])
    else:
        model = model_info['class'](weights=None)
    
    # Configure the classification layer
    classifier_layer = model_info['classifier_layer']
    
    # EfficientNet family (B0 to B4)
    if model_name.startswith('efficientnet_'):
        # EfficientNet: replace classifier[1]
        num_ftrs = model.classifier[1].in_features
        if dropout_rate > 0:
            model.classifier[1] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    
    # ResNet family
    elif model_name in ['resnet50', 'resnet50_v2', 'resnext50']:
        # ResNet: replace fc layer
        num_ftrs = model.fc.in_features
        if dropout_rate > 0:
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Vision Transformer family
    elif model_name.startswith('vit_'):
        # ViT: replace heads.head
        num_ftrs = model.heads.head.in_features
        if dropout_rate > 0:
            model.heads.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_classes)
            )
        else:
            model.heads.head = nn.Linear(num_ftrs, num_classes)
    
    # Freeze backbone if requested
    if freeze_backbone:
        _freeze_backbone(model, model_name)
    
    return model


def _freeze_backbone(model: nn.Module, model_name: str):
    """
    Freeze backbone layers, keeping only the classifier trainable.
    
    Args:
        model: PyTorch model
        model_name: Model name
    """
    # EfficientNet family (B0 to B4)
    if model_name.startswith('efficientnet_'):
        # Freeze all layers except classifier
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
    
    # ResNet family
    elif model_name in ['resnet50', 'resnet50_v2', 'resnext50']:
        # Freeze all layers except fc
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    
    # Vision Transformer family
    elif model_name.startswith('vit_'):
        # Freeze all layers except heads
        for param in model.parameters():
            param.requires_grad = False
        for param in model.heads.parameters():
            param.requires_grad = True


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count the number of model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter count
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    Return detailed information about a model.
    
    Args:
        model_name: Model name
        
    Returns:
        Dictionary with model information
    """
    if model_name not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Model '{model_name}' not supported. Available: {available}")
    
    model_info = AVAILABLE_MODELS[model_name].copy()
    
    # Add additional information
    model_info['input_size'] = 224  # Default for all models
    model_info['pretrained_weights'] = 'ImageNet'
    
    return model_info


def print_model_summary(model: nn.Module, model_name: str):
    """
    Print a model summary.
    
    Args:
        model: PyTorch model
        model_name: Model name
    """
    params = count_parameters(model)
    model_info = get_model_info(model_name)
    
    print(f"\n{'='*60}")
    print(f"MODEL SUMMARY: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Description: {model_info['description']}")
    print(f"Pre-trained weights: {model_info['pretrained_weights']}")
    print(f"Input size: {model_info['input_size']}x{model_info['input_size']}")
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters: {params['frozen']:,}")
    print(f"{'='*60}\n")
