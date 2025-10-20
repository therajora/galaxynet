"""
CNN architecture for galaxy classification.
Based on scratch.py with improvements and modularization.
"""

import torch
import torch.nn as nn


class GalaxyNetCNN(nn.Module):
    """
    Simple convolutional neural network for galaxy classification.
    
    Architecture:
    - 3 convolutional blocks with ReLU and MaxPool
    - 2 fully connected layers with dropout
    - Binary or multiclass classification
    
    Args:
        num_classes (int): Number of classes for classification
        input_size (tuple): Input image size (height, width)
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, num_classes=2, input_size=(224, 224), dropout_rate=0.5):
        super(GalaxyNetCNN, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Convolutional block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 224x224 -> 112x112
        
        # Convolutional block 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 112x112 -> 56x56
        
        # Convolutional block 3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 56x56 -> 28x28
        
        # Fully connected layers (classifier)
        # Calculate input size based on input_size
        self._calculate_fc_input_size()
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Weight initialization
        self._initialize_weights()

    def _calculate_fc_input_size(self):
        """Calculate input size for the first FC layer."""
        # Simulate a forward pass to calculate size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            x = self.pool1(self.relu1(self.conv1(dummy_input)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            self.fc_input_size = x.view(1, -1).size(1)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass of the network."""
        # Convolutional blocks
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten tensor for linear layer
        x = x.view(-1, self.fc_input_size)
        
        # Fully connected layers
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

    def get_model_info(self):
        """Returns information about the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'num_classes': self.num_classes,
            'input_size': self.input_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'fc_input_size': self.fc_input_size
        }

    def print_model_summary(self):
        """Prints a summary of the model architecture."""
        info = self.get_model_info()
        
        print("=" * 50)
        print("GALAXY NET CNN - ARCHITECTURE SUMMARY")
        print("=" * 50)
        print(f"Number of classes: {info['num_classes']}")
        print(f"Input size: {info['input_size']}")
        print(f"Dropout rate: {info['dropout_rate']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print(f"FC input size: {info['fc_input_size']}")
        print("=" * 50)
        
        # Print architecture
        print("\nARCHITECTURE:")
        print(self)
        print("=" * 50)


class GalaxyNetCNNV2(nn.Module):
    """
    Improved CNN version with more layers and BatchNorm.
    
    Args:
        num_classes (int): Number of classes for classification
        input_size (tuple): Input image size
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, num_classes=2, input_size=(224, 224), dropout_rate=0.5):
        super(GalaxyNetCNNV2, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Calculate FC input size
        self._calculate_fc_input_size()
        
        # Classifier
        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.relu5 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.relu6 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, num_classes)
        
        self._initialize_weights()

    def _calculate_fc_input_size(self):
        """Calculate input size for the first FC layer."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *self.input_size)
            x = self.pool1(self.relu1(self.bn1(self.conv1(dummy_input))))
            x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
            x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
            x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
            self.fc_input_size = x.view(1, -1).size(1)

    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass of the network."""
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        x = x.view(-1, self.fc_input_size)
        
        x = self.relu5(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu6(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x


def create_model(model_type='simple', num_classes=2, input_size=(224, 224), dropout_rate=0.5):
    """
    Factory function to create models.
    
    Args:
        model_type (str): Model type ('simple' or 'v2')
        num_classes (int): Number of classes
        input_size (tuple): Image size
        dropout_rate (float): Dropout rate
    
    Returns:
        nn.Module: Created model
    """
    if model_type == 'simple':
        return GalaxyNetCNN(num_classes, input_size, dropout_rate)
    elif model_type == 'v2':
        return GalaxyNetCNNV2(num_classes, input_size, dropout_rate)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def count_parameters(model):
    """Count the number of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }
