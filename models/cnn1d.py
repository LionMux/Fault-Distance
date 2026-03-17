"""
1D Convolutional Neural Networks for fault distance estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class CNN1D(nn.Module):
    """
    Simple 1D CNN for signal regression.
    
    Architecture:
        - Conv1d layers with ReLU activation
        - Batch normalization
        - MaxPooling to reduce dimensionality
        - Fully connected layers for regression
    """
    
    def __init__(self, seq_length=300, num_filters=64, kernel_size=5, 
                 dropout=0.3, num_fc_layers=2):
        """
        Args:
            seq_length (int): Input signal length
            num_filters (int): Initial number of filters
            kernel_size (int): Conv kernel size
            dropout (float): Dropout rate
            num_fc_layers (int): Number of fully connected layers
        """
        super(CNN1D, self).__init__()
        
        self.seq_length = seq_length
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        # ============ CONVOLUTIONAL BLOCKS ============
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Conv1d(
            in_channels=1, 
            out_channels=num_filters, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(
            in_channels=num_filters, 
            out_channels=num_filters*2, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.bn2 = nn.BatchNorm1d(num_filters*2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Block 3
        self.conv3 = nn.Conv1d(
            in_channels=num_filters*2, 
            out_channels=num_filters*4, 
            kernel_size=kernel_size, 
            padding=kernel_size//2
        )
        self.bn3 = nn.BatchNorm1d(num_filters*4)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size after convolutions and pooling
        # seq_length -> seq_length (conv) -> seq_length/2 (pool1) 
        #            -> seq_length/2 (conv) -> seq_length/4 (pool2)
        #            -> seq_length/4 (conv) -> seq_length/8 (pool3)
        self.flattened_size = (seq_length // 8) * (num_filters * 4)
        
        # ============ FULLY CONNECTED LAYERS ============
        fc_layers = []
        input_size = self.flattened_size
        
        for i in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(input_size, 256))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            input_size = 256
        
        # Output layer (regression: 1 value)
        fc_layers.append(nn.Linear(input_size, 128))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        fc_layers.append(nn.Linear(128, 1))  # Single output for regression
        
        self.fc = nn.Sequential(*fc_layers)
        
        # Dropout between conv and fc
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, 1, seq_length) - Input signal
        
        Returns:
            (batch_size, 1) - Predicted distances
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x
    
    def get_num_parameters(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_parameters(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CNN1DRegressor:
    """
    Wrapper for CNN1D with training utilities.
    """
    
    def __init__(self, seq_length=300, num_filters=64, kernel_size=5, 
                 dropout=0.3, device='cpu'):
        """
        Args:
            seq_length (int): Input signal length
            num_filters (int): Initial number of filters
            kernel_size (int): Conv kernel size
            dropout (float): Dropout rate
            device: torch device
        """
        self.device = device
        self.model = CNN1D(
            seq_length=seq_length,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout
        ).to(device)
    
    def get_model(self):
        """Get PyTorch model."""
        return self.model
    
    def predict(self, x):
        """
        Make predictions (inference mode).
        
        Args:
            x: Input signal (batch_size, 1, seq_length) or (1, seq_length)
        
        Returns:
            Predicted distances
        """
        self.model.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            x = x.to(self.device)
            output = self.model(x)
        return output.cpu().numpy()
    
    def save(self, path):
        """Save model weights."""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model weights."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"✓ Model loaded from {path}")


class DilatedCNN1D(nn.Module):
    """
    1D CNN with dilated convolutions for larger receptive field.
    Better for capturing long-range dependencies in signals.
    """
    
    def __init__(self, seq_length=300, num_filters=64, kernel_size=3, 
                 dropout=0.3, dilations=[1, 2, 4, 8]):
        """
        Args:
            seq_length (int): Input signal length
            num_filters (int): Number of filters per layer
            kernel_size (int): Conv kernel size
            dropout (float): Dropout rate
            dilations (list): Dilation rates for each layer
        """
        super(DilatedCNN1D, self).__init__()
        
        self.seq_length = seq_length
        
        # Build dilated conv blocks
        layers = []
        in_channels = 1
        
        for i, dilation in enumerate(dilations):
            out_channels = num_filters * (i + 1)
            
            layers.append(nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=(kernel_size - 1) * dilation // 2
            ))
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.MaxPool1d(kernel_size=2))
            
            in_channels = out_channels
        
        self.conv_blocks = nn.Sequential(*layers)
        
        # Calculate flattened size
        self.flattened_size = self._calculate_flattened_size()
        
        # Fully connected
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def _calculate_flattened_size(self):
        """Calculate flattened size by running dummy forward pass."""
        x = torch.randn(1, 1, self.seq_length)
        x = self.conv_blocks(x)
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, 1, seq_length)
        
        Returns:
            (batch_size, 1) - Predicted distances
        """
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
