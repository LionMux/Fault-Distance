"""
Reusable building blocks for 1D-CNN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals.
    
    Adaptively recalibrates channel-wise feature responses by
    learning which channels are important.
    
    Paper: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
    """
    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, seq_length)
        Returns:
            x weighted by attention: (batch_size, channels, seq_length)
        """
        # Get channel-wise weights
        scale = self.fc(x)  # (batch_size, channels)
        scale = scale.unsqueeze(-1)  # (batch_size, channels, 1)
        return x * scale


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional SE-Attention.
    
    Solves vanishing gradient problem through skip connections.
    Each block learns residual = output - input instead of direct output.
    
    Structure:
        Conv1d -> BatchNorm -> ReLU -> Dropout ->
        Conv1d -> BatchNorm -> [SE-Block] -> ReLU + Skip Connection
    """
    def __init__(self, channels, kernel_size=3, dropout=0.3, use_se=True, expansion=1):
        """
        Args:
            channels: Number of channels (must be same in and out for skip connection)
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            use_se: Whether to use SE-Block
            expansion: Channel expansion factor (for bottleneck)
        """
        super().__init__()
        padding = kernel_size // 2
        mid_channels = int(channels * expansion)
        
        self.conv1 = nn.Conv1d(
            channels, mid_channels, kernel_size,
            padding=padding, bias=False
        )
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            mid_channels, channels, kernel_size,
            padding=padding, bias=False
        )
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Squeeze-and-Excitation attention (optional)
        self.se = SEBlock1D(channels) if use_se else nn.Identity()
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, seq_length)
        Returns:
            output: (batch_size, channels, seq_length)
        """
        residual = x  # Skip connection
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        
        # Add skip connection and apply activation
        out = out + residual
        out = self.relu(out)
        out = self.dropout2(out)
        
        return out


class InvertedResBlock1D(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 style).
    
    First expands channels, then compresses them.
    More efficient than standard ResBlock for resource-constrained devices.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion=6, dropout=0.3):
        """
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            expansion: Channel expansion factor
            dropout: Dropout probability
        """
        super().__init__()
        padding = kernel_size // 2
        mid_channels = in_channels * expansion
        
        # Expansion phase
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding,
                     groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Projection (compression) phase
        self.project = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        
        # Skip connection if input/output same
        self.use_skip = (in_channels == out_channels)
    
    def forward(self, x):
        residual = x if self.use_skip else None
        
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        
        if self.use_skip:
            out = out + residual
        
        return out
