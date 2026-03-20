"""
Reusable building blocks for 1D-CNN architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals.
    """
    def __init__(self, channels, reduction=16):
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
        scale = self.fc(x)
        scale = scale.unsqueeze(-1)
        return x * scale


class ResBlock1D(nn.Module):
    """
    1D Residual Block with optional SE-Attention.
    """
    def __init__(self, channels, kernel_size=3, dropout=0.3, use_se=True, expansion=1):
        super().__init__()
        padding = kernel_size // 2
        mid_channels = int(channels * expansion)
        self.conv1 = nn.Conv1d(channels, mid_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(mid_channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout2 = nn.Dropout(dropout)
        self.se = SEBlock1D(channels) if use_se else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = out + residual
        out = self.relu(out)
        out = self.dropout2(out)
        return out


class InvertedResBlock1D(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 style).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion=6, dropout=0.3):
        super().__init__()
        padding = kernel_size // 2
        mid_channels = in_channels * expansion
        self.expand = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size, padding=padding, groups=mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        self.project = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        self.use_skip = (in_channels == out_channels)

    def forward(self, x):
        residual = x if self.use_skip else None
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_skip:
            out = out + residual
        return out
