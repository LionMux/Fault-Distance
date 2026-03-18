"""
1D Convolutional Neural Networks for fault distance estimation.

Input tensor shape : (batch_size, NUM_CHANNELS, SEQ_LENGTH)
                     e.g. (32, 6, 400)
Output             : (batch_size, 1)  -- regression distance in km (normalized)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    """
    Simple 3-block 1D CNN for signal regression.

    Architecture:
        Block 1-3 : Conv1d -> BN -> ReLU -> MaxPool2 -> Dropout
        Head      : Flatten -> Linear(256) -> ReLU -> Dropout
                             -> Linear(128) -> ReLU -> Dropout
                             -> Linear(1)
    """

    def __init__(
        self,
        seq_length: int = 400,
        num_channels: int = 6,
        num_filters: int = 64,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        """
        Args:
            seq_length   : number of time steps in each signal
            num_channels : number of input channels (6 for Ia,Ib,Ic,Ua,Ub,Uc)
            num_filters  : base filter count (doubled each block)
            kernel_size  : convolutional kernel width
            dropout      : dropout probability
        """
        super().__init__()
        self.seq_length = seq_length
        self.num_channels = num_channels

        # Block 1
        self.conv1 = nn.Conv1d(num_channels, num_filters, kernel_size,
                               padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.pool1 = nn.MaxPool1d(2)

        # Block 2
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size,
                               padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(num_filters * 2)
        self.pool2 = nn.MaxPool1d(2)

        # Block 3
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size,
                               padding=kernel_size // 2)
        self.bn3 = nn.BatchNorm1d(num_filters * 4)
        self.pool3 = nn.MaxPool1d(2)

        # Flattened size after 3× MaxPool2: seq_length // 8
        self.flattened_size = (seq_length // 8) * (num_filters * 4)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, NUM_CHANNELS, SEQ_LENGTH)
        Returns:
            (B, 1)  -- predicted normalized distance
        """
        x = self.dropout(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DilatedCNN1D(nn.Module):
    """
    1D CNN with dilated convolutions.
    Larger effective receptive field without extra pooling.
    Better for capturing long-range patterns in fault signals.

    Dilations: [1, 2, 4, 8] by default
    """

    def __init__(
        self,
        seq_length: int = 400,
        num_channels: int = 6,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.3,
        dilations: list = None,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.num_channels = num_channels
        if dilations is None:
            dilations = [1, 2, 4, 8]

        layers = []
        in_ch = num_channels
        for i, d in enumerate(dilations):
            out_ch = num_filters * (i + 1)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size,
                          dilation=d, padding=(kernel_size - 1) * d // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
            ]
            in_ch = out_ch

        self.conv_blocks = nn.Sequential(*layers)

        # Infer flattened size with a dummy pass
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, seq_length)
            flat = self.conv_blocks(dummy).view(1, -1).size(1)
        self.flattened_size = flat

        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class CNN1DRegressor:
    """Lightweight wrapper for CNN1D with predict/save/load helpers."""

    def __init__(self, seq_length=400, num_channels=6, num_filters=64,
                 kernel_size=5, dropout=0.3, device='cpu'):
        self.device = device
        self.model = CNN1D(
            seq_length=seq_length,
            num_channels=num_channels,
            num_filters=num_filters,
            kernel_size=kernel_size,
            dropout=dropout,
        ).to(device)

    def get_model(self):
        return self.model

    def predict(self, x: torch.Tensor) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            if x.dim() == 2:           # (C, T) -> (1, C, T)
                x = x.unsqueeze(0)
            return self.model(x.to(self.device)).cpu().numpy()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved -> {path}")

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded <- {path}")
