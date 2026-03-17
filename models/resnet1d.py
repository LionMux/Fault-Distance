"""
1D Residual Network for Short-Circuit Fault Distance Detection

Architecture:
    - Stem: Conv + BN + ReLU + MaxPool
    - Residual Stages: Progressive downsampling with residual blocks
    - SE-Attention: Channel-wise attention
    - Head: Global average pooling + FC layers for regression

Based on:
    - ResNet (He et al., 2015)
    - Squeeze-and-Excitation Networks (Hu et al., 2018)
    - 1D-CNN for time series (Wang et al., 2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock1D, SEBlock1D


class FaultResNet1D(nn.Module):
    """
    1D Residual Network for predicting distance to short-circuit fault.
    
    Input: Oscillogram signals (300 samples = 100ms pre-fault + 200ms fault)
    Output: Fault distance in kilometers (or classification if configured)
    
    Expected input shape: (batch_size, num_channels, seq_length)
    Example: (32, 1, 300) for 32 samples, 1 signal channel, 300 time points
    """
    
    def __init__(self, cfg):
        """
        Args:
            cfg: Config object with model parameters
        """
        super().__init__()
        self.cfg = cfg
        C = cfg.BASE_CHANNELS  # Usually 64
        
        # ============ STEM: Initial layer ============
        # Purpose: Extract initial features and reduce sequence length
        self.stem = nn.Sequential(
            nn.Conv1d(
                cfg.NUM_CHANNELS, C,
                kernel_size=7, stride=1, padding=3,
                bias=False
            ),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)  # 300 -> 150
        )
        
        # ============ RESIDUAL STAGES ============
        # Each stage: ResBlock + Progressive channel increase + Downsampling
        
        if cfg.DEPTH >= 1:
            self.stage1 = self._make_stage(
                in_channels=C,
                out_channels=C,
                num_blocks=2,
                stride=1,
                dropout=cfg.DROPOUT_RATE
            )
            self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 150 -> 75
        
        if cfg.DEPTH >= 2:
            self.stage2 = self._make_stage(
                in_channels=C,
                out_channels=C * 2,
                num_blocks=2,
                stride=1,
                dropout=cfg.DROPOUT_RATE
            )
            self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 75 -> 37
        
        if cfg.DEPTH >= 3:
            self.stage3 = self._make_stage(
                in_channels=C * 2,
                out_channels=C * 4,
                num_blocks=2,
                stride=1,
                dropout=cfg.DROPOUT_RATE
            )
            self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 37 -> 18
        
        if cfg.DEPTH >= 4:
            self.stage4 = self._make_stage(
                in_channels=C * 4,
                out_channels=C * 8,
                num_blocks=2,
                stride=1,
                dropout=cfg.DROPOUT_RATE
            )
            self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 18 -> 9
        
        # ============ ADAPTIVE POOLING ============
        # Convert (batch, channels, seq) -> (batch, channels, 1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        # ============ CLASSIFICATION/REGRESSION HEAD ============
        # Determine final channel size based on depth
        final_channels = C * (2 ** (cfg.DEPTH - 1))
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT_RATE),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.DROPOUT_RATE),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # Output layer
        if cfg.TASK == "regression":
            self.output = nn.Linear(64, 1)  # Single value: distance in km
        else:  # classification
            num_classes = cfg.NUM_CLASSES if hasattr(cfg, 'NUM_CLASSES') else 10
            self.output = nn.Linear(64, num_classes)
        
        # ============ INITIALIZATION ============
        self._init_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks, stride, dropout):
        """
        Create a residual stage with multiple ResBlocks.
        
        If in_channels != out_channels, first block changes dimensions.
        """
        layers = []
        
        # First block: possibly change channels (1x1 conv if needed)
        if in_channels != out_channels:
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Add residual blocks
        for i in range(num_blocks):
            layers.append(
                ResBlock1D(
                    channels=out_channels,
                    kernel_size=3,
                    dropout=dropout,
                    use_se=self.cfg.USE_SE_BLOCK
                )
            )
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """
        Initialize weights using Kaiming initialization.
        Important for ReLU networks to prevent vanishing gradients.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor (batch_size, num_channels, seq_length)
               Example: (32, 1, 300)
        
        Returns:
            output: Prediction (batch_size, 1) for regression
                   or (batch_size, num_classes) for classification
        """
        # Stem
        x = self.stem(x)  # (B, 64, 150)
        
        # Stage 1
        if self.cfg.DEPTH >= 1:
            x = self.stage1(x)  # (B, 64, 150)
            x = self.pool1(x)   # (B, 64, 75)
        
        # Stage 2
        if self.cfg.DEPTH >= 2:
            x = self.stage2(x)  # (B, 128, 75)
            x = self.pool2(x)   # (B, 128, 37)
        
        # Stage 3
        if self.cfg.DEPTH >= 3:
            x = self.stage3(x)  # (B, 256, 37)
            x = self.pool3(x)   # (B, 256, 18)
        
        # Stage 4
        if self.cfg.DEPTH >= 4:
            x = self.stage4(x)  # (B, 512, 18)
            x = self.pool4(x)   # (B, 512, 9)
        
        # Global Average Pooling
        x = self.gap(x)  # (B, C, 1)
        
        # Classification/Regression Head
        x = self.head(x)  # (B, 64)
        x = self.output(x)  # (B, 1) or (B, num_classes)
        
        return x
    
    def get_num_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_backbone(self):
        """Freeze all layers except head for transfer learning."""
        for name, param in self.named_parameters():
            if 'head' not in name and 'output' not in name:
                param.requires_grad = False
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True


def create_model(cfg):
    """
    Factory function to create model.
    
    Args:
        cfg: Config object
    
    Returns:
        FaultResNet1D model
    """
    model = FaultResNet1D(cfg)
    print(f"\n✅ Model created: FaultResNet1D")
    print(f"   Total parameters: {model.get_num_parameters():,}")
    print(f"   Depth: {cfg.DEPTH} stages")
    print(f"   Base channels: {cfg.BASE_CHANNELS}")
    print(f"   Task: {cfg.TASK}")
    return model
