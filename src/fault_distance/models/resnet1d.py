"""
1D Residual Network for Short-Circuit Fault Distance Detection.

Input  : (batch_size, NUM_CHANNELS, SEQ_LENGTH)  e.g. (32, 6, 400)
Output : (batch_size, 1)  regression -- normalized fault distance

Based on:
    - ResNet (He et al., 2015)
    - Squeeze-and-Excitation Networks (Hu et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ResBlock1D, SEBlock1D


class FaultResNet1D(nn.Module):
    """
    1D ResNet for predicting distance to short-circuit fault.

    Stages (depth 1-4):
        Stem  : Conv7 -> BN -> ReLU -> MaxPool2  (T -> T/2)
        Stage1: 2x ResBlock  (no channel change)  + MaxPool2  -> T/4
        Stage2: channel C->2C  + 2x ResBlock      + MaxPool2  -> T/8
        Stage3: channel 2C->4C + 2x ResBlock      + MaxPool2  -> T/16
        Stage4: channel 4C->8C + 2x ResBlock      + MaxPool2  -> T/32
        GAP   : (B, C_final, N) -> (B, C_final, 1)
        Head  : Linear -> BN -> ReLU -> Dropout  x2  -> Linear(1)
    """

    def __init__(self, cfg):
        """
        Args:
            cfg: Config object. Required attributes:
                 NUM_CHANNELS, BASE_CHANNELS, DEPTH, DROPOUT_RATE,
                 USE_SE_BLOCK, TASK ("regression" | "classification")
        """
        super().__init__()
        self.cfg = cfg
        C = cfg.BASE_CHANNELS
        num_channels = cfg.NUM_CHANNELS  # e.g. 6

        # ---- Stem ----
        self.stem = nn.Sequential(
            nn.Conv1d(num_channels, C, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(C),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        # ---- Residual stages (depth-controlled) ----
        if cfg.DEPTH >= 1:
            self.stage1 = self._make_stage(C, C, 2, cfg.DROPOUT_RATE)
            self.pool1 = nn.MaxPool1d(2, 2)

        if cfg.DEPTH >= 2:
            self.stage2 = self._make_stage(C, C * 2, 2, cfg.DROPOUT_RATE)
            self.pool2 = nn.MaxPool1d(2, 2)

        if cfg.DEPTH >= 3:
            self.stage3 = self._make_stage(C * 2, C * 4, 2, cfg.DROPOUT_RATE)
            self.pool3 = nn.MaxPool1d(2, 2)

        if cfg.DEPTH >= 4:
            self.stage4 = self._make_stage(C * 4, C * 8, 2, cfg.DROPOUT_RATE)
            self.pool4 = nn.MaxPool1d(2, 2)

        self.gap = nn.AdaptiveAvgPool1d(1)
        final_ch = C * (2 ** (cfg.DEPTH - 1))

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, 256),
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

        if cfg.TASK == 'regression':
            self.output = nn.Linear(64, 1)
        else:
            num_classes = getattr(cfg, 'NUM_CLASSES', 10)
            self.output = nn.Linear(64, num_classes)

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, num_blocks, dropout):
        layers = []
        if in_ch != out_ch:
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ))
        for _ in range(num_blocks):
            layers.append(ResBlock1D(
                channels=out_ch,
                kernel_size=3,
                dropout=dropout,
                use_se=self.cfg.USE_SE_BLOCK
            ))
        return nn.Sequential(*layers)

    def _init_weights(self):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        if self.cfg.DEPTH >= 1:
            x = self.pool1(self.stage1(x))
        if self.cfg.DEPTH >= 2:
            x = self.pool2(self.stage2(x))
        if self.cfg.DEPTH >= 3:
            x = self.pool3(self.stage3(x))
        if self.cfg.DEPTH >= 4:
            x = self.pool4(self.stage4(x))
        x = self.gap(x)
        return self.output(self.head(x))

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias used in train.py
ResNet1D = FaultResNet1D


def create_model(cfg):
    model = FaultResNet1D(cfg)
    print(f"\nFaultResNet1D | params: {model.get_num_parameters():,} | depth: {cfg.DEPTH}")
    return model
