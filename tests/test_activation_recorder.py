"""
Unit tests for activation recorder/export utilities.

Run:
  pytest -q tests/test_activation_recorder.py
"""

from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn

from utils.activation_export import export_layer_snapshots
from utils.activation_recorder import ActivationRecorder, LayerRecordMode


class _TinyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Keep names stable: "conv1", "conv2"
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (B, C, 1)
            nn.Flatten(),               # (B, C)
            nn.Linear(2, 1),          # (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, T)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return self.head(x)


def _make_input(batch_size: int = 2, t: int = 50) -> torch.Tensor:
    # Deterministic signals.
    rng = np.random.default_rng(123)
    x = rng.normal(0, 1, size=(batch_size, 1, t)).astype(np.float32)
    return torch.tensor(x)


def test_recorder_capture_and_stats_are_finite():
    model = _TinyConvModel()
    model.eval()

    layer_names = ["conv1", "conv2"]
    recorder = ActivationRecorder(model, layer_names=layer_names, mode=LayerRecordMode())

    inputs = _make_input()
    snapshots = recorder.capture(inputs, epoch=0, probe_id="val_0", max_channels_to_plot=8)

    # Two hooks => two snapshots
    assert len(snapshots) == 2
    assert {s.layer_name for s in snapshots} == {"conv1", "conv2"}

    for snap in snapshots:
        assert snap.activation_shape[0] == inputs.shape[0]  # B
        assert "mean_abs" in snap.activation_stats
        assert "sparsity" in snap.activation_stats

        # finite sanity
        for k, v in snap.activation_stats.items():
            assert np.isfinite(v), f"Non-finite activation stat: {k}={v}"

        # curve should exist for conv outputs (B,C,T)
        assert snap.activation_curve is not None
        assert "time_axis" in snap.activation_curve
        assert "values" in snap.activation_curve
        assert len(snap.activation_curve["time_axis"]) == len(snap.activation_curve["values"])

    recorder.remove()


def test_recorder_remove_disables_capture():
    model = _TinyConvModel()
    model.eval()

    recorder = ActivationRecorder(model, layer_names=["conv1"], mode=LayerRecordMode())
    recorder.remove()

    inputs = _make_input()
    # After hooks are removed, recorder.capture will still run model, but hook outputs will remain None
    # and snapshots should be empty (skipping non-tensor outputs).
    snapshots = recorder.capture(inputs, epoch=0, probe_id="val_0", max_channels_to_plot=8)
    assert snapshots == []


def test_export_layer_snapshots_writes_png_and_csv():
    model = _TinyConvModel()
    model.eval()

    recorder = ActivationRecorder(model, layer_names=["conv1"], mode=LayerRecordMode())
    inputs = _make_input()

    snapshots = recorder.capture(inputs, epoch=1, probe_id="val_0", max_channels_to_plot=8)

    with tempfile.TemporaryDirectory() as tmp:
        export_layer_snapshots(
            snapshots,
            out_dir=tmp,
            formats=["png", "csv"],
            dpi=100,
            max_channels_to_plot=8,
        )

        csv_path = os.path.join(tmp, "snapshots.csv")
        assert os.path.exists(csv_path)

        pngs = glob.glob(os.path.join(tmp, "epoch_001__layer_*__val_0.png"))
        # Expect at least one PNG
        assert len(pngs) >= 1

    recorder.remove()
