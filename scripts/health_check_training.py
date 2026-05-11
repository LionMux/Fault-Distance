#!/usr/bin/env python3
"""
Training health check for diploma activation export pipeline.

What it verifies:
  1) One train batch forward pass (shape sanity)
  2) One backward + optimizer step (train health)
  3) Checkpoint save/load works
  4) inference.py can load checkpoint and predict on a single CSV

Usage:
  python scripts/health_check_training.py --config configs/base.yaml
  python scripts/health_check_training.py --data-dir data/data_training
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tempfile
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam

# Ensure repo root import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config, load_config, get_config
from data.dataset import DataLoaderFactory
from models.cnn1d import CNN1D, DilatedCNN1D
from models.resnet1d import ResNet1D
from inference import FaultDistancePredictor


SIGNAL_COLS = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
DISTANCE_COL = 'distance_km'


def _build_model(cfg: Config) -> nn.Module:
    if cfg.MODEL_TYPE == 'cnn1d':
        return CNN1D(
            seq_length=cfg.SEQ_LENGTH,
            num_channels=cfg.NUM_CHANNELS,
            num_filters=cfg.NUM_FILTERS,
            kernel_size=cfg.KERNEL_SIZE,
            dropout=cfg.DROPOUT,
        )
    if cfg.MODEL_TYPE == 'dilated_cnn1d':
        return DilatedCNN1D(
            seq_length=cfg.SEQ_LENGTH,
            num_channels=cfg.NUM_CHANNELS,
            num_filters=cfg.NUM_FILTERS,
            kernel_size=cfg.KERNEL_SIZE,
            dropout=cfg.DROPOUT,
        )
    if cfg.MODEL_TYPE == 'resnet1d':
        if not hasattr(cfg, 'BASE_CHANNELS'):
            cfg.BASE_CHANNELS = cfg.NUM_FILTERS
        if not hasattr(cfg, 'DEPTH'):
            cfg.DEPTH = 3
        if not hasattr(cfg, 'DROPOUT_RATE'):
            cfg.DROPOUT_RATE = cfg.DROPOUT
        if not hasattr(cfg, 'USE_SE_BLOCK'):
            cfg.USE_SE_BLOCK = True
        if not hasattr(cfg, 'TASK'):
            cfg.TASK = 'regression'
        return ResNet1D(cfg)

    raise ValueError(f"Unknown model type: {cfg.MODEL_TYPE}")


def _pick_inference_csv() -> Optional[str]:
    # Prefer explicit test csv folder if it exists
    candidates = []
    for pat in [
        "data/data_test_csv/*.csv",
        "data/data_test/*.csv",
        "data/data_training/*.csv",
    ]:
        candidates.extend(glob.glob(pat))
    candidates = sorted(candidates)
    return candidates[0] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Training health check")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--data-dir", type=str, default=None, help="Override DATA_DIR")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--epochs", type=int, default=1, help="Not used for training; kept for future extension")
    args = parser.parse_args()

    cfg: Config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = get_config()

    if args.data_dir:
        cfg.DATA_DIR = args.data_dir
    if args.device:
        cfg.DEVICE = args.device

    device = torch.device(cfg.DEVICE)
    print(f"[HealthCheck] DEVICE={cfg.DEVICE} MODEL_TYPE={cfg.MODEL_TYPE}")

    # 1) Load data
    train_loader, test_loader, scalers = DataLoaderFactory.create_loaders(cfg.DATA_DIR, cfg)
    assert train_loader is not None
    assert len(train_loader) > 0

    # 2) Build model
    model = _build_model(cfg).to(device)
    model.train()

    # 3) One train batch: forward + backward
    optimizer = Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    criterion = {"mse": nn.MSELoss(), "mae": nn.L1Loss(), "smooth_l1": nn.SmoothL1Loss()}[cfg.LOSS_FUNCTION]

    batch_signals, batch_distances = next(iter(train_loader))
    batch_signals = batch_signals.to(device)
    batch_distances = batch_distances.to(device)

    print(f"[HealthCheck] train batch signals shape: {tuple(batch_signals.shape)} distances shape: {tuple(batch_distances.shape)}")

    # Shape sanity for CNN1D-like models: (B, C, T)
    if batch_signals.dim() == 3:
        b, c, t = batch_signals.shape
        assert c == cfg.NUM_CHANNELS, f"Expected NUM_CHANNELS={cfg.NUM_CHANNELS}, got {c}"
        assert t == cfg.SEQ_LENGTH, f"Expected SEQ_LENGTH={cfg.SEQ_LENGTH}, got {t}"
    else:
        print(f"[HealthCheck] WARNING: unexpected signals ndim={batch_signals.dim()} (expected 3 for CNN1D)")

    optimizer.zero_grad(set_to_none=True)
    preds = model(batch_signals)
    loss = criterion(preds, batch_distances)
    print(f"[HealthCheck] forward ok. loss={loss.item():.6f}")

    loss.backward()
    if cfg.GRADIENT_CLIP:
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
    optimizer.step()
    print("[HealthCheck] backward + optimizer step ok")

    # 4) Save checkpoint to temp dir, then load via inference
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "health_check_model.pth")
        checkpoint = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "scalers": scalers,
        }
        torch.save(checkpoint, ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file was not created"
        print(f"[HealthCheck] checkpoint saved -> {ckpt_path}")

        csv_path = _pick_inference_csv()
        if not csv_path:
            print("[HealthCheck] WARNING: no CSV found for inference smoke test; skipping inference step.")
            return

        print(f"[HealthCheck] inference smoke test on: {csv_path}")
        predictor = FaultDistancePredictor(ckpt_path, device=str(cfg.DEVICE))

        # has_labels: detect if distance_km exists
        # (in this repo some training CSVs include distance_km, but it's not guaranteed)
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=1)
        has_labels = DISTANCE_COL in df.columns

        _ = predictor.predict_from_csv(csv_path, has_labels=has_labels)
        print("[HealthCheck] inference ok")

    print("[HealthCheck] ALL OK")


if __name__ == "__main__":
    main()
