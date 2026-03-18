"""
Configuration settings for training and inference.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Training configuration."""

    # ============ DEVICE ============
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============ DATA PATHS ============
    # Directory containing oscillogram CSV files (one file per fault event)
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'oscillograms')
    TRAIN_SPLIT: float = 0.8  # 80% train, 20% test

    # ============ DATA FORMAT ============
    # Each CSV file = one oscillogram sample.
    # Columns: distance_km, CT1IA, CT1IB, CT1IC, S1)BUS1UA, S1)BUS1UB, S1)BUS1UC
    # Rows: time steps of the signal window
    NUM_CHANNELS: int = 6        # Number of signal channels (Ia, Ib, Ic, Ua, Ub, Uc)
    SEQ_LENGTH: int = 400        # Expected number of time steps per file
    NORMALIZE_DATA: bool = True  # Per-channel StandardScaler normalization

    # ============ MODEL ARCHITECTURE ============
    MODEL_TYPE: str = 'cnn1d'   # 'cnn1d', 'dilated_cnn1d', 'resnet1d'
    NUM_FILTERS: int = 64        # Initial number of filters
    KERNEL_SIZE: int = 5         # Conv kernel size
    DROPOUT: float = 0.3         # Dropout rate

    # ============ TRAINING HYPERPARAMETERS ============
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5   # L2 regularization

    # Learning rate scheduler
    SCHEDULER_TYPE: Optional[str] = 'cosine'  # 'cosine', 'linear', 'exponential', None
    WARMUP_EPOCHS: int = 5

    # ============ OPTIMIZATION ============
    OPTIMIZER: str = 'adam'          # 'adam', 'sgd', 'adamw'
    LOSS_FUNCTION: str = 'smooth_l1' # 'mse', 'mae', 'smooth_l1'
    GRADIENT_CLIP: Optional[float] = 1.0

    # ============ EARLY STOPPING ============
    EARLY_STOPPING: bool = True
    PATIENCE: int = 15
    MIN_DELTA: float = 1e-4

    # ============ CHECKPOINTING ============
    SAVE_DIR: str = 'checkpoints'
    SAVE_BEST_ONLY: bool = True
    SAVE_EVERY_N_EPOCHS: int = 10

    # ============ LOGGING ============
    LOG_DIR: str = 'logs'
    LOG_EVERY_N_BATCHES: int = 50
    SEED: int = 42

    # ============ VALIDATION ============
    VALIDATE_EVERY_N_EPOCHS: int = 1

    def __post_init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.print_config()

    def print_config(self):
        print("\n" + "=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        sections = {
            'DEVICE': ['DEVICE'],
            'DATA': ['DATA_DIR', 'TRAIN_SPLIT', 'NUM_CHANNELS', 'SEQ_LENGTH', 'NORMALIZE_DATA'],
            'MODEL': ['MODEL_TYPE', 'NUM_FILTERS', 'KERNEL_SIZE', 'DROPOUT'],
            'TRAINING': ['BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY',
                         'SCHEDULER_TYPE', 'WARMUP_EPOCHS'],
            'OPTIMIZATION': ['OPTIMIZER', 'LOSS_FUNCTION', 'GRADIENT_CLIP'],
            'EARLY_STOPPING': ['EARLY_STOPPING', 'PATIENCE', 'MIN_DELTA'],
            'CHECKPOINTING': ['SAVE_DIR', 'SAVE_BEST_ONLY', 'SAVE_EVERY_N_EPOCHS'],
            'LOGGING': ['LOG_DIR', 'LOG_EVERY_N_BATCHES', 'SEED', 'VALIDATE_EVERY_N_EPOCHS'],
        }
        for section, keys in sections.items():
            print(f"\n[{section}]")
            for key in keys:
                if hasattr(self, key):
                    print(f"  {key:<30} = {getattr(self, key)}")
        print("\n" + "=" * 70 + "\n")


CFG = Config()


def get_config(**kwargs) -> Config:
    """
    Get configuration with custom overrides.

    Example:
        cfg = get_config(NUM_EPOCHS=200, BATCH_SIZE=64)
    """
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config
