"""
Configuration settings for training and inference.
"""

import os
import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Training configuration."""
    
    # ============ DEVICE ============
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ============ DATA PATHS ============
    DATA_DIR: str = os.path.join(os.path.dirname(__file__), 'data')
    CSV_FILE: str = os.path.join(DATA_DIR, 'oscillograms.csv')  # Place your CSV here
    TRAIN_SPLIT: float = 0.8  # 80% train, 20% test
    
    # ============ DATA PREPROCESSING ============
    SEQ_LENGTH: int = 400  # Signal length (columns in CSV)
    NORMALIZE_DATA: bool = True
    
    # ============ MODEL ARCHITECTURE ============
    MODEL_TYPE: str = 'cnn1d'  # 'cnn1d', 'dilated_cnn1d', 'resnet1d'
    NUM_FILTERS: int = 64  # Initial number of filters
    KERNEL_SIZE: int = 5  # Conv kernel size
    DROPOUT: float = 0.3  # Dropout rate
    
    # ============ TRAINING HYPERPARAMETERS ============
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5  # L2 regularization
    
    # Learning rate scheduler
    SCHEDULER_TYPE: Optional[str] = 'cosine'  # 'cosine', 'linear', 'exponential', None
    WARMUP_EPOCHS: int = 5
    
    # ============ OPTIMIZATION ============
    OPTIMIZER: str = 'adam'  # 'adam', 'sgd', 'adamw'
    LOSS_FUNCTION: str = 'mse'  # 'mse', 'mae', 'smooth_l1'
    GRADIENT_CLIP: Optional[float] = 1.0  # None to disable
    
    # ============ EARLY STOPPING ============
    EARLY_STOPPING: bool = True
    PATIENCE: int = 15  # Stop if val_loss doesn't improve for N epochs
    MIN_DELTA: float = 1e-4  # Minimum improvement threshold
    
    # ============ CHECKPOINTING ============
    SAVE_DIR: str = 'checkpoints'
    SAVE_BEST_ONLY: bool = True  # Save only best model
    SAVE_EVERY_N_EPOCHS: int = 10  # Also save every N epochs
    
    # ============ LOGGING ============
    LOG_DIR: str = 'logs'
    LOG_EVERY_N_BATCHES: int = 50  # Log training metrics every N batches
    SEED: int = 42  # Random seed for reproducibility
    
    # ============ VALIDATION ============
    VALIDATE_EVERY_N_EPOCHS: int = 1  # Run validation every N epochs
    
    def __post_init__(self):
        """Create necessary directories."""
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        # Print configuration
        self.print_config()
    
    def print_config(self):
        """Print configuration in formatted way."""
        print("\n" + "="*70)
        print("CONFIGURATION")
        print("="*70)
        
        sections = {
            'DEVICE': ['DEVICE'],
            'DATA': ['DATA_DIR', 'CSV_FILE', 'TRAIN_SPLIT', 'SEQ_LENGTH', 'NORMALIZE_DATA'],
            'MODEL': ['MODEL_TYPE', 'NUM_FILTERS', 'KERNEL_SIZE', 'DROPOUT'],
            'TRAINING': ['BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 
                        'SCHEDULER_TYPE', 'WARMUP_EPOCHS'],
            'OPTIMIZATION': ['OPTIMIZER', 'LOSS_FUNCTION', 'GRADIENT_CLIP'],
            'EARLY_STOPPING': ['EARLY_STOPPING', 'PATIENCE', 'MIN_DELTA'],
            'CHECKPOINTING': ['SAVE_DIR', 'SAVE_BEST_ONLY', 'SAVE_EVERY_N_EPOCHS'],
            'LOGGING': ['LOG_DIR', 'LOG_EVERY_N_BATCHES', 'SEED', 'VALIDATE_EVERY_N_EPOCHS']
        }
        
        for section, keys in sections.items():
            print(f"\n[{section}]")
            for key in keys:
                if hasattr(self, key):
                    value = getattr(self, key)
                    print(f"  {key:<30} = {value}")
        
        print("\n" + "="*70 + "\n")


# Default configuration
CFG = Config()


def get_config(**kwargs) -> Config:
    """
    Get configuration with custom overrides.
    
    Example:
        cfg = get_config(NUM_EPOCHS=200, BATCH_SIZE=64)
    
    Args:
        **kwargs: Configuration fields to override
    
    Returns:
        Config object
    """
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config
