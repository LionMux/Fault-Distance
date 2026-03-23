"""Configuration settings for training and inference.
"""
import os
import yaml
import torch
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Training configuration."""
    
    # ============ DEVICE ============
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ============ DATA PATHS ============
    # Directory containing oscillogram CSV files (one file per fault event).
    # Kept separate from data/ Python package to avoid mixing .csv with .py files.
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data_training')
    TRAIN_SPLIT: float = 0.8  # 80% train, 20% test
    
    # ============ DATA FORMAT ============
    # Each CSV file = one oscillogram sample.
    # Columns: distance_km, CT1IA, CT1IB, CT1IC, S1)BUS1UA, S1)BUS1UB, S1)BUS1UC
    # Rows: time steps of the signal window
    NUM_CHANNELS: int = 6  # Number of signal channels (Ia, Ib, Ic, Ua, Ub, Uc)
    SEQ_LENGTH: int = 400  # Expected number of time steps per file
    NORMALIZE_DATA: bool = True  # Per-channel StandardScaler normalization
    
    
    # ============ NORMALIZATION MODE ============
    # 'standard' = statistical normalization (StandardScaler + MinMaxScaler)
    # 'pu' = physical per-unit normalization using line parameters
    NORMALIZATION_MODE: str = 'standard'
    
    
    # ============ LINE PARAMETERS (for p.u. normalization) ============
    # Only used when NORMALIZATION_MODE = 'pu'
    LINE_UNOM_KV: float = 110.0      # Nominal voltage [kV]
    LINE_L_KM: float = 50.0          # Line length [km]
    LINE_R1_OHM_KM: float = 0.20046  # Positive sequence resistance [Ohm/km]
    LINE_X1_OHM_KM: float = 0.4155   # Positive sequence reactance [Ohm/km]
    
    
    # ============ SIGNAL PREPROCESSING (NEW) ============
    # Butterworth filter for DC component (aperiodic) removal
    BUTTERWORTH_ENABLED: bool = False
    BUTTERWORTH_CUTOFF: float = 10.0   # Cutoff frequency in Hz (typically 5-15 Hz)
    BUTTERWORTH_FS: float = 1000.0     # Sampling frequency in Hz
    BUTTERWORTH_ORDER: int = 2         # Filter order
    BUTTERWORTH_TYPE: str = 'highpass'
    
    
    # ============ MODEL ARCHITECTURE ============
    MODEL_TYPE: str = 'cnn1d'  # 'cnn1d', 'dilated_cnn1d', 'resnet1d'
    NUM_FILTERS: int = 64      # Initial number of filters
    KERNEL_SIZE: int = 5       # Conv kernel size
    DROPOUT: float = 0.3       # Dropout rate
    
    
    # ============ TRAINING HYPERPARAMETERS ============
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5  # L2 regularization
    
    # Learning rate scheduler
    SCHEDULER_TYPE: Optional[str] = 'cosine' # 'cosine', 'linear', 'exponential', None
    WARMUP_EPOCHS: int = 5
    
    
    # ============ OPTIMIZATION ============
    OPTIMIZER: str = 'adam'       # 'adam', 'sgd', 'adamw'
    LOSS_FUNCTION: str = 'smooth_l1'  # 'mse', 'mae', 'smooth_l1'
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
    
    
    # ============ EXPERIMENT (YAML-mode) ============
    EXPERIMENT_NAME: str = 'unnamed'

    def __post_init__(self):
        os.makedirs(self.SAVE_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        self.print_config()

    def print_config(self):
        print("" + "=" * 70)
        print("CONFIGURATION")
        print("=" * 70)
        
        sections = {
            'DEVICE': ['DEVICE'],
            'DATA': ['DATA_DIR', 'TRAIN_SPLIT', 'NUM_CHANNELS', 'SEQ_LENGTH', 'NORMALIZE_DATA', 'NORMALIZATION_MODE'],
            'PREPROCESSING': ['BUTTERWORTH_ENABLED', 'BUTTERWORTH_CUTOFF', 'BUTTERWORTH_FS', 'BUTTERWORTH_ORDER'],
            'LINE_PARAMS': ['LINE_UNOM_KV', 'LINE_L_KM', 'LINE_R1_OHM_KM', 'LINE_X1_OHM_KM'],
            'MODEL': ['MODEL_TYPE', 'NUM_FILTERS', 'KERNEL_SIZE', 'DROPOUT'],
            'TRAINING': ['BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'SCHEDULER_TYPE', 'WARMUP_EPOCHS'],
            'OPTIMIZATION': ['OPTIMIZER', 'LOSS_FUNCTION', 'GRADIENT_CLIP'],
            'EARLY_STOPPING':['EARLY_STOPPING', 'PATIENCE', 'MIN_DELTA'],
            'CHECKPOINTING': ['SAVE_DIR', 'SAVE_BEST_ONLY', 'SAVE_EVERY_N_EPOCHS'],
            'LOGGING': ['LOG_DIR', 'LOG_EVERY_N_BATCHES', 'SEED', 'VALIDATE_EVERY_N_EPOCHS'],
            'EXPERIMENT': ['EXPERIMENT_NAME'],
        }
        
        for section, keys in sections.items():
            print(f"[{section}]")
            for key in keys:
                if hasattr(self, key):
                    print(f"  {key:<30} = {getattr(self, key)}")
        print("" + "=" * 70 + "")


CFG = Config()

def get_config(**kwargs) -> 'Config':
    """
    Get configuration with custom overrides.
    Example: cfg = get_config(NUM_EPOCHS=200, BATCH_SIZE=64)
    """
    config = Config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return config

# -----------------------------------------------------------------------
# YAML support
# -----------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges override on top of base."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def _autocast(value: str):
    """Cast string to int / float / bool if possible."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() == 'true': return True
    if value.lower() == 'false': return False
    return value

def load_config(yaml_path: str, overrides: dict = None) -> 'Config':
    """
    Load Config from a YAML file.
    
    Priority (lowest to highest):
    1. Config dataclass defaults
    2. configs/base.yaml (if it exists next to yaml_path)
    3. The specified yaml_path
    4. overrides dict (from --set key=value)
    
    Args:
        yaml_path : path to the experiment YAML file
        overrides : {'training.learning_rate': '0.0003', ...}
        
    Example:
        cfg = load_config('configs/augment_train_cnn1d.yaml')
        cfg = load_config('configs/base.yaml', overrides={'training.num_epochs': '200'})
    """
    # 1. base.yaml if present next to yaml_path
    base_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), 'base.yaml')
    raw: dict = {}
    if os.path.exists(base_path) and os.path.abspath(yaml_path) != os.path.abspath(base_path):
        with open(base_path, encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
            
    # 2. experiment yaml on top of base
    with open(yaml_path, encoding='utf-8') as f:
        experiment = yaml.safe_load(f) or {}
        raw = _deep_merge(raw, experiment)
        
    # 3. CLI overrides like 'training.learning_rate=0.0003'
    if overrides:
        for dotted_key, value in overrides.items():
            keys = dotted_key.split('.')
            d = raw
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = _autocast(value) if isinstance(value, str) else value

    # 4. Map YAML sections -> Config UPPERCASE fields
    data_sec = raw.get('data', {})
    model_sec = raw.get('model', {})
    train_sec = raw.get('training', {})
    sched_sec = raw.get('scheduler', {})
    es_sec = raw.get('early_stopping', {})
    ck_sec = raw.get('checkpointing', {})
    log_sec = raw.get('logging', {})
    line_sec = raw.get('line_params', {})
    prep_sec = raw.get('preprocessing', {}) # NEW

    device_raw = raw.get('device', 'auto')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') \
             if device_raw == 'auto' else device_raw

    cfg = Config(
        EXPERIMENT_NAME = raw.get('experiment_name', 'unnamed'),
        DEVICE          = device,
        SEED            = raw.get('seed', 42),
        
        # data
        DATA_DIR        = data_sec.get('data_dir', os.path.join(
                            os.path.dirname(os.path.abspath(__file__)), 'data', 'data_training')),
        TRAIN_SPLIT     = data_sec.get('train_split', 0.8),
        NUM_CHANNELS    = data_sec.get('num_channels', 6),
        SEQ_LENGTH      = data_sec.get('seq_length', 400),
        NORMALIZE_DATA  = data_sec.get('normalize', True),
        NORMALIZATION_MODE = data_sec.get('normalization_mode', 'standard'),
        
        # line parameters
        LINE_UNOM_KV    = line_sec.get('Unom_kv', 110.0),
        LINE_L_KM       = line_sec.get('L_km', 50.0),
        LINE_R1_OHM_KM  = line_sec.get('r1_ohm_km', 0.20046),
        LINE_X1_OHM_KM  = line_sec.get('x1_ohm_km', 0.4155),
        
        # preprocessing (NEW)
        BUTTERWORTH_ENABLED = prep_sec.get('butterworth_enabled', False),
        BUTTERWORTH_CUTOFF  = prep_sec.get('butterworth_cutoff', 10.0),
        BUTTERWORTH_FS      = prep_sec.get('butterworth_fs', 1000.0),
        BUTTERWORTH_ORDER   = prep_sec.get('butterworth_order', 2),
        BUTTERWORTH_TYPE    = prep_sec.get('butterworth_type', 'highpass'),

        # model
        MODEL_TYPE      = model_sec.get('type', 'cnn1d'),
        NUM_FILTERS     = model_sec.get('num_filters', 64),
        KERNEL_SIZE     = model_sec.get('kernel_size', 5),
        DROPOUT         = model_sec.get('dropout', 0.3),
        
        # training
        BATCH_SIZE      = train_sec.get('batch_size', 32),
        NUM_EPOCHS      = train_sec.get('num_epochs', 100),
        OPTIMIZER       = train_sec.get('optimizer', 'adam'),
        LEARNING_RATE   = train_sec.get('learning_rate', 0.001),
        WEIGHT_DECAY    = train_sec.get('weight_decay', 1e-5),
        LOSS_FUNCTION   = train_sec.get('loss', 'smooth_l1'),
        GRADIENT_CLIP   = train_sec.get('gradient_clip', 1.0),
        
        # scheduler
        SCHEDULER_TYPE  = train_sec.get('scheduler_type', # support old key
                          sched_sec.get('type', 'cosine')),
        WARMUP_EPOCHS   = sched_sec.get('warmup_epochs', 5),
        
        # early stopping
        EARLY_STOPPING  = es_sec.get('enabled', True),
        PATIENCE        = es_sec.get('patience', 15),
        MIN_DELTA       = es_sec.get('min_delta', 1e-4),
        
        # checkpointing
        SAVE_DIR        = ck_sec.get('save_dir', 'checkpoints'),
        SAVE_BEST_ONLY  = ck_sec.get('save_best_only', True),
        SAVE_EVERY_N_EPOCHS = ck_sec.get('save_every_n_epochs', 10),
        
        # logging
        LOG_DIR         = log_sec.get('log_dir', 'logs'),
        LOG_EVERY_N_BATCHES = log_sec.get('log_every_n_batches', 50),
    )
    
    # Attach augmentation section for pipeline scripts
    cfg._augmentation_cfg = raw.get('augmentation', {})
    
    return cfg
