"""Configuration settings for training and inference.
"""
import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class Config:
    """Training configuration."""

    # ============ DEVICE ============
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ============ OUTPUT / ARTIFACTS ============
    # Diploma-level plots copied into output/thesis/
    # (в compare_models.py обычно выключаем, чтобы не раздувать output/thesis)
    SAVE_THESIS_PLOTS: bool = True

    # ============ DATA PATHS ============
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'data_training')
    TRAIN_SPLIT: float = 0.8  # 80% train, 20% test

    # ============ DATA FORMAT ============
    NUM_CHANNELS: int = 12   # Number of signal channels (6 phase + 6 phasor magnitudes)
    SEQ_LENGTH: int = 240    # Expected number of time steps per file (fs=2000 Hz, [50ms pre + 70ms post] = 240 samples)
    NORMALIZE_DATA: bool = True

    # ============ NORMALIZATION MODE ============
    # 'standard' = statistical normalization (StandardScaler + MinMaxScaler)
    # 'pu'       = physical per-unit normalization using line parameters
    NORMALIZATION_MODE: str = 'standard'

    # ============ LINE PARAMETERS (for p.u. normalization) ============
    LINE_UNOM_KV: float = 110.0
    LINE_L_KM: float = 50.0
    LINE_R1_OHM_KM: float = 0.20046
    LINE_X1_OHM_KM: float = 0.4155

    # ============ BASE POWER for p.u. normalization ============
    S_BASE_MVA: float = 100.0

    # ============ SIGNAL PREPROCESSING ============
    # --- DC-period removal (notebook stage 2) ---
    REMOVE_DC_ENABLED: bool = True      # Enable rolling-mean DC removal (notebook method)

    # --- Symmetrical components (phasor-based, notebook stage 5 adapted) ---
    SYMSEQ_ENABLED: bool = True         # Append |I1|,|I2|,|I0|,|U1|,|U2|,|U0|

    # --- Fault-inception (t0) detection ---
    # SAMPLING_FREQ_HZ is a *fallback* value used only for CSV files that
    # were converted without the 'fs_hz' column (legacy files).
    # New files produced by comtrade_to_csv.py carry their own fs_hz column
    # and it is used automatically — no manual configuration needed.
    SAMPLING_FREQ_HZ: float = 5000.0   # fallback [Hz] for legacy CSV files (notebook Fs)
    MAINS_FREQ_HZ: float = 50.0        # Power-system frequency [Hz]
    T0_ENABLED: bool = True             # Enable fault-inception cropping (notebook)
    T0_PRE_MS: float = 50.0             # notebook: 50 ms pre-fault
    T0_POST_MS: float = 70.0            # notebook: 70 ms post-fault
    T0_ETA_I: float = 0.5               # notebook: current-rise threshold (50%)
    T0_ETA_U: float = 0.85              # notebook: voltage-drop threshold (85%)

    # ============ MODEL ARCHITECTURE ============
    MODEL_TYPE: str = 'cnn1d'  # 'cnn1d', 'dilated_cnn1d', 'resnet1d'
    NUM_FILTERS: int = 64
    KERNEL_SIZE: int = 5
    DROPOUT: float = 0.3

    # ============ TRAINING HYPERPARAMETERS ============
    BATCH_SIZE: int = 32
    NUM_EPOCHS: int = 100
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 1e-5

    SCHEDULER_TYPE: Optional[str] = 'cosine'  # 'cosine', 'linear', 'exponential', None
    WARMUP_EPOCHS: int = 5

    # ============ OPTIMIZATION ============
    OPTIMIZER: str = 'adam'
    LOSS_FUNCTION: str = 'smooth_l1'
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

    # ============ DIPLOMA ACTIVATION EXPORT ============
    # Additive, disabled by default.
    ACTIVATION_EXPORT_ENABLED: bool = False
    # "auto", "auto_all_conv_blocks", or explicit list of module names.
    ACTIVATION_EXPORT_LAYERS: Any = "auto"
    # e.g. ["first", "best", "every_n:5"].
    ACTIVATION_EXPORT_CAPTURE_EPOCHS: List[str] = field(default_factory=lambda: ["first", "best", "every_n:5"])
    ACTIVATION_EXPORT_PROBE_MODE: str = "first_val_sample"
    ACTIVATION_EXPORT_EXPORT_FORMATS: List[str] = field(default_factory=lambda: ["png", "csv"])
    # Saved under run_dir/ACTIVATION_EXPORT_OUTPUT_DIR
    ACTIVATION_EXPORT_OUTPUT_DIR: str = "activations"
    ACTIVATION_EXPORT_MAX_CHANNELS_TO_PLOT: int = 8
    ACTIVATION_EXPORT_DPI: int = 300
    STRICT_ACTIVATION_EXPORT: bool = False

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
            'PREPROCESSING': [
                'SAMPLING_FREQ_HZ',
                'MAINS_FREQ_HZ',
                'T0_ENABLED',
                'T0_PRE_MS',
                'T0_POST_MS',
                'T0_ETA_I',
                'T0_ETA_U',
            ],
            'LINE_PARAMS': ['LINE_UNOM_KV', 'LINE_L_KM', 'LINE_R1_OHM_KM', 'LINE_X1_OHM_KM'],
            'MODEL': ['MODEL_TYPE', 'NUM_FILTERS', 'KERNEL_SIZE', 'DROPOUT'],
            'TRAINING': ['BATCH_SIZE', 'NUM_EPOCHS', 'LEARNING_RATE', 'WEIGHT_DECAY', 'SCHEDULER_TYPE', 'WARMUP_EPOCHS'],
            'OPTIMIZATION': ['OPTIMIZER', 'LOSS_FUNCTION', 'GRADIENT_CLIP'],
            'EARLY_STOPPING': ['EARLY_STOPPING', 'PATIENCE', 'MIN_DELTA'],
            'CHECKPOINTING': ['SAVE_DIR', 'SAVE_BEST_ONLY', 'SAVE_EVERY_N_EPOCHS'],
            'LOGGING': ['LOG_DIR', 'LOG_EVERY_N_BATCHES', 'SEED', 'VALIDATE_EVERY_N_EPOCHS'],
            'EXPERIMENT': ['EXPERIMENT_NAME'],
            'ACTIVATION_EXPORT': [
                'ACTIVATION_EXPORT_ENABLED',
                'ACTIVATION_EXPORT_LAYERS',
                'ACTIVATION_EXPORT_CAPTURE_EPOCHS',
                'ACTIVATION_EXPORT_PROBE_MODE',
                'ACTIVATION_EXPORT_EXPORT_FORMATS',
                'ACTIVATION_EXPORT_OUTPUT_DIR',
                'ACTIVATION_EXPORT_MAX_CHANNELS_TO_PLOT',
                'ACTIVATION_EXPORT_DPI',
                'STRICT_ACTIVATION_EXPORT',
            ],
        }

        for section, keys in sections.items():
            print(f"[{section}]")
            for key in keys:
                if hasattr(self, key):
                    print(f"  {key:<30} = {getattr(self, key)}")
        print("" + "=" * 70 + "")


CFG = Config()


def get_config(**kwargs) -> 'Config':
    """Get configuration with custom overrides.

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
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    return value


def load_config(yaml_path: str, overrides: dict = None) -> 'Config':
    """Load Config from a YAML file.

    Priority (lowest to highest):
        1. Config dataclass defaults
        2. configs/base.yaml (if it exists next to yaml_path)
        3. The specified yaml_path
        4. overrides dict (from --set key=value)

    Args:
        yaml_path : Path to the experiment YAML file.
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
    data_sec  = raw.get('data', {})
    model_sec = raw.get('model', {})
    train_sec = raw.get('training', {})
    sched_sec = raw.get('scheduler', {})
    es_sec    = raw.get('early_stopping', {})
    ck_sec    = raw.get('checkpointing', {})
    log_sec   = raw.get('logging', {})
    line_sec  = raw.get('line_params', {})
    prep_sec  = raw.get('preprocessing', {})

    device_raw = raw.get('device', 'auto')
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if device_raw == 'auto' else device_raw

    _sampling_freq = prep_sec.get('sampling_freq_hz', 2000.0)

    cfg = Config(
        EXPERIMENT_NAME=raw.get('experiment_name', 'unnamed'),
        DEVICE=device,
        SEED=raw.get('seed', 42),

        # data
        DATA_DIR=data_sec.get('data_dir', os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data', 'data_training')
        ),
        TRAIN_SPLIT=data_sec.get('train_split', 0.8),
        NUM_CHANNELS=data_sec.get('num_channels', 12),
        SEQ_LENGTH=data_sec.get('seq_length', 400),
        NORMALIZE_DATA=data_sec.get('normalize', True),
        NORMALIZATION_MODE=data_sec.get('normalization_mode', 'standard'),

        # line parameters
        LINE_UNOM_KV=line_sec.get('Unom_kv', 110.0),
        LINE_L_KM=line_sec.get('L_km', 50.0),
        LINE_R1_OHM_KM=line_sec.get('r1_ohm_km', 0.20046),
        LINE_X1_OHM_KM=line_sec.get('x1_ohm_km', 0.4155),

        # preprocessing
        SAMPLING_FREQ_HZ=_sampling_freq,
        MAINS_FREQ_HZ=prep_sec.get('mains_freq_hz', 50.0),
        REMOVE_DC_ENABLED=prep_sec.get('remove_dc_enabled', False),
        SYMSEQ_ENABLED=prep_sec.get('symseq_enabled', True),
        T0_ENABLED=prep_sec.get('t0_enabled', False),
        T0_PRE_MS=prep_sec.get('t0_pre_ms', 50.0),
        T0_POST_MS=prep_sec.get('t0_post_ms', 150.0),
        T0_ETA_I=prep_sec.get('t0_eta_i', 0.5),
        T0_ETA_U=prep_sec.get('t0_eta_u', 0.85),

        # model
        MODEL_TYPE=model_sec.get('type', 'cnn1d'),
        NUM_FILTERS=model_sec.get('num_filters', 64),
        KERNEL_SIZE=model_sec.get('kernel_size', 5),
        DROPOUT=model_sec.get('dropout', 0.3),

        # training
        BATCH_SIZE=train_sec.get('batch_size', 32),
        NUM_EPOCHS=train_sec.get('num_epochs', 100),
        OPTIMIZER=train_sec.get('optimizer', 'adam'),
        LEARNING_RATE=train_sec.get('learning_rate', 0.001),
        WEIGHT_DECAY=train_sec.get('weight_decay', 1e-5),
        LOSS_FUNCTION=train_sec.get('loss', 'smooth_l1'),
        GRADIENT_CLIP=train_sec.get('gradient_clip', 1.0),

        # scheduler
        SCHEDULER_TYPE=train_sec.get('scheduler_type', sched_sec.get('type', 'cosine')),
        WARMUP_EPOCHS=sched_sec.get('warmup_epochs', 5),

        # early stopping
        EARLY_STOPPING=es_sec.get('enabled', True),
        PATIENCE=es_sec.get('patience', 15),
        MIN_DELTA=es_sec.get('min_delta', 1e-4),

        # checkpointing
        SAVE_DIR=ck_sec.get('save_dir', 'checkpoints'),
        SAVE_BEST_ONLY=ck_sec.get('save_best_only', True),
        SAVE_EVERY_N_EPOCHS=ck_sec.get('save_every_n_epochs', 10),

        # logging
        LOG_DIR=log_sec.get('log_dir', 'logs'),
        LOG_EVERY_N_BATCHES=log_sec.get('log_every_n_batches', 50),

        # activation_export (diploma visuals)
        ACTIVATION_EXPORT_ENABLED=raw.get('activation_export', {}).get('enabled', False),
        ACTIVATION_EXPORT_LAYERS=raw.get('activation_export', {}).get('layers', 'auto'),
        ACTIVATION_EXPORT_CAPTURE_EPOCHS=raw.get('activation_export', {}).get(
            'capture_epochs', ["first", "best", "every_n:5"]
        ),
        ACTIVATION_EXPORT_PROBE_MODE=raw.get('activation_export', {}).get('probe_mode', 'first_val_sample'),
        ACTIVATION_EXPORT_EXPORT_FORMATS=raw.get('activation_export', {}).get('export_formats', ['png', 'csv']),
        ACTIVATION_EXPORT_OUTPUT_DIR=raw.get('activation_export', {}).get('output_dir', 'activations'),
        ACTIVATION_EXPORT_MAX_CHANNELS_TO_PLOT=raw.get('activation_export', {}).get('max_channels_to_plot', 8),
        ACTIVATION_EXPORT_DPI=raw.get('activation_export', {}).get('dpi', 300),
        STRICT_ACTIVATION_EXPORT=raw.get('activation_export', {}).get('strict_activation_export', False),
    )

    # --- Post-process types (YAML can sometimes parse scientific notation like 1e-5 as str) ---
    # Fix common numeric fields that must be floats/ints for torch optimizers.
    if isinstance(cfg.WEIGHT_DECAY, str):
        cfg.WEIGHT_DECAY = float(cfg.WEIGHT_DECAY)
    if isinstance(cfg.LEARNING_RATE, str):
        cfg.LEARNING_RATE = float(cfg.LEARNING_RATE)
    if isinstance(cfg.GRADIENT_CLIP, str):
        cfg.GRADIENT_CLIP = float(cfg.GRADIENT_CLIP) if cfg.GRADIENT_CLIP.lower() not in ("none", "null") else None
    if isinstance(cfg.BATCH_SIZE, str):
        cfg.BATCH_SIZE = int(cfg.BATCH_SIZE)
    if isinstance(cfg.NUM_EPOCHS, str):
        cfg.NUM_EPOCHS = int(cfg.NUM_EPOCHS)
    if isinstance(cfg.SEED, str):
        cfg.SEED = int(cfg.SEED)

    # If symseq is enabled the dataset appends 6 extra channels (6 -> 12).
    # Many YAML configs keep data.num_channels=6 as "phase channels count",
    # so we align model input channels with dataset output.
    if bool(getattr(cfg, "SYMSEQ_ENABLED", False)) and int(cfg.NUM_CHANNELS) == 6:
        cfg.NUM_CHANNELS = 12

    # Attach augmentation section for pipeline scripts
    cfg._augmentation_cfg = raw.get('augmentation', {})

    return cfg
