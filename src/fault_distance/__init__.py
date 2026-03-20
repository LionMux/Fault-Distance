"""Fault Distance - 1D-CNN for short-circuit fault location in power systems.

Package structure:
    fault_distance/
    ├── config.py          - Configuration dataclass & YAML loader
    ├── inference.py       - Inference pipeline
    ├── data/
    │   ├── dataset.py     - FaultDataset & DataLoaderFactory
    │   ├── preprocessing.py - Signal preprocessing (Butterworth filter)
    │   └── augmentation.py  - Data augmentation transforms
    ├── models/
    │   ├── cnn1d.py       - CNN1D & DilatedCNN1D architectures
    │   ├── resnet1d.py    - ResNet1D architecture
    │   └── blocks.py      - Shared building blocks
    └── utils/
        ├── logger.py      - Training logger
        ├── metrics.py     - Evaluation metrics
        └── plots.py       - Visualization utilities
"""

from .config import Config, load_config, get_config

__version__ = "0.1.0"
__all__ = ["Config", "load_config", "get_config"]
