"""
Preprocessing package.

This package is a refactor of the former monolithic module `data/preprocessing.py`.
It keeps the same public API by re-exporting the functions/classes used across
the project.
"""

from .dc_filters import center_by_prehistory, remove_dc_period, analyze_signals_full
from .symseq import sliding_window_symseq
from .classes import DataPreprocessor, DataAugmentation
from .inference import apply_pu_normalization, preprocess_signal_for_inference

__all__ = [
    "center_by_prehistory",
    "remove_dc_period",
    "analyze_signals_full",
    "sliding_window_symseq",
    "DataPreprocessor",
    "DataAugmentation",
    "apply_pu_normalization",
    "preprocess_signal_for_inference",
]
