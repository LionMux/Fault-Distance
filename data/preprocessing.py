"""
Backwards-compatible wrapper for the former monolithic module.

The real implementation now lives under the package:
    data/preprocessing/

This file keeps existing imports working, e.g.:
    from data.preprocessing import remove_dc_period, sliding_window_symseq, DataPreprocessor, ...
"""

from __future__ import annotations

from data.preprocessing.dc_filters import (  # noqa: F401
    center_by_prehistory,
    remove_dc_period,
    analyze_signals_full,
)
from data.preprocessing.symseq import sliding_window_symseq  # noqa: F401
from data.preprocessing.classes import DataPreprocessor, DataAugmentation  # noqa: F401
from data.preprocessing.inference import (  # noqa: F401
    apply_pu_normalization,
    preprocess_signal_for_inference,
)

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
