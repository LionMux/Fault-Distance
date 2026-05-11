"""
Preprocessing/augmentation helper classes.

This module contains:
- DataPreprocessor
- DataAugmentation

Code was moved out of the former monolithic `data/preprocessing.py`.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d


class DataPreprocessor:
    """
    Advanced preprocessing for oscillogram signals.
    All methods that require a sampling frequency accept it explicitly as an
    argument — there are no hardcoded defaults.
    """

    @staticmethod
    def apply_smoothing(signal_data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to reduce high-frequency noise."""
        return gaussian_filter1d(signal_data, sigma=sigma)

    @staticmethod
    def normalize_signal(
        signal_data: np.ndarray, method: str = "standard"
    ) -> np.ndarray:
        """Normalize signal.

        Args:
            method: 'standard' (z-score) or 'minmax' ([0, 1]).
        """
        if method == "standard":
            return (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        elif method == "minmax":
            mn, mx = np.min(signal_data), np.max(signal_data)
            return (signal_data - mn) / (mx - mn + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def compute_statistics(signal_data: np.ndarray) -> dict:
        """Compute basic statistical features from a 1-D signal."""
        return {
            "mean": float(np.mean(signal_data)),
            "std": float(np.std(signal_data)),
            "max": float(np.max(signal_data)),
            "min": float(np.min(signal_data)),
            "rms": float(np.sqrt(np.mean(signal_data**2))),
            "peak_to_peak": float(np.max(signal_data) - np.min(signal_data)),
            "kurtosis": float(scipy_signal.kurtosis(signal_data)),
            "skewness": float(scipy_signal.skew(signal_data)),
        }


class DataAugmentation:
    """Data augmentation techniques for oscillogram signals."""

    @staticmethod
    def add_gaussian_noise(
        signal_data: np.ndarray, noise_std: float = 0.01
    ) -> np.ndarray:
        """Add Gaussian white noise."""
        return signal_data + np.random.normal(0, noise_std, signal_data.shape)

    @staticmethod
    def time_shift(signal_data: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """Circular time shift by a random amount in [-max_shift, max_shift]."""
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal_data, shift)

    @staticmethod
    def amplitude_scaling(
        signal_data: np.ndarray,
        scale_range: tuple = (0.8, 1.2),
    ) -> np.ndarray:
        """Scale amplitude by a uniform random factor."""
        return signal_data * np.random.uniform(*scale_range)

    @staticmethod
    def mixup(
        signal1: np.ndarray,
        signal2: np.ndarray,
        alpha: float = 0.2,
    ) -> np.ndarray:
        """Mixup augmentation: blend two signals with a Beta-distributed weight."""
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2
