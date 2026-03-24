"""
Data preprocessing and augmentation utilities.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d


def apply_butterworth_filter(signals: np.ndarray, cfg) -> np.ndarray:
    """Apply Butterworth high-pass filter to remove the DC / aperiodic component.

    This is the function called by dataset.py when cfg.BUTTERWORTH_ENABLED is True.
    The sampling frequency is taken from cfg.BUTTERWORTH_FS which, after the
    recent refactor of config.py / base.yaml, automatically mirrors
    cfg.SAMPLING_FREQ_HZ when not set explicitly — so there is a single source
    of truth for fs even in the Butterworth path.

    Args:
        signals: Array of shape (N, C, T) — batch of multi-channel oscillograms.
        cfg:     Config object with BUTTERWORTH_* fields.

    Returns:
        Filtered signals with the same shape.
    """
    from scipy.signal import butter, filtfilt

    fs      = float(cfg.BUTTERWORTH_FS)
    cutoff  = float(cfg.BUTTERWORTH_CUTOFF)
    order   = int(cfg.BUTTERWORTH_ORDER)
    btype   = str(cfg.BUTTERWORTH_TYPE)

    nyquist = fs / 2.0
    norm_cut = cutoff / nyquist
    norm_cut = float(np.clip(norm_cut, 1e-6, 1.0 - 1e-6))

    b, a = butter(order, norm_cut, btype=btype)

    out = signals.copy()
    N, C, T = out.shape
    for n in range(N):
        for c in range(C):
            out[n, c, :] = filtfilt(b, a, out[n, c, :])
    return out


class DataPreprocessor:
    """
    Advanced preprocessing for oscillogram signals.
    All methods that require a sampling frequency accept it explicitly as an
    argument — there are no hardcoded defaults.
    """

    @staticmethod
    def apply_bandpass_filter(
        signal_data: np.ndarray,
        lowcut: float,
        highcut: float,
        fs: float,
    ) -> np.ndarray:
        """Apply Butterworth bandpass filter.

        Args:
            signal_data: 1-D or N-D input signal (filtered along last axis).
            lowcut:  Low-frequency cutoff [Hz].
            highcut: High-frequency cutoff [Hz].
            fs:      Sampling frequency [Hz] — must match the data.
        """
        nyquist = fs / 2.0
        low  = float(np.clip(lowcut  / nyquist, 1e-6, 1.0 - 1e-6))
        high = float(np.clip(highcut / nyquist, 1e-6, 1.0 - 1e-6))
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, signal_data)

    @staticmethod
    def apply_smoothing(signal_data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to reduce high-frequency noise."""
        return gaussian_filter1d(signal_data, sigma=sigma)

    @staticmethod
    def normalize_signal(
        signal_data: np.ndarray, method: str = 'standard'
    ) -> np.ndarray:
        """Normalize signal.

        Args:
            method: 'standard' (z-score) or 'minmax' ([0, 1]).
        """
        if method == 'standard':
            return (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        elif method == 'minmax':
            mn, mx = np.min(signal_data), np.max(signal_data)
            return (signal_data - mn) / (mx - mn + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def compute_statistics(signal_data: np.ndarray) -> dict:
        """Compute basic statistical features from a 1-D signal."""
        return {
            'mean':          float(np.mean(signal_data)),
            'std':           float(np.std(signal_data)),
            'max':           float(np.max(signal_data)),
            'min':           float(np.min(signal_data)),
            'rms':           float(np.sqrt(np.mean(signal_data ** 2))),
            'peak_to_peak':  float(np.max(signal_data) - np.min(signal_data)),
            'kurtosis':      float(scipy_signal.kurtosis(signal_data)),
            'skewness':      float(scipy_signal.skew(signal_data)),
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
