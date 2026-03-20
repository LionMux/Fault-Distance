"""
Data preprocessing and augmentation utilities.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d


def apply_butterworth_filter(signals: np.ndarray, cfg) -> np.ndarray:
    """
    Apply Butterworth filter to all channels of all samples.

    Args:
        signals: numpy array of shape (N, C, T)
        cfg: Config object with BUTTERWORTH_* fields

    Returns:
        Filtered signals, same shape (N, C, T)
    """
    from scipy.signal import butter, filtfilt

    cutoff = cfg.BUTTERWORTH_CUTOFF
    fs = cfg.BUTTERWORTH_FS
    order = cfg.BUTTERWORTH_ORDER
    btype = cfg.BUTTERWORTH_TYPE  # 'highpass', 'lowpass', 'bandpass'

    nyquist = fs / 2.0
    normalized_cutoff = cutoff / nyquist
    normalized_cutoff = np.clip(normalized_cutoff, 1e-6, 1.0 - 1e-6)

    b, a = butter(order, normalized_cutoff, btype=btype)

    filtered = signals.copy()
    N, C, T = signals.shape
    for i in range(N):
        for ch in range(C):
            filtered[i, ch, :] = filtfilt(b, a, signals[i, ch, :])

    return filtered


class DataPreprocessor:
    """
    Advanced preprocessing for oscillogram signals.
    """

    @staticmethod
    def apply_bandpass_filter(signal_data, lowcut=10, highcut=5000, fs=50000):
        """
        Apply Butterworth bandpass filter to remove noise.
        """
        nyquist = fs / 2
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        return scipy_signal.filtfilt(b, a, signal_data)

    @staticmethod
    def apply_smoothing(signal_data, sigma=1.0):
        """
        Apply Gaussian smoothing to reduce high-frequency noise.
        """
        return gaussian_filter1d(signal_data, sigma=sigma)

    @staticmethod
    def normalize_signal(signal_data, method='standard'):
        """
        Normalize signal using different methods.
        """
        if method == 'standard':
            return (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-8)
        elif method == 'minmax':
            return (signal_data - np.min(signal_data)) / (np.max(signal_data) - np.min(signal_data) + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    @staticmethod
    def compute_statistics(signal_data):
        """
        Compute statistical features from signal.
        """
        return {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'max': np.max(signal_data),
            'min': np.min(signal_data),
            'rms': np.sqrt(np.mean(signal_data ** 2)),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'kurtosis': scipy_signal.kurtosis(signal_data),
            'skewness': scipy_signal.skew(signal_data)
        }


class DataAugmentation:
    """
    Data augmentation techniques for signal data.
    """

    @staticmethod
    def add_gaussian_noise(signal_data, noise_std=0.01):
        noise = np.random.normal(0, noise_std, signal_data.shape)
        return signal_data + noise

    @staticmethod
    def time_shift(signal_data, max_shift=10):
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal_data, shift)

    @staticmethod
    def amplitude_scaling(signal_data, scale_range=(0.8, 1.2)):
        scale = np.random.uniform(*scale_range)
        return signal_data * scale

    @staticmethod
    def mixup(signal1, signal2, alpha=0.2):
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2
