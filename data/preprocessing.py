"""
Data preprocessing and augmentation utilities.
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d


class DataPreprocessor:
    """
    Advanced preprocessing for oscillogram signals.
    """
    
    @staticmethod
    def apply_bandpass_filter(signal_data, lowcut=10, highcut=5000, fs=50000):
        """
        Apply Butterworth bandpass filter to remove noise.
        
        Args:
            signal_data: Input signal
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
            fs: Sampling frequency (Hz)
        
        Returns:
            Filtered signal
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
        
        Args:
            signal_data: Input signal
            sigma: Standard deviation of Gaussian kernel
        
        Returns:
            Smoothed signal
        """
        return gaussian_filter1d(signal_data, sigma=sigma)
    
    @staticmethod
    def normalize_signal(signal_data, method='standard'):
        """
        Normalize signal using different methods.
        
        Args:
            signal_data: Input signal
            method: 'standard' (z-score) or 'minmax' ([0,1])
        
        Returns:
            Normalized signal
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
        
        Args:
            signal_data: Input signal (1D array)
        
        Returns:
            dict: Statistics
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
        """
        Add Gaussian white noise to signal.
        
        Args:
            signal_data: Input signal
            noise_std: Standard deviation of noise
        
        Returns:
            Signal with added noise
        """
        noise = np.random.normal(0, noise_std, signal_data.shape)
        return signal_data + noise
    
    @staticmethod
    def time_shift(signal_data, max_shift=10):
        """
        Shift signal in time (circular shift).
        
        Args:
            signal_data: Input signal
            max_shift: Maximum shift in samples
        
        Returns:
            Time-shifted signal
        """
        shift = np.random.randint(-max_shift, max_shift)
        return np.roll(signal_data, shift)
    
    @staticmethod
    def amplitude_scaling(signal_data, scale_range=(0.8, 1.2)):
        """
        Scale amplitude of signal.
        
        Args:
            signal_data: Input signal
            scale_range: (min_scale, max_scale)
        
        Returns:
            Amplitude-scaled signal
        """
        scale = np.random.uniform(*scale_range)
        return signal_data * scale
    
    @staticmethod
    def mixup(signal1, signal2, alpha=0.2):
        """
        Mixup augmentation: blend two signals.
        
        Args:
            signal1: First signal
            signal2: Second signal
            alpha: Mixing parameter (0 = all signal1, 1 = all signal2)
        
        Returns:
            Mixed signal
        """
        lam = np.random.beta(alpha, alpha)
        return lam * signal1 + (1 - lam) * signal2
