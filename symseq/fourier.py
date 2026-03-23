"""
Phasor estimation from time-domain waveforms using FFT.

Assumes single-frequency (50 or 60 Hz) signals sampled at fs Hz.
Uses numpy.fft — battle-tested, part of numpy core.
"""

import numpy as np


def estimate_phasor(
    signal: np.ndarray,
    fs: float,
    f0: float = 50.0,
    window: bool = True,
) -> complex:
    """
    Estimate the fundamental-frequency phasor from a 1-D time-domain signal.

    The phasor is defined as the complex amplitude at frequency f0:
        phasor = (2 / N) * X[k]
    where X = FFT(signal * w), k = round(f0 * N / fs), w is the window.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Real-valued time-domain waveform.
    fs : float
        Sampling frequency in Hz.
    f0 : float
        Fundamental frequency in Hz (default 50).
    window : bool
        If True, apply a Hann window before FFT to reduce spectral leakage.
        Set False only when N is an exact integer multiple of T0 = fs/f0.

    Returns
    -------
    complex
        Phasor: abs() = peak amplitude, angle() = phase in radians.

    Examples
    --------
    >>> import numpy as np
    >>> fs, f0, N = 1000, 50, 400   # exactly 20 periods
    >>> t = np.arange(N) / fs
    >>> A, phi = 100.0, np.pi / 4
    >>> sig = A * np.sin(2 * np.pi * f0 * t + phi)
    >>> p = estimate_phasor(sig, fs, f0, window=False)
    >>> assert abs(abs(p) - A) < 0.01
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    if window:
        w = np.hanning(N)
        # Компенсируем энергетическое затухание окна
        scale = N / w.sum()
        signal = signal * w * scale

    X = np.fft.rfft(signal)
    k = int(round(f0 * N / fs))
    # Масштабируем: делим на N/2, чтобы получить пиковую амплитуду
    phasor = 2.0 * X[k] / N
    return complex(phasor)


def estimate_phasors_batch(
    signals: np.ndarray,
    fs: float,
    f0: float = 50.0,
    window: bool = True,
) -> np.ndarray:
    """
    Vectorised version of estimate_phasor for a batch of signals.

    Parameters
    ----------
    signals : np.ndarray, shape (batch, N)
        Each row is one time-domain waveform.
    fs : float
        Sampling frequency in Hz.
    f0 : float
        Fundamental frequency in Hz.
    window : bool
        Apply Hann window (default True).

    Returns
    -------
    np.ndarray, shape (batch,), dtype complex128
        Array of phasors, one per signal.
    """
    signals = np.asarray(signals, dtype=float)
    N = signals.shape[-1]

    if window:
        w = np.hanning(N)
        scale = N / w.sum()
        signals = signals * w * scale

    X = np.fft.rfft(signals, axis=-1)
    k = int(round(f0 * N / fs))
    phasors = 2.0 * X[..., k] / N
    return phasors
