"""
Phasor estimation from time-domain waveforms using FFT.

Assumes single-frequency (50 or 60 Hz) signals sampled at fs Hz.
Uses numpy.fft — battle-tested, part of numpy core.

Normalisation:
    phasor = 2 * X[k] / norm
    where norm = sum(window)  if windowed
                 N             if rectangular

The factor 2 converts one-sided spectrum to peak amplitude.
Dividing by sum(window) — not by N — is the correct amplitude-preserving
normalisation when a non-rectangular window is applied.
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
    >>> p_w = estimate_phasor(sig, fs, f0, window=True)
    >>> assert abs(abs(p_w) - A) < 1.0   # Hann window — small leakage error
    """
    signal = np.asarray(signal, dtype=float)
    N = len(signal)

    if window:
        w    = np.hanning(N)
        sig_w = signal * w
        norm  = w.sum()          # правильная нормировка для оконного FFT
    else:
        sig_w = signal
        norm  = float(N)

    X = np.fft.rfft(sig_w)
    k = int(round(f0 * N / fs))
    # Множитель 2: односторонний спектр → пиковая амплитуда
    phasor = 2.0 * X[k] / norm
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

    Examples
    --------
    >>> import numpy as np
    >>> fs, f0, N = 1000, 50, 400
    >>> t = np.arange(N) / fs
    >>> A = 100.0
    >>> sigs = np.stack([
    ...     A * np.sin(2*np.pi*f0*t),
    ...     A * np.sin(2*np.pi*f0*t - 2*np.pi/3),
    ...     A * np.sin(2*np.pi*f0*t + 2*np.pi/3),
    ... ])
    >>> p = estimate_phasors_batch(sigs, fs, f0, window=False)
    >>> assert np.all(np.abs(np.abs(p) - A) < 0.01)
    """
    signals = np.asarray(signals, dtype=float)
    N = signals.shape[-1]

    if window:
        w     = np.hanning(N)
        sig_w = signals * w
        norm  = w.sum()
    else:
        sig_w = signals
        norm  = float(N)

    X = np.fft.rfft(sig_w, axis=-1)
    k = int(round(f0 * N / fs))
    phasors = 2.0 * X[..., k] / norm
    return phasors
