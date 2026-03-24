"""
High-level functions for three-phase power system signals.
Combines Fortescue transform (core.py) with phasor estimation (fourier.py).

Key principle:
    The method of symmetrical components operates on COMPLEX PHASORS —
    complex amplitudes at a specific frequency obtained via FFT/DFT.
    It is NOT a time-domain operation. Do not apply the Fortescue matrix
    to instantaneous real values.

Workflow:
    real waveforms (N samples)
          ↓  FFT → fundamental bin → complex phasor
    Xa, Xb, Xc  (complex)
          ↓  Fortescue matrix
    X0, X1, X2  (zero / positive / negative sequence phasors)
"""

import numpy as np
from .core import abc_to_seq
from .fourier import estimate_phasors_batch


def symseq_from_waveforms(
    xa: np.ndarray,
    xb: np.ndarray,
    xc: np.ndarray,
    fs: float,
    f0: float = 50.0,
    window: bool = True,
) -> dict:
    """
    Compute symmetrical sequence phasors from three-phase waveforms.

    Extracts the fundamental-frequency phasor from each phase via FFT,
    then applies the Fortescue matrix to obtain sequence components.

    Parameters
    ----------
    xa, xb, xc : np.ndarray, shape (N,)
        Real-valued time-domain waveforms of phases A, B, C.
    fs : float
        Sampling frequency in Hz.
    f0 : float
        Fundamental frequency in Hz (default 50).
    window : bool
        Apply Hann window before FFT (default True).
        Set False only when N is an exact integer multiple of fs/f0.

    Returns
    -------
    dict with keys:
        'X0', 'X1', 'X2'         : complex — zero / positive / negative phasors
        'X0_mag', 'X1_mag', 'X2_mag' : float — peak amplitudes
        'X0_ang', 'X1_ang', 'X2_ang' : float — phase angles in radians

    Examples
    --------
    >>> import numpy as np
    >>> fs, f0, N = 1000, 50, 400
    >>> t = np.arange(N) / fs
    >>> A = 100.0
    >>> # Balanced positive-sequence system
    >>> xa = A * np.sin(2*np.pi*f0*t)
    >>> xb = A * np.sin(2*np.pi*f0*t - 2*np.pi/3)
    >>> xc = A * np.sin(2*np.pi*f0*t + 2*np.pi/3)
    >>> r = symseq_from_waveforms(xa, xb, xc, fs, f0, window=False)
    >>> assert abs(r['X1_mag'] - A) < 0.1   # прямая ≈ A
    >>> assert r['X2_mag'] < 1.0            # обратная ≈ 0
    >>> assert r['X0_mag'] < 1.0            # нулевая ≈ 0
    """
    signals = np.stack([xa, xb, xc], axis=0)          # (3, N)
    phasors = estimate_phasors_batch(signals, fs, f0, window)  # (3,) complex

    x0, x1, x2 = abc_to_seq(phasors[0], phasors[1], phasors[2])

    return {
        'X0': x0, 'X1': x1, 'X2': x2,
        'X0_mag': float(abs(x0)),
        'X1_mag': float(abs(x1)),
        'X2_mag': float(abs(x2)),
        'X0_ang': float(np.angle(x0)),
        'X1_ang': float(np.angle(x1)),
        'X2_ang': float(np.angle(x2)),
    }
