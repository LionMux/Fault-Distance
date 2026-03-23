"""
High-level functions for three-phase power system signals.
Combines Fortescue transform (core.py) with phasor estimation (fourier.py).
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

    Parameters
    ----------
    xa, xb, xc : np.ndarray, shape (N,)
        Time-domain waveforms of phases A, B, C.
    fs : float
        Sampling frequency in Hz.
    f0 : float
        Fundamental frequency in Hz.
    window : bool
        Apply Hann window before FFT.

    Returns
    -------
    dict with keys:
        'X0', 'X1', 'X2' : complex — zero/positive/negative sequence phasors
        'X0_mag', 'X1_mag', 'X2_mag' : float — magnitudes
        'X0_ang', 'X1_ang', 'X2_ang' : float — angles in radians
    """
    signals = np.stack([xa, xb, xc], axis=0)  # (3, N)
    phasors = estimate_phasors_batch(signals, fs, f0, window)  # (3,)

    x0, x1, x2 = abc_to_seq(phasors[0], phasors[1], phasors[2])

    return {
        'X0': x0, 'X1': x1, 'X2': x2,
        'X0_mag': abs(x0), 'X1_mag': abs(x1), 'X2_mag': abs(x2),
        'X0_ang': float(np.angle(x0)),
        'X1_ang': float(np.angle(x1)),
        'X2_ang': float(np.angle(x2)),
    }


def instantaneous_symseq(
    xa: np.ndarray,
    xb: np.ndarray,
    xc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute instantaneous symmetrical components (no FFT).

    Applies the Fortescue matrix to each time sample directly.
    Result is complex-valued time series — useful for visualising
    how sequence components evolve during fault transients.

    Parameters
    ----------
    xa, xb, xc : np.ndarray, shape (N,)
        Real-valued time-domain waveforms.

    Returns
    -------
    x0, x1, x2 : np.ndarray, shape (N,), dtype complex128
        Instantaneous zero / positive / negative sequence time series.
    """
    xa = np.asarray(xa, dtype=complex)
    xb = np.asarray(xb, dtype=complex)
    xc = np.asarray(xc, dtype=complex)
    return abc_to_seq(xa, xb, xc)
