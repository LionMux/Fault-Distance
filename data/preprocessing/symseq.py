"""
Symmetrical-sequence (Fortescue) related utilities.

Contains:
- sliding_window_symseq
"""

from __future__ import annotations

import numpy as np


def sliding_window_symseq(sig: np.ndarray, fs: float, f0: float = 50.0, window_cycles: int = 1) -> np.ndarray:
    from symseq.core import abc_to_seq
    T = sig.shape[1]
    period = max(int(round(fs / f0)), 1)
    window_len = period * window_cycles
    half = window_len // 2
    result = np.zeros((6, T), dtype=np.float32)
    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)  # +1 для нечётной длины
        n = end - start
        w = sig[:, start:end]
        k = min(int(round(f0 * n / fs)), n - 1)
        # critical fix: hann должен иметь длину ровно n, иначе бывает broadcast error на краях
        hann = np.hanning(n).astype(np.float32)
        dft = np.fft.fft(w * hann[np.newaxis, :], axis=1)
        ph = dft[:, k] * (2.0 / np.sum(hann))

        # Currents (channels 0-2)
        i0, i1, i2 = abc_to_seq(ph[0], ph[1], ph[2])
        result[0, t] = float(abs(i1))
        result[1, t] = float(abs(i2))
        result[2, t] = float(abs(i0))

        # Voltages (channels 3-5)
        u0, u1, u2 = abc_to_seq(ph[3], ph[4], ph[5])
        result[3, t] = float(abs(u1))
        result[4, t] = float(abs(u2))
        result[5, t] = float(abs(u0))

    return result
