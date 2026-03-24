"""Fault inception time detection and window cropping utilities.

This module implements a two-stage algorithm for precise localisation of the
fault inception moment t0 in power system oscillograms, based on the
specification provided in the project documentation:

Stage I  : coarse localisation using a 4th-order discrete differential operator
Stage II : precise localisation using a cycle-to-cycle current difference index

The implementation is kept generic and reusable so it can be applied to any
1D current waveforms sampled at a known frequency, not only within this
project.

IMPORTANT — sampling frequency (fs_hz)
---------------------------------------
Do NOT hardcode fs_hz here or in the call-site config.  The real sampling
frequency is stored by comtrade_to_csv.py in every CSV file under the column
'fs_hz', and dataset.py reads it automatically per file before constructing
FaultInceptionParams.  fs_hz in FaultInceptionParams therefore has NO default
value: it must always be supplied explicitly so that any wrong-default bug is
caught immediately as a TypeError at construction time.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample


@dataclass
class FaultInceptionParams:
    """Configuration for fault inception detection.

    Attributes:
        fs_hz:            Sampling frequency of the signal in Hz.
                          **No default** — must be supplied explicitly.
                          dataset.py reads the real value from the 'fs_hz'
                          column written by comtrade_to_csv.py.
        mains_hz:         Power system fundamental frequency (50 or 60 Hz).
        coarse_top_k:     Number of largest peaks of the 4th-order difference
                          to consider when choosing the earliest candidate.
        coarse_window_ms: Half-width of the search window around the coarse
                          index, in milliseconds.
        pre_fault_ms:     Amount of pre-fault history to keep when cropping
                          around t0, in milliseconds.
        post_fault_ms:    Amount of post-fault data to keep after t0, in ms.
        threshold_mult:   Multiplier for the adaptive threshold based on the
                          mean of the derivative of the cycle index on the
                          pre-fault segment.
    """

    fs_hz: float            # NO default — must be provided (read from CSV)
    mains_hz: float = 50.0
    coarse_top_k: int = 5
    coarse_window_ms: float = 200.0
    pre_fault_ms: float = 20.0
    post_fault_ms: float = 60.0
    threshold_mult: float = 1.0


# ---------------------------------------------------------------------------
# Core 1-phase algorithms
# ---------------------------------------------------------------------------


def _fourth_order_difference(i: np.ndarray) -> np.ndarray:
    """Compute 4th-order discrete difference D4(k).

    D4(k) = i(k+2) - 4 i(k+1) + 6 i(k) - 4 i(k-1) + i(k-2)

    Implemented via convolution with kernel [1, -4, 6, -4, 1] using
    'same' padding to keep the original length.
    """
    kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0], dtype=np.float32)
    return np.convolve(i.astype(np.float32), kernel, mode="same")


def _coarse_fault_index(i: np.ndarray, params: FaultInceptionParams) -> int:
    """Coarse localisation of the fault interval using D4(k).

    The earliest index among the top-K absolute maxima of D4(k) is returned,
    which corresponds to the first strong high-frequency disturbance.
    """
    d4 = np.abs(_fourth_order_difference(i))
    if d4.size == 0:
        raise ValueError("Signal for coarse fault detection is empty")

    k = min(params.coarse_top_k, d4.size)
    if k <= 0:
        return int(np.argmax(d4))

    top_idx = np.argpartition(d4, -k)[-k:]
    return int(np.min(top_idx))


def _cycle_difference_index(
    i: np.ndarray, params: FaultInceptionParams
) -> Tuple[np.ndarray, int]:
    """Compute cycle-to-cycle difference index Delta_i(k).

    Delta_i(k) = |i(k) - i(k-N)| - |i(k-N) - i(k-2N)|

    where N is the number of samples per mains period.
    """
    fs = float(params.fs_hz)
    mains = float(params.mains_hz)
    if fs <= 0 or mains <= 0:
        raise ValueError("Sampling frequency and mains frequency must be positive")

    N = max(int(round(fs / mains)), 1)

    di = np.zeros_like(i, dtype=np.float32)
    if i.size < 2 * N + 1:
        return di, N

    i_k   = i[2 * N:]
    i_k_N = i[N:-N]
    i_k_2N = i[:-2 * N]

    di[2 * N:] = np.abs(i_k - i_k_N) - np.abs(i_k_N - i_k_2N)
    return di, N


def detect_t0_single_phase(
    i: np.ndarray, params: FaultInceptionParams
) -> Optional[int]:
    """Detect fault inception index t0 for a single current waveform.

    1) Coarse localisation via 4th-order difference D4(k).
    2) Cycle-difference index Delta_i(k) and its first derivative inside a
       window around the coarse index.
    3) Adaptive threshold; first index exceeding it is returned as t0.
    """
    if i.ndim != 1:
        raise ValueError("Signal i must be 1-D for single-phase detection")
    if i.size < 10:
        return None

    coarse_idx = _coarse_fault_index(i, params)

    radius_samples = max(int(round(params.coarse_window_ms * 1e-3 * params.fs_hz)), 1)
    start = max(0, coarse_idx - radius_samples)
    end   = min(i.size, coarse_idx + radius_samples)
    if end - start < 3:
        return int(coarse_idx)

    di, _ = _cycle_difference_index(i, params)
    window = di[start:end]

    d_idx = np.zeros_like(window, dtype=np.float32)
    d_idx[1:] = window[1:] - window[:-1]

    rel_coarse  = coarse_idx - start
    pre_end     = max(1, min(rel_coarse, d_idx.size))
    pre_segment = d_idx[:pre_end]

    if np.allclose(pre_segment, 0.0):
        pre_segment = d_idx

    threshold = float(pre_segment.mean() * params.threshold_mult)

    for local_k in range(pre_end, d_idx.size):
        if d_idx[local_k] > threshold:
            return int(start + local_k)

    return int(coarse_idx)


# ---------------------------------------------------------------------------
# Multi-phase helpers and window cropping
# ---------------------------------------------------------------------------


def detect_t0_multi_phase(
    currents: np.ndarray,
    params: FaultInceptionParams,
) -> Optional[int]:
    """Detect t0 across several phase currents.

    Args:
        currents: Array with shape (num_phases, T) holding phase currents.

    Returns:
        Global t0 index (earliest among phases) or None if all phases failed.
    """
    if currents.ndim != 2:
        raise ValueError("currents must have shape (num_phases, T)")

    t_candidates = []
    for ph in range(currents.shape[0]):
        t0 = detect_t0_single_phase(currents[ph], params)
        if t0 is not None:
            t_candidates.append(int(t0))

    return int(min(t_candidates)) if t_candidates else None


def crop_around_t0(
    signals: np.ndarray,
    t0_idx: int,
    params: FaultInceptionParams,
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Crop multi-channel signal around t0 and optionally resample.

    Args:
        signals:       Array of shape (T, C) – time along axis 0.
        t0_idx:        Fault inception index in the original signal.
        params:        FaultInceptionParams instance.
        target_length: If given, resample the cropped window to this length.

    Returns:
        cropped: Shape (T_crop, C) or (target_length, C).
        t0_local: Index of t0 within the returned window.
    """
    if signals.ndim != 2:
        raise ValueError("signals must have shape (T, C)")

    T, _ = signals.shape
    t0_idx = int(np.clip(t0_idx, 0, max(T - 1, 0)))

    pre_samp  = max(int(round(params.pre_fault_ms  * 1e-3 * params.fs_hz)), 1)
    post_samp = max(int(round(params.post_fault_ms * 1e-3 * params.fs_hz)), 1)

    start = max(0, t0_idx - pre_samp)
    end   = min(T, t0_idx + post_samp)

    if end <= start:
        return signals.copy(), int(t0_idx)

    window = signals[start:end, :].astype(np.float32, copy=True)

    if target_length is not None and window.shape[0] != target_length:
        window   = resample(window, target_length, axis=0).astype(np.float32, copy=False)
        total    = pre_samp + post_samp
        frac     = pre_samp / float(total) if total > 0 else 0.25
        t0_local = int(np.clip(round(frac * (target_length - 1)), 0, target_length - 1))
    else:
        t0_local = int(np.clip(t0_idx - start, 0, window.shape[0] - 1))

    return window, t0_local


def detect_t0_and_crop(
    signals: np.ndarray,
    params: FaultInceptionParams,
    current_channel_indices: Sequence[int] = (0, 1, 2),
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[int]]:
    """High-level helper: detect t0 and return cropped multi-channel window.

    Args:
        signals:                 Array of shape (T, C).
        params:                  FaultInceptionParams — fs_hz must be set to
                                 the value read from the CSV 'fs_hz' column.
        current_channel_indices: Column indices of phase currents.
        target_length:           Optional output length (resampled if needed).

    Returns:
        cropped_signals: Shape (T_crop, C) or (target_length, C).
        t0_local:        Index of t0 within cropped_signals, or None.
    """
    if signals.ndim != 2:
        raise ValueError("signals must have shape (T, C)")

    T, C = signals.shape
    if C == 0 or T == 0:
        return signals, None

    current_channel_indices = tuple(current_channel_indices)
    if any(ch < 0 or ch >= C for ch in current_channel_indices):
        raise IndexError("current_channel_indices contain out-of-range values")

    currents  = signals[:, current_channel_indices].T  # (num_phases, T)
    t0_global = detect_t0_multi_phase(currents, params)

    if t0_global is None:
        if target_length is not None and T != target_length:
            resized = resample(signals, target_length, axis=0).astype(np.float32, copy=False)
            return resized, None
        return signals, None

    cropped, t0_local = crop_around_t0(signals, t0_global, params, target_length)
    return cropped, int(t0_local)
