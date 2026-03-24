"""Fault inception time detection and window cropping utilities.

This module implements a two-stage algorithm for precise localisation of the
fault inception moment t0 in power system oscillograms, based on the
specification provided in the project documentation:

Stage I  : coarse localisation using a 4th‑order discrete differential operator
Stage II : precise localisation using a cycle‑to‑cycle current difference index

The implementation is kept generic and reusable so it can be applied to any
1D current waveforms sampled at a known frequency, not only within this
project.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import resample


@dataclass
class FaultInceptionParams:
    """Configuration for fault inception detection.

    Attributes:
        fs_hz:          Sampling frequency of the signal in Hz.
        mains_hz:       Power system fundamental frequency (typically 50 or 60 Hz).
        coarse_top_k:   Number of largest peaks of the 4th‑order difference to
                        consider when choosing the earliest candidate index.
        coarse_window_ms: Half‑width of the search window around the coarse
                        index, in milliseconds.
        pre_fault_ms:   Amount of pre‑fault history to keep when cropping
                        around t0, in milliseconds.
        post_fault_ms:  Amount of post‑fault data to keep after t0, in ms.
        threshold_mult: Multiplier for the adaptive threshold based on the
                        mean value of the derivative of the cycle index
                        on the pre‑fault segment.
    """

    fs_hz: float = 2000.0
    mains_hz: float = 50.0
    coarse_top_k: int = 5
    coarse_window_ms: float = 200.0
    pre_fault_ms: float = 20.0
    post_fault_ms: float = 60.0
    threshold_mult: float = 1.0


# ---------------------------------------------------------------------------
# Core 1‑phase algorithms
# ---------------------------------------------------------------------------


def _fourth_order_difference(i: np.ndarray) -> np.ndarray:
    """Compute 4th‑order discrete difference D4(k).

    D4(k) = i(k+2) - 4 i(k+1) + 6 i(k) - 4 i(k-1) + i(k-2)

    Implemented via convolution with kernel [1, -4, 6, -4, 1] using
    'same' padding to keep the original length.
    """

    kernel = np.array([1.0, -4.0, 6.0, -4.0, 1.0], dtype=np.float32)
    # Use mode="same" to preserve length; boundary behaviour is acceptable for
    # coarse localisation of fast transients.
    return np.convolve(i.astype(np.float32), kernel, mode="same")


def _coarse_fault_index(i: np.ndarray, params: FaultInceptionParams) -> int:
    """Coarse localisation of the fault interval using D4(k).

    The earliest index among the top‑K absolute maxima of D4(k) is returned,
    which corresponds to the first strong high‑frequency disturbance.
    """

    d4 = np.abs(_fourth_order_difference(i))
    if d4.size == 0:
        raise ValueError("Signal for coarse fault detection is empty")

    k = min(params.coarse_top_k, d4.size)
    if k <= 0:
        # Fallback: just take the global maximum
        return int(np.argmax(d4))

    # np.argpartition is O(n) and works for large arrays
    top_idx = np.argpartition(d4, -k)[-k:]
    return int(np.min(top_idx))


def _cycle_difference_index(
    i: np.ndarray, params: FaultInceptionParams
) -> Tuple[np.ndarray, int]:
    """Compute cycle‑to‑cycle difference index Δi(k).

    Δi(k) = |i(k) - i(k-N)| - |i(k-N) - i(k-2N)|

    where N is the number of samples per mains period.
    """

    fs = float(params.fs_hz)
    mains = float(params.mains_hz)
    if fs <= 0 or mains <= 0:
        raise ValueError("Sampling frequency and mains frequency must be positive")

    # Number of points per mains cycle (rounded to nearest integer)
    N = max(int(round(fs / mains)), 1)

    di = np.zeros_like(i, dtype=np.float32)
    if i.size < 2 * N + 1:
        # Not enough data to compute two full cycles – return zeros
        return di, N

    # Vectorised computation for k >= 2N
    # Build shifted views using slicing; pad the beginning with zeros via
    # explicit indexing.
    i_k = i[2 * N :]
    i_k_N = i[N : -N]
    i_k_2N = i[: -2 * N]

    a = np.abs(i_k - i_k_N)
    b = np.abs(i_k_N - i_k_2N)
    di[2 * N :] = a - b
    return di, N


def detect_t0_single_phase(
    i: np.ndarray, params: FaultInceptionParams
) -> Optional[int]:
    """Detect fault inception index t0 for a single current waveform.

    The procedure closely follows the description in the project PDFs:

    1) Coarse localisation via 4th‑order difference D4(k) and earliest index
       among top‑K peaks.
    2) Construction of cycle‑difference index Δi(k) and its first derivative
       Δ'i(k) = Δi(k) - Δi(k-1) inside a wide window around the coarse index.
    3) Adaptive threshold based on the mean of Δ'i(k) on the pre‑fault part of
       the window. The first index where Δ'i(k) exceeds this threshold is
       returned as t0.
    """

    if i.ndim != 1:
        raise ValueError("Signal i must be 1‑D for single‑phase detection")
    if i.size < 10:
        return None

    # Stage I: coarse index
    coarse_idx = _coarse_fault_index(i, params)

    # Window around coarse index (± coarse_window_ms)
    radius_samples = int(round(params.coarse_window_ms * 1e-3 * params.fs_hz))
    radius_samples = max(radius_samples, 1)
    start = max(0, coarse_idx - radius_samples)
    end = min(i.size, coarse_idx + radius_samples)
    if end - start < 3:
        # Degenerate window – fall back to coarse index
        return int(coarse_idx)

    di, _ = _cycle_difference_index(i, params)
    window = di[start:end]

    # First derivative of the index inside the window
    d_idx = np.zeros_like(window, dtype=np.float32)
    d_idx[1:] = window[1:] - window[:-1]

    # Pre‑fault segment: everything before the coarse index within the window
    rel_coarse = coarse_idx - start
    pre_end = max(1, min(rel_coarse, d_idx.size))
    pre_segment = d_idx[:pre_end]

    if np.allclose(pre_segment, 0.0):
        # If pre‑fault derivative is numerically zero (very flat), fall back
        # to using the whole window to estimate a noise level.
        pre_segment = d_idx

    threshold = float(pre_segment.mean() * params.threshold_mult)

    # Scan forward from the end of the pre‑fault segment to find first
    # crossing of the adaptive threshold.
    for local_k in range(pre_end, d_idx.size):
        if d_idx[local_k] > threshold:
            return int(start + local_k)

    # If nothing crosses the threshold, use coarse index as a safe default.
    return int(coarse_idx)


# ---------------------------------------------------------------------------
# Multi‑phase helpers and window cropping
# ---------------------------------------------------------------------------


def detect_t0_multi_phase(
    currents: np.ndarray,
    params: FaultInceptionParams,
) -> Optional[int]:
    """Detect t0 across several phase currents.

    Args:
        currents: Array with shape (num_phases, T) holding phase currents.

    Returns:
        Global t0 index (earliest among phases) in sample indices, or None
        if detection failed for all phases.
    """

    if currents.ndim != 2:
        raise ValueError("currents must have shape (num_phases, T)")

    t_candidates = []
    for ph in range(currents.shape[0]):
        t0 = detect_t0_single_phase(currents[ph], params)
        if t0 is not None:
            t_candidates.append(int(t0))

    if not t_candidates:
        return None
    return int(min(t_candidates))


def crop_around_t0(
    signals: np.ndarray,
    t0_idx: int,
    params: FaultInceptionParams,
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Crop multi‑channel signal around t0 and optionally resample.

    Args:
        signals:       Array of shape (T, C) – time along axis 0.
        t0_idx:        Fault inception index in the original signal (samples).
        params:        FaultInceptionParams instance.
        target_length: Optional number of samples in the output window.
                       If None, the raw cropped window is returned.

    Returns:
        cropped: Signal with shape (T_crop, C) or (target_length, C).
        t0_local: Index of t0 within the returned window.
    """

    if signals.ndim != 2:
        raise ValueError("signals must have shape (T, C)")

    T, _ = signals.shape
    t0_idx = int(np.clip(t0_idx, 0, max(T - 1, 0)))

    pre_samp = int(round(params.pre_fault_ms * 1e-3 * params.fs_hz))
    post_samp = int(round(params.post_fault_ms * 1e-3 * params.fs_hz))
    pre_samp = max(pre_samp, 1)
    post_samp = max(post_samp, 1)

    start = max(0, t0_idx - pre_samp)
    end = min(T, t0_idx + post_samp)

    if end <= start:
        # Degenerate – return original signal
        return signals.copy(), int(t0_idx)

    window = signals[start:end, :].astype(np.float32, copy=True)

    if target_length is not None and window.shape[0] != target_length:
        # Resample along time axis to target_length samples per channel
        window = resample(window, target_length, axis=0).astype(np.float32, copy=False)
        # Place t0 proportionally at the same relative position inside the new window
        total = pre_samp + post_samp
        if total <= 0:
            t0_local = min(target_length // 4, target_length - 1)
        else:
            frac = pre_samp / float(total)
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
    """High‑level helper: detect t0 and return cropped multi‑channel window.

    Args:
        signals:                 Array of shape (T, C) with all channels
                                 (currents + voltages).
        params:                  FaultInceptionParams configuration.
        current_channel_indices: Indices of columns corresponding to phase
                                 currents used for t0 detection.
        target_length:           Optional length of the output window.

    Returns:
        cropped_signals: Array of shape (T_crop, C) or (target_length, C).
        t0_local:        Index of t0 within cropped_signals, or None if
                         detection failed and no cropping was applied.
    """

    if signals.ndim != 2:
        raise ValueError("signals must have shape (T, C)")

    T, C = signals.shape
    if C == 0 or T == 0:
        return signals, None

    # Extract phase currents for detection
    current_channel_indices = tuple(current_channel_indices)
    if any(ch < 0 or ch >= C for ch in current_channel_indices):
        raise IndexError("current_channel_indices contain out‑of‑range values")

    currents = signals[:, current_channel_indices].T  # (num_phases, T)
    t0_global = detect_t0_multi_phase(currents, params)

    if t0_global is None:
        # Detection failed – optionally just resample to target_length
        if target_length is not None and T != target_length:
            resized = resample(signals, target_length, axis=0).astype(np.float32, copy=False)
            return resized, None
        return signals, None

    cropped, t0_local = crop_around_t0(signals, t0_global, params, target_length)
    return cropped, int(t0_local)
