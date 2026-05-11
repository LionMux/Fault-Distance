"""Fault inception time detection and window cropping utilities.

Notebook-compliant implementation of the RMS-based fault inception detection
from the fault_location_preprocessing_pipeline notebook.

Algorithm (notebook Stage 3):
    1. Compute RMS in two adjacent windows of length T/4 (k = fs/(4*f_net)).
    2. For every sample compute I_post/I_pre and U_post/U_pre.
    3. Fault is declared when max phase current rises (>1+eta_I)
       AND min phase voltage drops (<eta_U).
    4. Skip the first 2 windows to avoid false triggering on start-up.

Window cropping (notebook Stage 4):
    - pre_fault_ms  = 50 ms  (notebook)
    - post_fault_ms = 150 ms (notebook)
    - total = 200 ms => 1000 samples @ Fs=5000 Hz
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
# resample is no longer used inside crop_around_t0 (no resampling / target-length fitting)

@dataclass
class FaultInceptionParams:
    """Configuration for fault inception detection (notebook-compliant).

    Attributes:
        fs_hz:         Sampling frequency of the signal in Hz.
                       **No default** — must be supplied explicitly.
        mains_hz:      Power system fundamental frequency (50 or 60 Hz).
        eta_I:         Current-rise threshold multiplier (default 0.5 = 50%%).
        eta_U:         Voltage-drop threshold multiplier (default 0.85 = 85%%).
        pre_fault_ms:  Amount of pre-fault history to keep (notebook: 50 ms).
        post_fault_ms: Amount of post-fault data to keep (notebook: 150 ms).
    """

    fs_hz: float            # NO default — must be provided (read from CSV)
    mains_hz: float = 50.0
    eta_I: float = 0.5
    eta_U: float = 0.85
    pre_fault_ms: float = 50.0
    post_fault_ms: float = 150.0


# ---------------------------------------------------------------------------
# Core RMS-based algorithm (notebook)
# ---------------------------------------------------------------------------


def _window_rms(x: np.ndarray, start: int, end: int) -> float:
    """Root-mean-square over a slice."""
    return float(np.sqrt(np.mean(x[start:end] ** 2)))


def detect_t0_rms(
    currents: np.ndarray,
    voltages: np.ndarray,
    params: FaultInceptionParams,
) -> Optional[int]:
    """Detect fault inception using the notebook RMS-based threshold algorithm.

    Args:
        currents: Array shape (3, T) — [IA, IB, IC].
        voltages: Array shape (3, T) — [UA, UB, UC].
        params:   FaultInceptionParams — fs_hz must be set.

    Returns:
        Fault inception sample index or None if not detected.
    """
    if currents.ndim != 2 or voltages.ndim != 2:
        raise ValueError("currents and voltages must be 2-D with shape (3, T)")
    if currents.shape[0] != 3 or voltages.shape[0] != 3:
        raise ValueError("Exactly 3 phases required for currents and voltages")

    fs = float(params.fs_hz)
    f_net = float(params.mains_hz)
    k = int(fs / (4 * f_net))
    n = currents.shape[1]

    if n < 2 * k:
        return None

    ia, ib, ic = currents[0], currents[1], currents[2]
    ua, ub, uc = voltages[0], voltages[1], voltages[2]

    I_ratio = np.ones(n, dtype=np.float32)
    U_ratio = np.ones(n, dtype=np.float32)

    for i in range(k, n - k):
        I_pre = max(
            _window_rms(ia, i - k, i),
            _window_rms(ib, i - k, i),
            _window_rms(ic, i - k, i),
        )
        I_post = max(
            _window_rms(ia, i, i + k),
            _window_rms(ib, i, i + k),
            _window_rms(ic, i, i + k),
        )
        U_pre = min(
            _window_rms(ua, i - k, i),
            _window_rms(ub, i - k, i),
            _window_rms(uc, i - k, i),
        )
        U_post = min(
            _window_rms(ua, i, i + k),
            _window_rms(ub, i, i + k),
            _window_rms(uc, i, i + k),
        )

        I_ratio[i] = I_post / (I_pre + 1e-9) if I_pre > 1e-6 else 1.0
        U_ratio[i] = U_post / (U_pre + 1e-9) if U_pre > 1e-6 else 1.0

    # Skip first 2 windows to avoid false triggering on start-up transients
    skip = 2 * k
    candidates = np.where(
        (I_ratio > (1.0 + params.eta_I)) & (U_ratio < params.eta_U)
    )[0]
    candidates = candidates[candidates >= skip]

    if len(candidates) == 0:
        return None

    return int(candidates[0])


# ---------------------------------------------------------------------------
# Window cropping
# ---------------------------------------------------------------------------


def crop_around_t0(
    signals: np.ndarray,
    t0_idx: int,
    params: FaultInceptionParams,
) -> Tuple[np.ndarray, int]:
    """Crop multi-channel signal around t0.

    ВАЖНО: функция больше НЕ делает resample/подгонку под target_length.
    Длина окна определяется только кропом относительно t0.
    """
    if signals.ndim != 2:
        raise ValueError("signals must have shape (T, C)")

    T, _ = signals.shape
    t0_idx = int(np.clip(t0_idx, 0, max(T - 1, 0)))

    pre_samp = max(int(round(params.pre_fault_ms * 1e-3 * params.fs_hz)), 1)
    post_samp = max(int(round(params.post_fault_ms * 1e-3 * params.fs_hz)), 1)
    window_len = pre_samp + post_samp

    # Хотим всегда вернуть окно фиксированной длины window_len,
    # даже если t0 близко к краям сигнала.
    # Базовое расположение окна: [t0_idx - pre_samp, t0_idx + post_samp)
    start = int(t0_idx - pre_samp)
    end = int(start + window_len)

    # Сдвигаем окно внутрь диапазона [0, T)
    if start < 0:
        end = int(end - start)  # увеличиваем end на |start|
        start = 0
    if end > T:
        shift = int(end - T)
        start = int(start - shift)
        end = T
    start = max(0, start)
    end = min(T, end)

    # Если сигнал слишком короткий, чем запрошенное окно,
    # возвращаем всё, что есть (это крайний случай).
    if end <= start:
        return signals.copy(), int(t0_idx)

    window = signals[start:end, :].astype(np.float32, copy=True)
    t0_local = int(np.clip(t0_idx - start, 0, window.shape[0] - 1))
    return window, t0_local


def detect_t0_and_crop(
    signals: np.ndarray,
    params: FaultInceptionParams,
    current_channel_indices: Sequence[int] = (0, 1, 2),
    voltage_channel_indices: Sequence[int] = (3, 4, 5),
    target_length: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[int]]:
    """High-level helper: detect t0 and return cropped multi-channel window.

    Args:
        signals:                 Array of shape (T, C).
        params:                  FaultInceptionParams — fs_hz must be set to
                                 the value read from the CSV 'fs_hz' column.
        current_channel_indices: Column indices of phase currents.
        voltage_channel_indices: Column indices of phase voltages.
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
    voltage_channel_indices = tuple(voltage_channel_indices)

    if any(ch < 0 or ch >= C for ch in current_channel_indices):
        raise IndexError("current_channel_indices contain out-of-range values")
    if any(ch < 0 or ch >= C for ch in voltage_channel_indices):
        raise IndexError("voltage_channel_indices contain out-of-range values")

    currents = signals[:, current_channel_indices].T  # (3, T)
    voltages = signals[:, voltage_channel_indices].T  # (3, T)
    t0_global = detect_t0_rms(currents, voltages, params)

    # ВАЖНО:
    # Если t0 не найден — НЕ делаем resample/подгонку длины.
    # Это убирает "тихую" подгонку окна к SEQ_LENGTH и гарантирует,
    # что обрезка (кроп) происходит только по t0.
    if t0_global is None:
        return signals, None

    cropped, t0_local = crop_around_t0(signals, t0_global, params)
    return cropped, int(t0_local)
