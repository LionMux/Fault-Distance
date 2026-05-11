"""
DC-centering and DC-removal filters.

This module contains the exact logic that previously lived in the monolithic
`data/preprocessing.py`:
- center_by_prehistory
- remove_dc_period
- analyze_signals_full
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d


def center_by_prehistory(signal: np.ndarray, fs: float, pre_ms: float = 20.0) -> np.ndarray:
    """Subtract mean of the pre-history window (first ``pre_ms`` milliseconds).

    This matches the notebook stage where signals are centred on the
    pre-fault history before DC-period removal.

    Parameters
    ----------
    signal : np.ndarray, shape (C, T) or (T,)
        Multi-channel or single-channel oscillogram.
    fs : float
        Sampling frequency [Hz].
    pre_ms : float
        Length of the pre-history window [ms] (default 20 ms, notebook value).

    Returns
    -------
    np.ndarray
        Centred signal with the same shape as input.
    """
    pre_window = max(int(pre_ms * 1e-3 * fs), 1)
    out = np.zeros_like(signal)

    if signal.ndim == 1:
        if pre_window < signal.size:
            out = signal - np.mean(signal[:pre_window])
        else:
            out = signal - np.mean(signal)
    elif signal.ndim == 2:
        for ch in range(signal.shape[0]):
            if pre_window < signal[ch].size:
                out[ch] = signal[ch] - np.mean(signal[ch, :pre_window])
            else:
                out[ch] = signal[ch] - np.mean(signal[ch])
    else:
        raise ValueError("center_by_prehistory supports only 1-D or 2-D arrays")

    return out


def remove_dc_period(signal: np.ndarray, fs: float, f_net: float = 50.0) -> np.ndarray:
    """Remove aperiodic (DC) component by subtracting rolling mean over one mains period.

    Notebook-compliant implementation using pandas.Series.rolling (centred window)
    to exactly match the reference notebook behaviour.

    Parameters
    ----------
    signal : np.ndarray, shape (C, T) or (T,)
        Multi-channel or single-channel oscillogram.
    fs : float
        Sampling frequency [Hz].
    f_net : float
        Power-system frequency [Hz] (default 50).

    Returns
    -------
    np.ndarray
        Filtered signal with the same shape as input.
    """
    period = max(int(round(fs / f_net)), 1)
    if period < 2:
        return signal.copy()

    # Fallback: if the signal is shorter than one mains period we cannot compute
    # a meaningful rolling mean over a full period.  Subtract the global mean
    # instead (equivalent to center_by_prehistory over the whole signal).
    sig_len = signal.shape[-1] if signal.ndim >= 1 else signal.size
    if period >= sig_len:
        return center_by_prehistory(
            signal, fs=fs, pre_ms=(sig_len / fs) * 1000.0 + 1.0
        )

    filtered = np.zeros_like(signal)

    if signal.ndim == 1:
        s = pd.Series(signal)
        aper = s.rolling(window=period, center=True, min_periods=1).mean()
        half = period // 2
        if half > 0:
            aper.iloc[:half] = aper.iloc[half:period].mean()
            aper.iloc[-half:] = aper.iloc[-period:-half].mean()
        filtered = signal - aper.values
    elif signal.ndim == 2:
        for ch in range(signal.shape[0]):
            s = pd.Series(signal[ch])
            aper = s.rolling(window=period, center=True, min_periods=1).mean()
            half = period // 2
            if half > 0:
                aper.iloc[:half] = aper.iloc[half:period].mean()
                aper.iloc[-half:] = aper.iloc[-period:-half].mean()
            filtered[ch] = signal[ch] - aper.values
    else:
        raise ValueError("remove_dc_period supports only 1-D or 2-D arrays")

    return filtered


def analyze_signals_full(
    data_dict: dict,
    fs: float = 5000.0,
    fault_idx: Optional[int] = None,
    threshold: float = 5.0,
    save_path: Optional[str] = None,
) -> Dict:
    """Analyse aperiodic (DC) component for each channel.

    Notebook-compliant implementation of the DC-analysis stage.

    Parameters
    ----------
    data_dict : dict
        Mapping ``{channel_name: 1-D np.ndarray}``.
    fs : float
        Sampling frequency [Hz].
    fault_idx : int, optional
        Sample index of the fault inception (used to restrict evaluation
        to the pre-fault segment).
    threshold : float
        DC%% threshold above which the status is flagged as high.
    save_path : str, optional
        If given, the figure is saved to this path instead of being shown.

    Returns
    -------
    dict
        Mapping ``{channel_name: {'ratio': float, 'ta': float}}``.
    """
    # Local import to keep this module lightweight for training/inference
    import matplotlib.pyplot as plt

    window_size = int(fs * 0.02)  # 20 ms window
    half_w = window_size // 2
    results: Dict = {}

    names = list(data_dict.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 4 * len(names)))
    if len(names) == 1:
        axes = [axes]

    print(f"{'Канал':<12} | {'DC %':<10} | {'Ta (мс)':<10} | {'Статус'}")
    print("-" * 50)

    for i, name in enumerate(names):
        signal_full = np.array(data_dict[name]).flatten()

        if fault_idx is not None and fault_idx > window_size:
            eval_sig = signal_full[:fault_idx]
        else:
            eval_sig = signal_full

        if len(eval_sig) > window_size:
            aper_eval = uniform_filter1d(eval_sig, size=window_size, mode="nearest")
            sig_eval = eval_sig[half_w:-half_w]
            aper_eval = aper_eval[half_w:-half_w]
        else:
            aper_eval = uniform_filter1d(eval_sig, size=len(eval_sig), mode="nearest")
            sig_eval = eval_sig

        ac_eval = sig_eval - aper_eval
        peak_ac = np.max(np.abs(ac_eval))
        max_dc = np.max(np.abs(aper_eval))
        ratio = (max_dc / peak_ac * 100) if peak_ac > 0 else 0

        abs_dc = np.abs(aper_eval)
        idx_peak = int(np.argmax(abs_dc))
        val_peak = abs_dc[idx_peak]
        target = val_peak * 0.368
        ta = 0.0
        fragment = abs_dc[idx_peak:]
        for j, val in enumerate(fragment):
            if val <= target:
                ta = (j / fs) * 1000
                break

        results[name] = {"ratio": ratio, "ta": ta}
        status = "⚠️ ВЫСОКАЯ" if ratio > threshold else "✅ НОРМА"
        ta_display = f"{ta:.1f}" if ta > 0 else "> окна"
        print(f"{name:<12} | {ratio:>7.2f}% | {ta_display:>9} | {status}")

        ax = axes[i]
        t_full = np.arange(len(signal_full)) / fs * 1000
        aper_full = uniform_filter1d(signal_full, size=window_size, mode="nearest")
        ax.plot(t_full, signal_full, color="gray", alpha=0.4, label="Исходный сигнал")
        ax.plot(t_full, aper_full, color="red", linewidth=2, label="DC (апериодика)")
        ax.plot(
            t_full,
            signal_full - aper_full,
            color="blue",
            alpha=0.7,
            label="AC (основная)",
        )
        if fault_idx is not None:
            ax.axvline(
                x=fault_idx / fs * 1000,
                color="magenta",
                linestyle="--",
                alpha=0.5,
                label="КЗ",
            )
        ax.axvspan(0, half_w / fs * 1000, color="yellow", alpha=0.1)
        ax.axvspan(t_full[-1] - half_w / fs * 1000, t_full[-1], color="yellow", alpha=0.1)
        ax.set_title(f"Анализ канала: {name} (DC={ratio:.1f}%, Ta={ta_display} мс)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] DC-analysis plot saved to {save_path}")
    plt.close(fig)
    return results
