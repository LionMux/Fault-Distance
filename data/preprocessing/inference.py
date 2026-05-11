"""
Inference helpers and single-file preprocessing.

This module contains the exact logic that was previously in the monolithic
`data/preprocessing.py`:
- apply_pu_normalization
- preprocess_signal_for_inference
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from .dc_filters import center_by_prehistory, remove_dc_period
from .symseq import sliding_window_symseq


def apply_pu_normalization(signals: np.ndarray, cfg) -> np.ndarray:
    """Apply per-unit normalization to a signal tensor.

    Mirrors the p.u. logic in ``FaultDataset.__init__``.
    Supports both (C, T) single-sample and (N, C, T) batch shapes.
    """
    if getattr(cfg, "NORMALIZATION_MODE", "standard") != "pu":
        return signals

    Unom_kv = cfg.LINE_UNOM_KV
    S_base_MVA = getattr(cfg, "S_BASE_MVA", 100.0)

    # Base current from base power (matches notebook formula)
    Ibase_A = (S_base_MVA * 1_000_000) / (3 ** 0.5 * Unom_kv * 1000)

    sig = signals.copy().astype(np.float32)
    num_channels = sig.shape[-2] if sig.ndim >= 2 else sig.shape[0]

    # Phase currents -> p.u.
    sig[..., 0:3, :] /= Ibase_A

    if num_channels == 6:
        # Legacy 6-ch: phase voltages -> p.u. (line voltage base)
        sig[..., 3:6, :] /= Unom_kv
    else:
        # 12-ch layout
        sig[..., 3:6, :] /= Ibase_A  # phasor current magnitudes
        sig[..., 6:9, :] /= Unom_kv  # phase voltages
        sig[..., 9:12, :] /= Unom_kv  # phasor voltage magnitudes

    return sig


def preprocess_signal_for_inference(
    df: pd.DataFrame,
    cfg,
    signal_cols: Optional[List[str]] = None,
    distance_col: str = "distance_km",
    fs_col: str = "fs_hz",
) -> Tuple[np.ndarray, Optional[float], float]:
    """Preprocess a single oscillogram DataFrame for inference/testing.

    Replicates the pipeline used in ``FaultDataset.__init__``:
    centering → DC removal → t0 detection/cropping → pad/trim → symseq.
    Does not apply normalization (use saved scalers afterwards).
    """
    if signal_cols is None:
        signal_cols = [
            "CT1IA",
            "CT1IB",
            "CT1IC",
            "S1) BUS1UA",
            "S1) BUS1UB",
            "S1) BUS1UC",
        ]

    seq_length = int(getattr(cfg, "SEQ_LENGTH", 400))
    mains_hz = float(getattr(cfg, "MAINS_FREQ_HZ", 50.0))

    # Distance label
    distance = float(df[distance_col].iloc[0]) if distance_col in df.columns else None

    # Sampling frequency
    if fs_col in df.columns:
        fs = float(df[fs_col].iloc[0])
    else:
        fs = float(getattr(cfg, "SAMPLING_FREQ_HZ", 2000.0))

    # Raw signal (T, 6)
    sig = df[signal_cols].values.astype(np.float32)

    # Stage 2: centering + DC-period removal
    if getattr(cfg, "REMOVE_DC_ENABLED", False):
        sig = center_by_prehistory(sig.T, fs=fs, pre_ms=20.0).T
        sig = remove_dc_period(sig.T, fs=fs, f_net=mains_hz).T

    # Stage 3: fault-inception detection and cropping
    if getattr(cfg, "T0_ENABLED", False):
        from .fault_inception import FaultInceptionParams, detect_t0_and_crop

        params = FaultInceptionParams(
            fs_hz=fs,
            mains_hz=mains_hz,
            pre_fault_ms=float(getattr(cfg, "T0_PRE_MS", 50.0)),
            post_fault_ms=float(getattr(cfg, "T0_POST_MS", 150.0)),
            eta_I=float(getattr(cfg, "T0_ETA_I", 0.5)),
            eta_U=float(getattr(cfg, "T0_ETA_U", 0.85)),
        )
        try:
            sig, _ = detect_t0_and_crop(
                sig,
                params,
                current_channel_indices=(0, 1, 2),
                voltage_channel_indices=(3, 4, 5),
            )
        except Exception:
            # fallback: сигналы дальше уйдут как есть (без pad/trim до seq_length)
            pass

    # IMPORTANT:
    # После t0-crop больше НЕ делаем pad/trim до seq_length.
    # Длина должна получаться только за счёт параметров t0_pre_ms/t0_post_ms.
    sig = sig.T

    # pu-согласование: если режим pu, то симм. составляющие нужно считать
    # именно из pu-значений токов/напряжений.
    if getattr(cfg, "NORMALIZATION_MODE", "standard") == "pu":
        sig = apply_pu_normalization(sig, cfg)

    # Stage 5: symmetrical components
    if getattr(cfg, "SYMSEQ_ENABLED", False):
        sym_ch = sliding_window_symseq(sig, fs=fs, f0=mains_hz, window_cycles=1)
        sig = np.vstack([sig[0:3, :], sym_ch[0:3, :], sig[3:6, :], sym_ch[3:6, :]])

    return sig, distance, fs
