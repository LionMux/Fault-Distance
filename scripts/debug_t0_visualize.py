#!/usr/bin/env python3
"""
Debug / Visualize Fault Inception (t0) detection

Usage:
  python scripts/debug_t0_visualize.py --csv data/data_test_csv/1A_val1.csv --out visualizations/t0

It will:
- read CSV
- run detect_t0_and_crop (RMS-based)
- plot phase currents/voltages with vertical line at detected t0

Two types of plots:
1) Full-signal plot in original time base with line at t0_global.
2) Cropped window plot with time axis explicitly built as [-pre, +post] ms,
   so that t=0 corresponds to t0_local by construction.
"""

import os
import argparse
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data.fault_inception import FaultInceptionParams, detect_t0_and_crop

SIGNAL_COLS = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']


def _save_plot(x: np.ndarray, y: np.ndarray, x_vline: float, title: str, ylabel: str, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y, linewidth=1.5, color="#0284c7")
    ax.axvline(x=x_vline, color="#ef4444", linestyle="--", linewidth=2, label=f"vline={x_vline:.3f}ms")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time [ms]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize t0 detection on a single CSV")
    parser.add_argument("--csv", required=True, help="Path to input CSV with oscillogram data")
    parser.add_argument("--out", default="visualizations/t0", help="Output folder for plots")
    parser.add_argument("--pre-ms", type=float, default=50.0, help="pre_fault_ms (default 50)")
    parser.add_argument("--post-ms", type=float, default=150.0, help="post_fault_ms (default 150)")
    parser.add_argument("--eta-i", type=float, default=0.5, help="eta_I (default 0.5)")
    parser.add_argument("--eta-u", type=float, default=0.85, help="eta_U (default 0.85)")
    parser.add_argument("--mains-hz", type=float, default=50.0, help="mains_hz (default 50)")
    parser.add_argument("--target-length", type=int, default=0, help="0 => no resample; >0 => resample to this length")
    args = parser.parse_args()

    csv_path = args.csv
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    missing = [c for c in SIGNAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV {csv_path}: {missing}")

    if "fs_hz" not in df.columns:
        raise ValueError(f"CSV must contain 'fs_hz' column for t0 detection debug. File: {csv_path}")

    fs_hz = float(df["fs_hz"].iloc[0])

    params = FaultInceptionParams(
        fs_hz=fs_hz,
        mains_hz=args.mains_hz,
        eta_I=args.eta_i,
        eta_U=args.eta_u,
        pre_fault_ms=args.pre_ms,
        post_fault_ms=args.post_ms,
    )

    sig = df[SIGNAL_COLS].values.astype(np.float32)  # (T, 6)
    cropped, t0_local = detect_t0_and_crop(
        sig,
        params,
        current_channel_indices=(0, 1, 2),
        voltage_channel_indices=(3, 4, 5),
        target_length=(args.target_length if args.target_length and args.target_length > 0 else None),
    )

    # We need t0_global to draw full signal line; recompute with detect_t0_and_crop internals:
    currents = sig[:, 0:3].T
    voltages = sig[:, 3:6].T
    from data.fault_inception import detect_t0_rms
    t0_global = detect_t0_rms(currents, voltages, params)

    t_full_ms = (np.arange(sig.shape[0]) / fs_hz) * 1000.0

    # If not detected
    if t0_local is None or t0_global is None:
        print("t0 was NOT detected (t0_global=None). Saving raw preview plot.")
        _save_plot(t_full_ms, sig[:, 0], x_vline=t_full_ms[0], title="Raw CT1IA (t0 not detected)", ylabel="Amplitude [A]",
                    out_path=os.path.join(out_dir, "raw_preview_CT1IA.png"))
        with open(os.path.join(out_dir, "t0_debug_info.txt"), "w", encoding="utf-8") as f:
            f.write(f"t0_global={t0_global}\n")
            f.write(f"t0_local={t0_local}\n")
        return

    # full-signal plots
    cur_names = ["CT1IA", "CT1IB", "CT1IC"]
    volt_names = ["S1) BUS1UA", "S1) BUS1UB", "S1) BUS1UC"]

    for i, name in enumerate(cur_names):
        out_path = os.path.join(out_dir, f"full_t0_CT_{name}.png")
        _save_plot(
            t_full_ms, sig[:, i], t0_global / fs_hz * 1000.0,
            title=f"Full signal t0 (detected) — {name}",
            ylabel="Amplitude [A]",
            out_path=out_path,
        )

    for i, name in enumerate(volt_names, start=3):
        out_path = os.path.join(out_dir, f"full_t0_UA_{name}.png")
        _save_plot(
            t_full_ms, sig[:, i], t0_global / fs_hz * 1000.0,
            title=f"Full signal t0 (detected) — {name}",
            ylabel="Amplitude",
            out_path=out_path,
        )

    # cropped-window plot: build time axis explicitly
    # pre_samp/post_samp computed in crop_around_t0; for target_length resample,
    # t=0 still corresponds to the detected position t0_local.
    pre_samp = max(int(round(params.pre_fault_ms * 1e-3 * params.fs_hz)), 1)
    post_samp = max(int(round(params.post_fault_ms * 1e-3 * params.fs_hz)), 1)

    if cropped.shape[0] == pre_samp + post_samp:
        # no resample: time is exactly [-pre, +post)
        x_win_ms = (np.arange(cropped.shape[0]) - pre_samp) / fs_hz * 1000.0
    else:
        # resampled: build monotonic axis from [-pre, +post] linearly
        x_win_ms = np.linspace(-params.pre_fault_ms, params.post_fault_ms, num=cropped.shape[0], dtype=np.float32)

    # Within cropped window, by construction:
    # - t0_local corresponds to x_win_ms ≈ 0
    for i, name in enumerate(cur_names):
        out_path = os.path.join(out_dir, f"window_t0_CT_{name}.png")
        _save_plot(
            x_win_ms, cropped[:, i], x_vline=0.0,
            title=f"Cropped window with t0 at 0ms — {name}",
            ylabel="Amplitude [A]",
            out_path=out_path,
        )

    for i, name in enumerate(volt_names, start=3):
        out_path = os.path.join(out_dir, f"window_t0_UA_{name}.png")
        _save_plot(
            x_win_ms, cropped[:, i], x_vline=0.0,
            title=f"Cropped window with t0 at 0ms — {name}",
            ylabel="Amplitude",
            out_path=out_path,
        )

    with open(os.path.join(out_dir, "t0_debug_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"csv: {csv_path}\n")
        f.write(f"fs_hz: {fs_hz}\n")
        f.write(f"mains_hz: {params.mains_hz}\n")
        f.write(f"eta_I: {params.eta_I}\n")
        f.write(f"eta_U: {params.eta_U}\n")
        f.write(f"pre_fault_ms: {params.pre_fault_ms}\n")
        f.write(f"post_fault_ms: {params.post_fault_ms}\n")
        f.write(f"t0_global: {t0_global}\n")
        f.write(f"t0_global_time_ms: {t0_global / fs_hz * 1000.0}\n")
        f.write(f"t0_local: {t0_local}\n")
        f.write(f"cropped_len: {cropped.shape[0]}\n")

    print(f"Saved plots to: {out_dir}")
    print(f"t0_global={t0_global}, t0_global_time_ms={t0_global / fs_hz * 1000.0:.3f}")


if __name__ == "__main__":
    main()
