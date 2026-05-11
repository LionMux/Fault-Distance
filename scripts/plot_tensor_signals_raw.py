"""
Визуализация сигналов, которые идут в тензор модели (ДО нормализации).
Показывает все 12 каналов в физических единицах + этапы фильтрации.

Usage:
    python scripts/plot_tensor_signals_raw.py --csv data/data_training/1A_0.5km.csv --output output/thesis/raw_tensor_signals
    python scripts/plot_tensor_signals_raw.py --csv data/data_training/1A_25km.csv --output output/thesis/raw_tensor_signals
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config, load_config

from data.preprocessing import center_by_prehistory, remove_dc_period, preprocess_signal_for_inference
from data.fault_inception import FaultInceptionParams, detect_t0_and_crop
from data.preprocessing import sliding_window_symseq

SIGNAL_COLS = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
FS_COL = 'fs_hz'


def plot_stages(csv_path, cfg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fname_base = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path)
    distance = float(df['distance_km'].iloc[0])

    if FS_COL in df.columns:
        fs = float(df[FS_COL].iloc[0])
    else:
        fs = float(getattr(cfg, 'SAMPLING_FREQ_HZ', 2000.0))

    # ------------------------------------------------------------------
    # 1. RAW signal
    # ------------------------------------------------------------------
    sig_raw = df[SIGNAL_COLS].values.astype(np.float32)  # (T, 6)
    T_raw = sig_raw.shape[0]
    t_raw_ms = np.arange(T_raw) / fs * 1000.0

    # ------------------------------------------------------------------
    # 2. After centering + DC removal
    # ------------------------------------------------------------------
    sig_centered = center_by_prehistory(sig_raw.T, fs=fs, pre_ms=20.0).T
    sig_dc_removed = remove_dc_period(sig_centered.T, fs=fs, f_net=cfg.MAINS_FREQ_HZ).T

    # ------------------------------------------------------------------
    # 3. After t0 crop
    # ------------------------------------------------------------------
    sig_t0 = sig_dc_removed.copy()
    t0_local = None
    if cfg.T0_ENABLED:
        params = FaultInceptionParams(
            fs_hz=fs,
            mains_hz=cfg.MAINS_FREQ_HZ,
            pre_fault_ms=cfg.T0_PRE_MS,
            post_fault_ms=cfg.T0_POST_MS,
            eta_I=cfg.T0_ETA_I,
            eta_U=cfg.T0_ETA_U,
        )
        try:
            sig_t0, t0_local = detect_t0_and_crop(
                sig_t0,
                params,
                current_channel_indices=(0, 1, 2),
                voltage_channel_indices=(3, 4, 5),
            )
        except Exception as e:
            print(f"[WARN] t0 detection failed: {e}")

    T_t0 = sig_t0.shape[0]
    t_t0_ms = np.arange(T_t0) / fs * 1000.0
    if t0_local is not None:
        t0_ms = t0_local / fs * 1000.0
    else:
        t0_ms = None

    # ------------------------------------------------------------------
    # 4. Final tensor (12-ch) via preprocess_signal_for_inference
    # ------------------------------------------------------------------
    sig_final, _, _ = preprocess_signal_for_inference(df, cfg)
    # sig_final shape: (NUM_CHANNELS, SEQ_LENGTH) — already 12-ch if symseq enabled
    # NO normalization applied
    T_final = sig_final.shape[1]
    t_final_ms = np.arange(T_final) / fs * 1000.0

    # ==================================================================
    # Plot 1: Raw vs DC-removed (6 channels)
    # ==================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()
    labels_6 = ['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C']
    units_6 = ['A', 'A', 'A', 'kV', 'kV', 'kV']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#e41a1c', '#377eb8', '#4daf4a']

    for idx in range(6):
        ax = axes[idx]
        ax.plot(t_raw_ms, sig_raw[:, idx], color=colors[idx], linewidth=0.8,
                label='Raw CSV', alpha=0.5)
        ax.plot(t_raw_ms, sig_dc_removed[:, idx], color=colors[idx], linewidth=1.2,
                label='After DC removal', linestyle='-')
        ax.set_xlabel('Time, ms', fontsize=10)
        ax.set_ylabel(f'{labels_6[idx]}, {units_6[idx]}', fontsize=10)
        ax.set_title(f'{labels_6[idx]}  (distance={distance} km)', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Stage 1: DC removal  |  {fname_base}', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out1 = os.path.join(output_dir, f'{fname_base}_stage1_dc_removal.png')
    plt.savefig(out1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out1}')

    # ==================================================================
    # Plot 2: t0 crop window (6 channels)
    # ==================================================================
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    axes = axes.flatten()

    for idx in range(6):
        ax = axes[idx]
        ax.plot(t_raw_ms, sig_dc_removed[:, idx], color=colors[idx], linewidth=0.8,
                label='Before crop', alpha=0.4)
        ax.plot(t_t0_ms, sig_t0[:, idx], color=colors[idx], linewidth=1.5,
                label='After t0 crop')
        if t0_ms is not None:
            ax.axvline(t0_ms, color='black', linestyle='--', linewidth=1.0, alpha=0.7,
                       label='t0')
        ax.set_xlabel('Time, ms', fontsize=10)
        ax.set_ylabel(f'{labels_6[idx]}, {units_6[idx]}', fontsize=10)
        ax.set_title(f'{labels_6[idx]}', fontsize=11)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Stage 2: t0 crop  |  {fname_base}  (window={T_t0} samples @ {fs:.0f} Hz)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = os.path.join(output_dir, f'{fname_base}_stage2_t0_crop.png')
    plt.savefig(out2, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out2}')

    # ==================================================================
    # Plot 3: Final 12-channel tensor (NO normalization)
    # ==================================================================
    fig, axes = plt.subplots(4, 3, figsize=(16, 18), dpi=300)
    axes = axes.flatten()

    labels_12 = [
        'I_A', 'I_B', 'I_C',
        '|I1|', '|I2|', '|I0|',
        'U_A', 'U_B', 'U_C',
        '|U1|', '|U2|', '|U0|',
    ]
    units_12 = [
        'A', 'A', 'A',
        'A', 'A', 'A',
        'kV', 'kV', 'kV',
        'kV', 'kV', 'kV',
    ]
    colors_12 = [
        '#e41a1c', '#377eb8', '#4daf4a',
        '#e41a1c', '#377eb8', '#4daf4a',
        '#e41a1c', '#377eb8', '#4daf4a',
        '#e41a1c', '#377eb8', '#4daf4a',
    ]

    for idx in range(12):
        ax = axes[idx]
        ax.plot(t_final_ms, sig_final[idx], color=colors_12[idx], linewidth=1.0)
        ax.set_xlabel('Time, ms', fontsize=10)
        ax.set_ylabel(f'{labels_12[idx]}, {units_12[idx]}', fontsize=10)
        ax.set_title(f'Ch{idx}: {labels_12[idx]}', fontsize=11)
        ax.grid(True, alpha=0.3)

        # Add stats text
        y_min = sig_final[idx].min()
        y_max = sig_final[idx].max()
        y_mean = sig_final[idx].mean()
        ax.text(0.02, 0.98, f'min={y_min:.2f}\nmax={y_max:.2f}\nmean={y_mean:.2f}',
                transform=ax.transAxes, fontsize=8, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle(
        f'Final tensor (12 channels, NO normalization)  |  {fname_base}\n'
        f'distance={distance} km  |  fs={fs:.0f} Hz  |  length={T_final} samples',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out3 = os.path.join(output_dir, f'{fname_base}_stage3_tensor_12ch_raw.png')
    plt.savefig(out3, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[OK] Saved: {out3}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file')
    parser.add_argument('--config', type=str, default='configs/compare_pu.yaml',
                        help='Config file (used for preprocessing params only)')
    parser.add_argument('--output', type=str, default='output/thesis/raw_tensor_signals',
                        help='Output directory')
    args = parser.parse_args()

    cfg = load_config(args.config) if os.path.exists(args.config) else Config()
    # Force normalize=False visualization regardless of config mode
    plot_stages(args.csv, cfg, args.output)
    print(f'\nAll plots saved to: {args.output}')


if __name__ == '__main__':
    main()
