#!/usr/bin/env python3
"""
Visualize Data Augmentation Effects
===================================

Plots examples of:
1. Time shifting variations
2. Gaussian noise at different SNR levels
3. Combined augmentation

Usage:
    python scripts/visualize_augmentation.py \
        --input data/data_training/1A_0.5km.csv \
        --output visualizations/
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from data.augmentation import TimeShiftAugmentation, GaussianNoiseAugmentation


def plot_time_shift_visualization(csv_path: str, output_dir: str):
    """
    Plot original + time-shifted versions.
    """
    df = pd.read_csv(csv_path)
    signal_col = 'CT1IA'  # Use Phase A current for visualization
    
    time_shift = TimeShiftAugmentation(seq_length=401)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Time Shift Augmentation (Phase A Current - CT1IA)', fontsize=16, fontweight='bold')
    
    # Original
    axes[0, 0].plot(df[signal_col].values, linewidth=2, color='#0284c7')
    axes[0, 0].set_title('Original', fontweight='bold')
    axes[0, 0].set_ylabel('Amplitude [A]')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=200, color='red', linestyle='--', alpha=0.5, label='Typical fault start')
    axes[0, 0].legend()
    
    # Shift left variations
    for idx, shift_amount in enumerate([10, 30, 50]):
        df_shifted = time_shift.shift_left(df, shift_amount)
        row, col = (idx + 1) // 2, (idx + 1) % 2
        axes[row, col].plot(df_shifted[signal_col].values, linewidth=2, color='#f97316')
        axes[row, col].set_title(f'Shift Left {shift_amount}px', fontweight='bold')
        axes[row, col].set_ylabel('Amplitude [A]')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].axvline(x=200, color='red', linestyle='--', alpha=0.3)
    
    # Shift right
    df_shifted = time_shift.shift_right(df, 30)
    axes[2, 1].plot(df_shifted[signal_col].values, linewidth=2, color='#10b981')
    axes[2, 1].set_title('Shift Right 30px', fontweight='bold')
    axes[2, 1].set_ylabel('Amplitude [A]')
    axes[2, 1].set_xlabel('Sample')
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].axvline(x=200, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '01_time_shift_variations.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_noise_augmentation(csv_path: str, output_dir: str):
    """
    Plot original + noise at different SNR levels.
    """
    df = pd.read_csv(csv_path)
    signal_col = 'CT1IA'
    signal = df[signal_col].values
    
    noise_aug = GaussianNoiseAugmentation()
    snr_levels = [1, 5, 10, 20, 40]
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle('Gaussian Noise Augmentation at Different SNR Levels\n(Phase A Current - CT1IA)', 
                 fontsize=16, fontweight='bold')
    
    # Original (no noise)
    axes[0, 0].plot(signal, linewidth=2, color='#0284c7')
    axes[0, 0].set_title('Original (No Noise)', fontweight='bold', fontsize=12)
    axes[0, 0].set_ylabel('Amplitude [A]')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([signal.min() - 20, signal.max() + 20])
    
    # SNR augmented
    colors = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e']
    
    for idx, snr_db in enumerate(snr_levels):
        df_noisy = noise_aug.add_gaussian_noise(df, snr_db, random_state=42)
        noisy_signal = df_noisy[signal_col].values
        
        row, col = (idx + 1) // 3, (idx + 1) % 3
        axes[row, col].plot(signal, linewidth=1, color='#0284c7', alpha=0.3, label='Original')
        axes[row, col].plot(noisy_signal, linewidth=1.5, color=colors[idx], label='Noisy')
        
        # Calculate actual SNR
        noise = noisy_signal - signal
        noise_std = np.std(noise)
        signal_rms = np.sqrt(np.mean(signal ** 2))
        actual_snr = 20 * np.log10(signal_rms / noise_std)
        
        label_map = {
            1: 'Very Noisy (Field)',
            5: 'Noisy (Industrial)',
            10: 'Moderate (Typical)',
            20: 'Clean (Good)',
            40: 'Very Clean (Lab)'
        }
        
        title = f'SNR {snr_db} dB - {label_map.get(snr_db, "Custom")}'
        axes[row, col].set_title(title, fontweight='bold', fontsize=11)
        axes[row, col].set_ylabel('Amplitude [A]')
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_ylim([signal.min() - 20, signal.max() + 20])
        axes[row, col].legend(loc='upper right')
    
    # Hide extra subplot
    axes[1, 2].axis('off')
    
    # Add SNR reference info
    info_text = (
        "SNR Reference (from IEEE power systems literature):\n"
        "• 1 dB: Highly noisy, real field measurements\n"
        "• 5 dB: Noisy industrial environment with EMI\n"
        "• 10 dB: Typical power system measurement\n"
        "• 20 dB: Good signal quality, normal operation\n"
        "• 40 dB: Clean lab conditions"
    )
    axes[1, 2].text(0.5, 0.5, info_text, 
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[1, 2].transAxes,
                    fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '02_gaussian_noise_snr_levels.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_combined_augmentation(csv_path: str, output_dir: str):
    """
    Plot combinations of time shift + Gaussian noise.
    """
    df = pd.read_csv(csv_path)
    signal_col = 'CT1IA'
    signal = df[signal_col].values
    
    time_shift = TimeShiftAugmentation()
    noise_aug = GaussianNoiseAugmentation()
    
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    fig.suptitle('Combined Time Shift + Gaussian Noise Augmentation\n(Phase A Current - CT1IA)', 
                 fontsize=16, fontweight='bold')
    
    # Original
    ax = fig.add_subplot(gs[0, :])
    ax.plot(signal, linewidth=2.5, color='#0284c7', label='Original')
    ax.set_title('Original Signal', fontweight='bold', fontsize=12)
    ax.set_ylabel('Amplitude [A]')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Time shift only (no noise)
    shifts = [10, 30, 50]
    for idx, shift in enumerate(shifts):
        ax = fig.add_subplot(gs[1, idx])
        df_shifted = time_shift.shift_left(df, shift)
        ax.plot(df_shifted[signal_col].values, linewidth=1.5, color='#f97316')
        ax.set_title(f'Shift Left {shift}px (No Noise)', fontweight='bold')
        ax.set_ylabel('Amplitude [A]')
        ax.grid(True, alpha=0.3)
    
    # Combined: Shift + Noise
    snr_levels = [5, 20, 40]
    for idx, snr_db in enumerate(snr_levels):
        ax = fig.add_subplot(gs[2, idx])
        
        # Apply time shift
        df_shifted = time_shift.shift_left(df, 30)
        
        # Apply noise
        df_combined = noise_aug.add_gaussian_noise(df_shifted, snr_db, random_state=42)
        
        ax.plot(df_combined[signal_col].values, linewidth=1.5, color='#10b981')
        ax.set_title(f'Shift Left 30px + SNR {snr_db} dB', fontweight='bold')
        ax.set_ylabel('Amplitude [A]')
        ax.set_xlabel('Sample')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '03_combined_augmentation.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def plot_augmentation_summary(num_originals: int = 100, output_dir: str = '.'):
    """
    Plot augmentation statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Data Augmentation Pipeline Summary', fontsize=16, fontweight='bold')
    
    # 1. Sample count growth
    stages = ['Original', 'After\nTime Shifts\n(5 left + 5 right)', 
              'After Gaussian\nNoise (5 levels)',
              'After\nBoth (Augmented)']
    counts = [
        num_originals,
        num_originals * 10,
        num_originals * 10,
        num_originals * 10 * 5
    ]
    colors_bar = ['#0284c7', '#f97316', '#10b981', '#ec4899']
    
    axes[0, 0].bar(stages, counts, color=colors_bar, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Sample Count', fontweight='bold')
    axes[0, 0].set_title('Dataset Growth', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(counts):
        axes[0, 0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')
    
    # 2. Train/Test split
    labels = ['Training\n(80%)', 'Testing\n(20%)']
    sizes = [0.8 * num_originals * 10 * 5, 0.2 * num_originals * 10 * 5]
    colors_pie = ['#0284c7', '#f97316']
    axes[0, 1].pie(sizes, labels=labels, autopct='%1.0f\n(%.1f%%)',
                   colors=colors_pie, startangle=90, 
                   textprops={'fontweight': 'bold', 'fontsize': 10})
    axes[0, 1].set_title(f'Train/Test Split\n(Total: {int(sum(sizes)):,} samples)', fontweight='bold')
    
    # 3. Time shift distribution
    time_shifts = [10, 20, 30, 40, 50]
    shift_labels = [f'{s}px' for s in time_shifts]
    shift_counts = [num_originals * 2] * len(time_shifts)  # 2 = left + right
    
    axes[1, 0].bar(shift_labels, shift_counts, color='#f97316', edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Samples per Shift', fontweight='bold')
    axes[1, 0].set_title('Time Shift Variations (per SNR level)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(shift_counts):
        axes[1, 0].text(i, v + 5, f'{int(v)}', ha='center', fontweight='bold')
    
    # 4. SNR distribution
    snr_levels = [1, 5, 10, 20, 40]
    snr_labels = ['1dB\nVery Noisy', '5dB\nNoisy', '10dB\nModerate', '20dB\nClean', '40dB\nVery Clean']
    snr_counts = [num_originals * 10] * len(snr_levels)
    colors_snr = ['#ef4444', '#f97316', '#eab308', '#84cc16', '#22c55e']
    
    axes[1, 1].bar(snr_labels, snr_counts, color=colors_snr, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Samples per SNR Level', fontweight='bold')
    axes[1, 1].set_title('SNR Distribution (per time shift)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(snr_counts):
        axes[1, 1].text(i, v + 5, f'{int(v)}', ha='center', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, '00_augmentation_summary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize data augmentation effects')
    parser.add_argument('--input', required=True, help='Path to original CSV file')
    parser.add_argument('--output', default='visualizations', help='Output directory for plots')
    parser.add_argument('--num-originals', type=int, default=100, help='Number of original samples')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Data Augmentation Visualization")
    print(f"{'='*70}")
    print(f"Input CSV: {args.input}")
    print(f"Output dir: {args.output}\n")
    
    # Generate plots
    print("Generating plots...\n")
    
    print("1. Time Shift Augmentation")
    plot_time_shift_visualization(args.input, args.output)
    
    print("2. Gaussian Noise Augmentation")
    plot_noise_augmentation(args.input, args.output)
    
    print("3. Combined Augmentation")
    plot_combined_augmentation(args.input, args.output)
    
    print("4. Augmentation Summary Statistics")
    plot_augmentation_summary(args.num_originals, args.output)
    
    print(f"\n{'='*70}")
    print(f"Visualization Complete!")
    print(f"{'='*70}")
    print(f"Output files saved to: {args.output}\n")


if __name__ == '__main__':
    main()
