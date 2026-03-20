# See data/augmentation.py for full implementation.
# This file is a copy of data/augmentation.py placed in the src package.
"""
Data Augmentation for Fault Distance Oscillograms
================================================

This module augments the training dataset by:
1. TIME SHIFTING: Artificially shifts fault event in time (5 variations per original)
2. GAUSSIAN NOISE: Adds realistic SNR levels based on power system measurements

Usage:
    from fault_distance.data.augmentation import AugmentationPipeline
    pipeline = AugmentationPipeline()
    pipeline.augment_dataset('data/data_training', 'data/data_augmented')
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
from pathlib import Path


class TimeShiftAugmentation:
    """Shift fault event timing while maintaining 401-row constraint."""

    FAULT_EVENT_START = 200

    def __init__(self, seq_length: int = 400):
        self.seq_length = seq_length

    def shift_left(self, df: pd.DataFrame, shift_amount: int = 10) -> pd.DataFrame:
        if len(df) != self.seq_length:
            raise ValueError(f"Input must have {self.seq_length} rows, got {len(df)}")
        if shift_amount <= 0:
            return df.copy()
        df_shifted = df.iloc[shift_amount:].reset_index(drop=True)
        pad_rows = pd.concat(
            [pd.DataFrame([df_shifted.iloc[-1]]) for _ in range(shift_amount)],
            ignore_index=True
        )
        df_shifted = pd.concat([df_shifted, pad_rows], ignore_index=True)
        return df_shifted.iloc[:self.seq_length]

    def shift_right(self, df: pd.DataFrame, shift_amount: int = 10) -> pd.DataFrame:
        if len(df) != self.seq_length:
            raise ValueError(f"Input must have {self.seq_length} rows, got {len(df)}")
        if shift_amount <= 0:
            return df.copy()
        pad_rows = pd.concat(
            [pd.DataFrame([df.iloc[0]]) for _ in range(shift_amount)],
            ignore_index=True
        )
        df_shifted = pd.concat([pad_rows, df], ignore_index=True)
        return df_shifted.iloc[:self.seq_length]


class GaussianNoiseAugmentation:
    """Add realistic Gaussian noise based on SNR levels."""

    SNR_LEVELS_DB = {
        'very_noisy': 1,
        'noisy': 5,
        'moderate': 10,
        'clean': 20,
        'very_clean': 40,
    }

    def __init__(self, seq_length: int = 400, num_channels: int = 6):
        self.seq_length = seq_length
        self.num_channels = num_channels

    def calculate_noise_std(self, signal: np.ndarray, snr_db: float) -> float:
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_std = signal_rms / (10 ** (snr_db / 20))
        return noise_std

    def add_gaussian_noise(
        self,
        df: pd.DataFrame,
        snr_db: float,
        random_state: int = None
    ) -> pd.DataFrame:
        if random_state is not None:
            np.random.seed(random_state)
        df_noisy = df.copy()
        signal_columns = df.columns[1:].tolist()
        for col in signal_columns:
            signal = df_noisy[col].values.astype(np.float32)
            noise_std = self.calculate_noise_std(signal, snr_db)
            noise = np.random.normal(0, noise_std, len(signal)).astype(np.float32)
            df_noisy[col] = signal + noise
        return df_noisy

    @staticmethod
    def snr_db_to_label(snr_db: float) -> str:
        if snr_db <= 1:
            return 'very_noisy'
        elif snr_db <= 5:
            return 'noisy'
        elif snr_db <= 10:
            return 'moderate'
        elif snr_db <= 20:
            return 'clean'
        else:
            return 'very_clean'


class AugmentationPipeline:
    """Full pipeline: time shifts + Gaussian noise augmentation."""

    TIME_SHIFTS = [10, 20, 30, 40, 50]
    SNR_LEVELS = [1, 5, 10, 20, 40]

    def __init__(self, seq_length: int = 400, num_channels: int = 6, seed: int = 42):
        self.seq_length = seq_length
        self.num_channels = num_channels
        self.seed = seed
        self.time_shift = TimeShiftAugmentation(seq_length)
        self.gaussian_noise = GaussianNoiseAugmentation(seq_length, num_channels)

    def augment_single_file(
        self,
        csv_path: str,
        output_dir: str,
        file_stem: str = None,
    ) -> List[str]:
        if file_stem is None:
            file_stem = Path(csv_path).stem
        created_files = []
        df_original = pd.read_csv(csv_path)

        for i, shift_amount in enumerate(self.TIME_SHIFTS):
            df_shifted_left = self.time_shift.shift_left(df_original, shift_amount)
            df_shifted_right = self.time_shift.shift_right(df_original, shift_amount)

            for j, snr_db in enumerate(self.SNR_LEVELS):
                df_noisy_left = self.gaussian_noise.add_gaussian_noise(
                    df_shifted_left, snr_db, random_state=self.seed + i * 100 + j
                )
                snr_label = self.gaussian_noise.snr_db_to_label(snr_db)
                fname = f"{file_stem}_shift_left_{shift_amount}px_snr_{snr_db}dB_{snr_label}.csv"
                fpath = os.path.join(output_dir, fname)
                df_noisy_left.to_csv(fpath, index=False)
                created_files.append(fpath)

                df_noisy_right = self.gaussian_noise.add_gaussian_noise(
                    df_shifted_right, snr_db, random_state=self.seed + i * 100 + j + 50
                )
                fname = f"{file_stem}_shift_right_{shift_amount}px_snr_{snr_db}dB_{snr_label}.csv"
                fpath = os.path.join(output_dir, fname)
                df_noisy_right.to_csv(fpath, index=False)
                created_files.append(fpath)

        return created_files

    def augment_dataset(self, input_dir: str, output_dir: str) -> dict:
        os.makedirs(output_dir, exist_ok=True)
        csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")

        all_created = []
        for idx, csv_path in enumerate(csv_files, 1):
            fname = os.path.basename(csv_path)
            print(f"[{idx}/{len(csv_files)}] Processing {fname}...")
            created = self.augment_single_file(csv_path, output_dir)
            all_created.extend(created)

        stats = {
            'original_count': len(csv_files),
            'time_shifts': len(self.TIME_SHIFTS),
            'snr_levels': len(self.SNR_LEVELS),
            'augmentations_per_sample': 2 * len(self.TIME_SHIFTS) * len(self.SNR_LEVELS),
            'total_created': len(all_created),
        }
        return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/data_training')
    parser.add_argument('--output', default='data/data_augmented')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    pipeline = AugmentationPipeline(seed=args.seed)
    pipeline.augment_dataset(args.input, args.output)
