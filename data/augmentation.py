"""
Data Augmentation for Fault Distance Oscillograms
================================================

This module augments the training dataset by:
1. TIME SHIFTING: Artificially shifts fault event in time (5 variations per original)
2. GAUSSIAN NOISE: Adds realistic SNR levels based on power system measurements

Reference SNR levels (from IEEE power systems literature):
- SNR 1 dB   : Highly noisy, real field measurements
- SNR 5 dB   : Noisy industrial environment  
- SNR 10 dB  : Typical power system measurement
- SNR 20 dB  : Good signal quality
- SNR 40 dB  : Clean lab conditions

Total augmentation strategy:
- 100 original samples
- × 5 time shifts      = 500 samples
- × 5 noise levels     = 2,500 samples

Usage:
    python -c "from data.augmentation import AugmentationPipeline; \
               pipeline = AugmentationPipeline(); \
               pipeline.augment_dataset('data/data_training', 'data/data_augmented')"

Or as a script:
    python data/augmentation.py --input data/data_training --output data/data_augmented
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

    # Key index where most fault action happens (roughly 200 out of 401 rows)
    FAULT_EVENT_START = 200
    
    def __init__(self, seq_length: int = 401):
        """
        Args:
            seq_length: Expected number of rows in output (default 401)
        """
        self.seq_length = seq_length
        
    def shift_left(self, df: pd.DataFrame, shift_amount: int = 10) -> pd.DataFrame:
        """
        Shift fault event LEFT by moving it earlier in time.
        
        Timeline visualization:
        Original: [Pre-fault data ... 200-401: FAULT ... Post-fault]
        After shift_left(10):
               [Pre-fault ... 190-391: FAULT ... Post-fault TRIMMED]
        
        Args:
            df: Input DataFrame with 401 rows
            shift_amount: Number of rows to shift left (default 10)
            
        Returns:
            New DataFrame with:
            - Fault event shifted earlier
            - Last shift_amount rows trimmed
            - Maintained as 401 rows
        """
        if len(df) != self.seq_length:
            raise ValueError(f"Input must have {self.seq_length} rows, got {len(df)}")
        
        if shift_amount <= 0:
            return df.copy()
            
        # Remove first shift_amount rows (shifts fault event earlier in time)
        df_shifted = df.iloc[shift_amount:].reset_index(drop=True)
        
        # Pad end with last row (maintains 401 rows)
        pad_rows = pd.concat(
            [pd.DataFrame([df_shifted.iloc[-1]]) for _ in range(shift_amount)],
            ignore_index=True
        )
        df_shifted = pd.concat([df_shifted, pad_rows], ignore_index=True)
        
        return df_shifted.iloc[:self.seq_length]
    
    def shift_right(self, df: pd.DataFrame, shift_amount: int = 10) -> pd.DataFrame:
        """
        Shift fault event RIGHT by moving it later in time.
        
        Timeline visualization:
        Original: [Pre-fault data ... 200-401: FAULT ... Post-fault]
        After shift_right(10):
               [Pre-fault PADDED ... 210-401: FAULT ... Post-fault TRIMMED]
        
        Args:
            df: Input DataFrame with 401 rows
            shift_amount: Number of rows to shift right (default 10)
            
        Returns:
            New DataFrame with fault event shifted later
        """
        if len(df) != self.seq_length:
            raise ValueError(f"Input must have {self.seq_length} rows, got {len(df)}")
        
        if shift_amount <= 0:
            return df.copy()
            
        # Pad beginning with first row (shifts fault event later in time)
        pad_rows = pd.concat(
            [pd.DataFrame([df.iloc[0]]) for _ in range(shift_amount)],
            ignore_index=True
        )
        df_shifted = pd.concat([pad_rows, df], ignore_index=True)
        
        # Take first 401 rows
        return df_shifted.iloc[:self.seq_length]


class GaussianNoiseAugmentation:
    """Add realistic Gaussian noise based on SNR levels."""
    
    # SNR levels (in dB) based on IEEE power system measurements
    # https://ieeexplore.ieee.org/document/10360822/ (SVD-Prony algorithms)
    SNR_LEVELS_DB = {
        'very_noisy': 1,      # Highly noisy field measurement
        'noisy': 5,           # Industrial environment with interference
        'moderate': 10,       # Typical power system measurement
        'clean': 20,          # Good signal quality
        'very_clean': 40,     # Lab conditions
    }
    
    def __init__(self, seq_length: int = 401, num_channels: int = 6):
        """
        Args:
            seq_length: Number of rows (time steps)
            num_channels: Number of signal channels (default 6)
        """
        self.seq_length = seq_length
        self.num_channels = num_channels
        
    def calculate_noise_std(self, signal: np.ndarray, snr_db: float) -> float:
        """
        Calculate Gaussian noise standard deviation from SNR_dB.
        
        Formula: SNR_dB = 20 * log10(signal_rms / noise_std)
        Rearranged: noise_std = signal_rms / 10^(SNR_dB / 20)
        
        Args:
            signal: Input signal (shape can be any)
            snr_db: Target SNR in decibels
            
        Returns:
            Standard deviation of Gaussian noise to add
        """
        signal_rms = np.sqrt(np.mean(signal ** 2))
        noise_std = signal_rms / (10 ** (snr_db / 20))
        return noise_std
    
    def add_gaussian_noise(
        self,
        df: pd.DataFrame,
        snr_db: float,
        random_state: int = None
    ) -> pd.DataFrame:
        """
        Add Gaussian white noise to all signal channels.
        
        Args:
            df: Input DataFrame (401 rows × 7 columns)
                [distance_km, CT1IA, CT1IB, CT1IC, BUS1UA, BUS1UB, BUS1UC]
            snr_db: Target SNR in decibels
            random_state: Random seed for reproducibility
            
        Returns:
            DataFrame with added Gaussian noise
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        df_noisy = df.copy()
        
        # Signal columns (skip 'distance_km')
        signal_columns = df.columns[1:].tolist()  # [CT1IA, CT1IB, ...]
        
        for col in signal_columns:
            signal = df_noisy[col].values.astype(np.float32)
            noise_std = self.calculate_noise_std(signal, snr_db)
            
            # Add zero-mean Gaussian noise
            noise = np.random.normal(0, noise_std, len(signal)).astype(np.float32)
            df_noisy[col] = signal + noise
        
        return df_noisy
    
    @staticmethod
    def snr_db_to_label(snr_db: float) -> str:
        """Convert SNR_dB to human-readable label."""
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
    
    # Configuration
    TIME_SHIFTS = [10, 20, 30, 40, 50]  # 5 shift variations (in rows)
    SNR_LEVELS = [1, 5, 10, 20, 40]     # 5 SNR levels (in dB)
    
    def __init__(self, seq_length: int = 401, num_channels: int = 6, seed: int = 42):
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
        """
        Augment a single CSV file.
        
        Creates:
        - 5 time-shifted versions (no noise)
        - 5 time-shifted × 5 noise-levels = 25 noisy versions per time-shift
        - Total: 30 new samples per original
        
        Args:
            csv_path: Path to original CSV file
            output_dir: Where to save augmented files
            file_stem: Base name for output files (auto-detect from csv_path if None)
            
        Returns:
            List of created file paths
        """
        if file_stem is None:
            file_stem = Path(csv_path).stem
        
        created_files = []
        df_original = pd.read_csv(csv_path)
        
        # Time shift augmentation (5 variations)
        for i, shift_amount in enumerate(self.TIME_SHIFTS):
            # Shift LEFT
            df_shifted_left = self.time_shift.shift_left(df_original, shift_amount)
            
            # Shift RIGHT
            df_shifted_right = self.time_shift.shift_right(df_original, shift_amount)
            
            # Gaussian noise augmentation on LEFT-shifted (5 SNR levels)
            for j, snr_db in enumerate(self.SNR_LEVELS):
                df_noisy_left = self.gaussian_noise.add_gaussian_noise(
                    df_shifted_left,
                    snr_db,
                    random_state=self.seed + i * 100 + j
                )
                
                snr_label = self.gaussian_noise.snr_db_to_label(snr_db)
                fname = f"{file_stem}_shift_left_{shift_amount}px_snr_{snr_db}dB_{snr_label}.csv"
                fpath = os.path.join(output_dir, fname)
                df_noisy_left.to_csv(fpath, index=False)
                created_files.append(fpath)
            
            # Gaussian noise augmentation on RIGHT-shifted (5 SNR levels)
            for j, snr_db in enumerate(self.SNR_LEVELS):
                df_noisy_right = self.gaussian_noise.add_gaussian_noise(
                    df_shifted_right,
                    snr_db,
                    random_state=self.seed + i * 100 + j + 50
                )
                
                snr_label = self.gaussian_noise.snr_db_to_label(snr_db)
                fname = f"{file_stem}_shift_right_{shift_amount}px_snr_{snr_db}dB_{snr_label}.csv"
                fpath = os.path.join(output_dir, fname)
                df_noisy_right.to_csv(fpath, index=False)
                created_files.append(fpath)
        
        return created_files
    
    def augment_dataset(self, input_dir: str, output_dir: str) -> dict:
        """
        Augment all CSV files in input directory.
        
        Args:
            input_dir: Directory containing original CSV files
            output_dir: Directory to save augmented files (created if not exists)
            
        Returns:
            Dictionary with augmentation statistics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {input_dir}")
        
        print(f"\n{'='*70}")
        print(f"Data Augmentation Pipeline")
        print(f"{'='*70}")
        print(f"Input directory : {input_dir}")
        print(f"Output directory: {output_dir}")
        print(f"\nOriginal samples: {len(csv_files)}")
        print(f"Time shifts     : {len(self.TIME_SHIFTS)} (left & right = {2*len(self.TIME_SHIFTS)} total)")
        print(f"SNR levels      : {len(self.SNR_LEVELS)} {self.SNR_LEVELS}")
        print(f"Augmentations per original: {2 * len(self.TIME_SHIFTS) * len(self.SNR_LEVELS)}")
        print(f"Expected total samples: {len(csv_files)} × {2 * len(self.TIME_SHIFTS) * len(self.SNR_LEVELS)} = {len(csv_files) * 2 * len(self.TIME_SHIFTS) * len(self.SNR_LEVELS)}")
        print(f"{'='*70}\n")
        
        all_created = []
        for idx, csv_path in enumerate(csv_files, 1):
            fname = os.path.basename(csv_path)
            print(f"[{idx}/{len(csv_files)}] Processing {fname}...")
            
            created = self.augment_single_file(csv_path, output_dir)
            all_created.extend(created)
            
            print(f"      → Created {len(created)} augmented samples\n")
        
        stats = {
            'original_count': len(csv_files),
            'time_shifts': len(self.TIME_SHIFTS),
            'snr_levels': len(self.SNR_LEVELS),
            'augmentations_per_sample': 2 * len(self.TIME_SHIFTS) * len(self.SNR_LEVELS),
            'total_created': len(all_created),
            'expected_total': len(csv_files) * 2 * len(self.TIME_SHIFTS) * len(self.SNR_LEVELS),
        }
        
        print(f"{'='*70}")
        print(f"Augmentation Complete!")
        print(f"{'='*70}")
        print(f"Total created files: {stats['total_created']}")
        print(f"Time shifts: {self.TIME_SHIFTS} (left & right)")
        print(f"SNR levels (dB): {self.SNR_LEVELS}")
        print(f"{'='*70}\n")
        
        return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Augment fault oscillogram dataset with time shifts and Gaussian noise'
    )
    parser.add_argument(
        '--input',
        default='data/data_training',
        help='Input directory with original CSV files'
    )
    parser.add_argument(
        '--output',
        default='data/data_augmented',
        help='Output directory for augmented CSV files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    pipeline = AugmentationPipeline(seed=args.seed)
    stats = pipeline.augment_dataset(args.input, args.output)
