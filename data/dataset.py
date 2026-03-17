"""
PyTorch Dataset for Short-Circuit Fault Oscillogram Data

Format:
    CSV with columns:
    - Column 0: Fault distance in kilometers
    - Columns 1-300: Instantaneous signal values (100ms pre-fault + 200ms fault)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os


class FaultDataset(Dataset):
    """
    PyTorch Dataset for short-circuit fault detection.
    
    Loads oscillogram data from CSV and provides normalized signal/distance pairs.
    """
    
    def __init__(self, csv_path, seq_length=300, normalize=True, 
                 scaler_signal=None, scaler_distance=None):
        """
        Args:
            csv_path (str): Path to oscillogram CSV file
            seq_length (int): Expected signal length (should be 300)
            normalize (bool): Whether to normalize data
            scaler_signal (StandardScaler): Pre-fitted scaler for signals (for test set)
            scaler_distance (MinMaxScaler): Pre-fitted scaler for distances (for test set)
        """
        
        # Check if file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV
        print(f"Loading data from {csv_path}...")
        self.data = pd.read_csv(csv_path)
        
        if self.data.shape[0] == 0:
            raise ValueError("CSV file is empty")
        
        # Extract distance (first column) and signals (remaining columns)
        self.distances = self.data.iloc[:, 0].values.astype(np.float32)
        self.signals = self.data.iloc[:, 1:].values.astype(np.float32)
        
        # Validate shape
        if self.signals.shape[1] != seq_length:
            raise ValueError(
                f"Expected {seq_length} signal columns, got {self.signals.shape[1]}"
            )
        
        self.seq_length = seq_length
        self.num_samples = len(self.distances)
        
        print(f"  Loaded {self.num_samples} samples")
        print(f"  Signal shape: {self.signals.shape}")
        print(f"  Distance range: [{self.distances.min():.2f}, {self.distances.max():.2f}] km")
        
        # ============ NORMALIZATION ============
        if normalize:
            # If scaler not provided, fit on this data
            if scaler_signal is None:
                self.scaler_signal = StandardScaler()
                self.signals = self.scaler_signal.fit_transform(self.signals)
            else:
                self.scaler_signal = scaler_signal
                self.signals = self.scaler_signal.transform(self.signals)
            
            # Normalize distances to [0, 1]
            if scaler_distance is None:
                self.scaler_distance = MinMaxScaler()
                self.distances = self.scaler_distance.fit_transform(
                    self.distances.reshape(-1, 1)
                ).flatten()
            else:
                self.scaler_distance = scaler_distance
                self.distances = self.scaler_distance.transform(
                    self.distances.reshape(-1, 1)
                ).flatten()
            
            print("  ✓ Data normalized")
        else:
            self.scaler_signal = None
            self.scaler_distance = None
    
    def __len__(self):
        """Return number of samples in dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get one sample.
        
        Args:
            idx: Sample index
        
        Returns:
            signal: (1, seq_length) - 1D signal reshaped for Conv1d
            distance: scalar - normalized distance
        """
        # Signal: (seq_length,) -> (1, seq_length) for Conv1d
        signal = torch.FloatTensor(self.signals[idx]).unsqueeze(0)
        
        # Distance: scalar
        distance = torch.FloatTensor([self.distances[idx]])
        
        return signal, distance
    
    def get_raw_sample(self, idx):
        """
        Get un-normalized sample for debugging.
        
        Args:
            idx: Sample index
        
        Returns:
            signal: Original signal values
            distance: Original distance in km
        """
        if self.scaler_signal is not None:
            signal = self.scaler_signal.inverse_transform([self.signals[idx]])[0]
            distance = self.scaler_distance.inverse_transform(
                [[self.distances[idx]]]
            )[0][0]
        else:
            signal = self.signals[idx]
            distance = self.distances[idx]
        
        return signal, distance


class DataLoaderFactory:
    """
    Factory for creating train/test DataLoaders with proper preprocessing.
    """
    
    @staticmethod
    def create_loaders(csv_path, cfg):
        """
        Create train and test DataLoaders.
        
        Args:
            csv_path (str): Path to CSV file
            cfg: Config object with BATCH_SIZE, TRAIN_SPLIT, etc.
        
        Returns:
            tuple: (train_loader, test_loader, scalers_dict)
        """
        from torch.utils.data import random_split, DataLoader
        
        # Load full dataset
        full_dataset = FaultDataset(
            csv_path,
            seq_length=cfg.SEQ_LENGTH,
            normalize=cfg.NORMALIZE_DATA
        )
        
        # Get scalers for later use
        scalers = {
            'signal': full_dataset.scaler_signal,
            'distance': full_dataset.scaler_distance
        }
        
        # Split into train/test
        train_size = int(cfg.TRAIN_SPLIT * len(full_dataset))
        test_size = len(full_dataset) - train_size
        
        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,  # Set > 0 if on Linux with multiple cores
            pin_memory=True if cfg.DEVICE == 'cuda' else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True if cfg.DEVICE == 'cuda' else False
        )
        
        print(f"\n✅ Data split:")
        print(f"   Train: {train_size} samples")
        print(f"   Test:  {test_size} samples")
        print(f"   Batch size: {cfg.BATCH_SIZE}")
        
        return train_loader, test_loader, scalers
