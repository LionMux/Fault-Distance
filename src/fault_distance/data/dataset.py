"""PyTorch Dataset for Short-Circuit Fault Oscillogram Data

Expected file layout:
    data/data_training/
        1A_0.5km.csv
        1A_1.0km.csv
        ...

Each CSV file represents ONE fault event (one training sample).
The data_training/ subfolder is intentionally separate from the data/ Python
package (which contains dataset.py, preprocessing.py, __init__.py) so that
CSV files and source files never get mixed together.

CSV format (rows = time steps, 7 columns):
    distance_km | CT1IA | CT1IB | CT1IC | S1)BUS1UA | S1)BUS1UB | S1)BUS1UC
    0.5         | ...   | ...   | ...   | ...       | ...       | ...
    0.5         | ...   | ...   | ...   | ...       | ...       | ...
    ...

Signal channels:
    0: CT1IA   - Phase A current [A] (small magnitude ~0.07-260)
    1: CT1IB   - Phase B current [A]
    2: CT1IC   - Phase C current [A]
    3: BUS1UA  - Phase A voltage [kV] (large magnitude ~100)
    4: BUS1UB  - Phase B voltage [kV]
    5: BUS1UC  - Phase C voltage [kV]

Outputs:
    signal tensor : (NUM_CHANNELS, SEQ_LENGTH) e.g. (6, 400)
    distance      : scalar [km]
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Tuple, Dict

# Known column names in the CSV
DISTANCE_COL = 'distance_km'
SIGNAL_COLS = ['CT1IA', 'CT1IB', 'CT1IC', 'S1)BUS1UA', 'S1)BUS1UB', 'S1)BUS1UC']


class FaultDataset(Dataset):
    """
    Loads a directory of oscillogram CSV files.
    Each file => one (signal, distance) sample.
    signal shape : (NUM_CHANNELS, SEQ_LENGTH) ready for Conv1d
    distance     : scalar float32 (optionally normalized to [0,1] or p.u.)
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 400,
        num_channels: int = 6,
        normalize: bool = True,
        signal_scalers: Optional[list] = None,
        distance_scaler: Optional[MinMaxScaler] = None,
        cfg=None,
    ):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir} "
                f"Create it and place your oscillogram CSV files inside: "
                f"  {data_dir}/1A_0.5km.csv "
                f"  {data_dir}/1A_1.0km.csv "
                f"  ..."
            )

        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir} "
                "Expected files like: 1A_0.5km.csv, 1B_2.0km.csv, ..."
            )

        print(f"Loading {len(csv_files)} oscillogram files from {data_dir} ...")

        signals_list: list = []
        distances_list: list = []
        skipped = 0

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath)
                missing = [c for c in [DISTANCE_COL] + SIGNAL_COLS if c not in df.columns]
                if missing:
                    print(f"  [SKIP] {os.path.basename(fpath)} - missing columns: {missing}")
                    skipped += 1
                    continue

                distance = float(df[DISTANCE_COL].iloc[0])
                sig = df[SIGNAL_COLS].values.astype(np.float32)

                T = sig.shape[0]
                if T < seq_length:
                    pad = np.zeros((seq_length - T, sig.shape[1]), dtype=np.float32)
                    sig = np.vstack([sig, pad])
                elif T > seq_length:
                    sig = sig[:seq_length, :]

                sig = sig.T
                signals_list.append(sig)
                distances_list.append(distance)

            except Exception as e:
                print(f"  [SKIP] {os.path.basename(fpath)} - error: {e}")
                skipped += 1

        if len(signals_list) == 0:
            raise ValueError("No valid samples loaded. Check your CSV files.")

        print(f"  Loaded {len(signals_list)} samples ({skipped} skipped)")

        self.seq_length = seq_length
        self.num_channels = num_channels
        self.num_samples = len(signals_list)
        self.cfg = cfg

        self.signals = np.stack(signals_list, axis=0)
        self.distances = np.array(distances_list, dtype=np.float32)

        print(f"  Signal tensor shape : {self.signals.shape}")
        print(f"  Distance range      : [{self.distances.min():.2f}, {self.distances.max():.2f}] km")

        # ============ PREPROCESSING ============
        if cfg and getattr(cfg, 'BUTTERWORTH_ENABLED', False):
            from .preprocessing import apply_butterworth_filter
            print(f"  Applying Butterworth {cfg.BUTTERWORTH_TYPE} filter (cutoff={cfg.BUTTERWORTH_CUTOFF} Hz)...")
            self.signals = apply_butterworth_filter(self.signals, cfg)

        # ============ NORMALIZATION ============
        if normalize:
            norm_mode = getattr(cfg, 'NORMALIZATION_MODE', 'standard') if cfg else 'standard'

            if norm_mode == 'pu':
                print("  Applying physical per-unit (p.u.) normalization...")
                if not cfg:
                    raise ValueError("cfg must be provided for p.u. normalization mode")

                Unom_kv = cfg.LINE_UNOM_KV
                L_km = cfg.LINE_L_KM
                r1 = cfg.LINE_R1_OHM_KM
                x1 = cfg.LINE_X1_OHM_KM

                Z1_total = ((r1 * L_km)**2 + (x1 * L_km)**2)**0.5
                Ubase_V = (Unom_kv * 1000) / (3**0.5)
                Ibase_A = Ubase_V / Z1_total

                print(f"  Unom = {Unom_kv} kV")
                print(f"  L    = {L_km} km")
                print(f"  Z1_total = {Z1_total:.2f} Ohm")
                print(f"  Ubase = {Ubase_V:.1f} V")
                print(f"  Ibase = {Ibase_A:.1f} A")

                self.signals[:, 0:3, :] /= Ibase_A
                self.signals[:, 3:6, :] /= Unom_kv
                self.distances /= L_km

                self.signal_scalers = None
                self.distance_scaler = None
                print("  p.u. normalization complete")

            else:
                if signal_scalers is None:
                    self.signal_scalers = []
                    for ch in range(self.signals.shape[1]):
                        scaler = StandardScaler()
                        flat = self.signals[:, ch, :].reshape(-1, 1)
                        scaler.fit(flat)
                        self.signals[:, ch, :] = scaler.transform(flat).reshape(
                            self.num_samples, self.seq_length
                        )
                        self.signal_scalers.append(scaler)
                else:
                    self.signal_scalers = signal_scalers
                    for ch, scaler in enumerate(signal_scalers):
                        flat = self.signals[:, ch, :].reshape(-1, 1)
                        self.signals[:, ch, :] = scaler.transform(flat).reshape(
                            self.num_samples, self.seq_length
                        )

                if distance_scaler is None:
                    self.distance_scaler = MinMaxScaler()
                    self.distances = self.distance_scaler.fit_transform(
                        self.distances.reshape(-1, 1)
                    ).flatten()
                else:
                    self.distance_scaler = distance_scaler
                    self.distances = self.distance_scaler.transform(
                        self.distances.reshape(-1, 1)
                    ).flatten()

                print("  Per-channel normalization applied (standard mode)")
        else:
            self.signal_scalers = None
            self.distance_scaler = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = torch.from_numpy(self.signals[idx])
        distance = torch.tensor([self.distances[idx]], dtype=torch.float32)
        return signal, distance

    def inverse_transform_distance(self, normalized: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to km."""
        if self.distance_scaler is not None:
            return self.distance_scaler.inverse_transform(
                normalized.reshape(-1, 1)
            ).flatten()
        return normalized


class DataLoaderFactory:
    """Factory for creating train/test DataLoaders with proper preprocessing."""

    @staticmethod
    def create_loaders(
        data_dir: str,
        cfg,
    ) -> Tuple[DataLoader, DataLoader, Dict]:
        full_dataset = FaultDataset(
            data_dir=data_dir,
            seq_length=cfg.SEQ_LENGTH,
            num_channels=cfg.NUM_CHANNELS,
            normalize=cfg.NORMALIZE_DATA,
            cfg=cfg,
        )

        scalers = {
            'signal': full_dataset.signal_scalers,
            'distance': full_dataset.distance_scaler,
        }

        train_size = int(cfg.TRAIN_SPLIT * len(full_dataset))
        test_size = len(full_dataset) - train_size

        train_dataset, test_dataset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(cfg.SEED),
        )

        pin = cfg.DEVICE == 'cuda'
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=pin
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=pin
        )

        print(f"  Train : {train_size} samples")
        print(f"  Test  : {test_size} samples")
        print(f"  Batch : {cfg.BATCH_SIZE}")

        return train_loader, test_loader, scalers
