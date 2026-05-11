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

CSV format (rows = time steps, columns):
    distance_km | [fs_hz] | <currents> | <voltages> | <others>

    distance_km — target label (constant per file, first column)
    fs_hz       — sampling frequency in Hz written by comtrade_to_csv.py
                  (optional; if absent, cfg.SAMPLING_FREQ_HZ is used as
                   fallback so old CSV files without this column still work)

Signal channels (6 expected by default):
    0: CT1IA    - Phase A current [A]
    1: CT1IB    - Phase B current [A]
    2: CT1IC    - Phase C current [A]
    3: BUS1UA   - Phase A voltage [kV]
    4: BUS1UB   - Phase B voltage [kV]
    5: BUS1UC   - Phase C voltage [kV]

When cfg.SYMSEQ_ENABLED is True the dataset appends 6 phasor-magnitude
channels (|I1|,|I2|,|I0|,|U1|,|U2|,|U0|) giving a 12-channel tensor:
    0-2 : IA, IB, IC            (time-domain)
    3-5 : |I1|, |I2|, |I0|      (phasor magnitudes, constant over time)
    6-8 : UA, UB, UC            (time-domain)
    9-11: |U1|, |U2|, |U0|      (phasor magnitudes, constant over time)

This matches the notebook pipeline where the final tensor contains both
phase and symmetrical-component channels, but here the symmetrical
components are obtained via FFT-based phasor estimation (as required by
the project specification) rather than instantaneous Fortescue.

Outputs:
    signal tensor : (NUM_CHANNELS, SEQ_LENGTH) e.g. (12, 400)
    distance      : scalar [km]
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Optional, Tuple, Dict, List

# Known column names in the CSV
DISTANCE_COL = 'distance_km'
FS_COL        = 'fs_hz'   # written by comtrade_to_csv.py; optional in old files
SIGNAL_COLS = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']

# --- notebook-matching channel order for the 12-channel tensor ---
# Phase channels (0-2, 6-8) come first; phasor channels (3-5, 9-11) follow.
PHASE_CURRENT_IDX = [0, 1, 2]
PHASE_VOLTAGE_IDX = [6, 7, 8]
SYMSEQ_CURRENT_IDX = [3, 4, 5]
SYMSEQ_VOLTAGE_IDX = [9, 10, 11]


class FaultDataset(Dataset):
    """
    Loads a directory of oscillogram CSV files.
    Each file => one (signal, distance) sample.
    signal shape : (NUM_CHANNELS, SEQ_LENGTH) ready for Conv1d
    distance     : scalar float32 (optionally normalized to [0,1] or p.u.)

    Fault-inception (t0) algorithm
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    When cfg.T0_ENABLED is True the dataset automatically detects the
    fault-inception moment t0 for every file individually, using the
    sampling frequency stored in the CSV column ``fs_hz`` (written by
    comtrade_to_csv.py).  If a file does not have the ``fs_hz`` column
    (legacy CSV), cfg.SAMPLING_FREQ_HZ is used as a fallback so that
    old files continue to work without any changes.
    """

    def __init__(
        self,
        data_dir: str,
        seq_length: int = 400,
        num_channels: int = 6,
        normalize: bool = True,
        signal_scalers: Optional[list] = None,   # list of NUM_CHANNELS fitted StandardScalers
        distance_scaler: Optional[MinMaxScaler] = None,
        cfg=None,                                # Config object (optional)
    ):
        """Create dataset from a folder of CSV oscillograms.

        Args:
            data_dir       : Folder containing *.csv oscillogram files.
            seq_length     : Number of time steps expected in each file
                              (also used as target length for t0-cropped
                               windows when enabled).
            num_channels   : Number of signal channels (default 6; becomes 12
                              when symseq is enabled).
            normalize      : Apply normalization (mode determined by cfg).
            signal_scalers : Pre-fitted scalers (for test set).
            distance_scaler: Pre-fitted MinMaxScaler for distance.
            cfg            : Config object (needed for p.u. normalization
                              mode and advanced preprocessing).
        """
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}\n"
                f"Create it and place your oscillogram CSV files inside:\n"
                f"  {data_dir}/1A_0.5km.csv\n"
                f"  {data_dir}/1A_1.0km.csv\n"
                f"  ..."
            )

        csv_files = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found in {data_dir}\n"
                "Expected files like: 1A_0.5km.csv, 1B_2.0km.csv, ..."
            )

        print(f"Loading {len(csv_files)} oscillogram files from {data_dir} ...")

        signals_list: list = []  # each: (NUM_CHANNELS, seq_length)
        distances_list: list = []
        skipped = 0

        # Optional fault-inception configuration from cfg
        self.cfg = cfg
        enable_t0 = bool(getattr(cfg, 'T0_ENABLED', False)) if cfg is not None else False
        enable_symseq = bool(getattr(cfg, 'SYMSEQ_ENABLED', False)) if cfg is not None else False
        enable_remove_dc = bool(getattr(cfg, 'REMOVE_DC_ENABLED', False)) if cfg is not None else False

        # Fallback fs when the CSV has no fs_hz column (legacy files)
        _cfg_fs_fallback = (
            float(getattr(cfg, 'SAMPLING_FREQ_HZ', 2000.0))
            if cfg is not None
            else 2000.0
        )
        _cfg_mains_hz = float(getattr(cfg, 'MAINS_FREQ_HZ', 50.0)) if cfg is not None else 50.0

        if enable_t0:
            from .fault_inception import FaultInceptionParams, detect_t0_and_crop
            _t0_pre_ms  = float(getattr(cfg, 'T0_PRE_MS',  50.0))
            _t0_post_ms = float(getattr(cfg, 'T0_POST_MS', 70.0))
            _t0_eta_I   = float(getattr(cfg, 'T0_ETA_I',   0.5))
            _t0_eta_U   = float(getattr(cfg, 'T0_ETA_U',   0.85))
        else:
            FaultInceptionParams = None  # type: ignore[assignment]
            detect_t0_and_crop   = None  # type: ignore[assignment]

        # Lazy imports for optional features
        if enable_remove_dc:
            from .preprocessing import center_by_prehistory, remove_dc_period
        if enable_symseq:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

        norm_mode_pre = getattr(cfg, 'NORMALIZATION_MODE', 'standard') if cfg else 'standard'

        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath)

                # ---- validate required columns ----
                missing = [c for c in [DISTANCE_COL] + SIGNAL_COLS if c not in df.columns]
                if missing:
                    print(f" [SKIP] {os.path.basename(fpath)} - missing columns: {missing}")
                    skipped += 1
                    continue

                # ---- target: constant for the whole file ----
                distance = float(df[DISTANCE_COL].iloc[0])

                # ---- sampling frequency: from CSV column or cfg fallback ----
                if FS_COL in df.columns:
                    file_fs_hz = float(df[FS_COL].iloc[0])
                else:
                    file_fs_hz = _cfg_fs_fallback
                    if enable_t0:
                        print(
                            f" [INFO] {os.path.basename(fpath)} has no '{FS_COL}' column; "
                            f"using cfg fallback fs={file_fs_hz:.1f} Hz"
                        )

                # ---- raw signals: shape (T, num_channels) ----
                sig = df[SIGNAL_COLS].values.astype(np.float32)

                # ---- notebook stage 2: centering + DC-period removal ----
                if enable_remove_dc:
                    sig = center_by_prehistory(sig.T, fs=file_fs_hz, pre_ms=20.0).T
                    sig = remove_dc_period(sig.T, fs=file_fs_hz, f_net=_cfg_mains_hz).T

                # ---- optional fault-inception detection and cropping ----
                if enable_t0 and detect_t0_and_crop is not None:
                    params = FaultInceptionParams(
                        fs_hz=file_fs_hz,
                        mains_hz=_cfg_mains_hz,
                        pre_fault_ms=_t0_pre_ms,
                        post_fault_ms=_t0_post_ms,
                        eta_I=_t0_eta_I,
                        eta_U=_t0_eta_U,
                    )
                    try:
                        sig, _t0_local = detect_t0_and_crop(
                            sig,
                            params,
                            current_channel_indices=(0, 1, 2),
                            voltage_channel_indices=(3, 4, 5),
                        )
                    except Exception as e:  # pragma: no cover
                        print(
                            f" [WARN] t0 detection failed for "
                            f"{os.path.basename(fpath)}: {e}. "
                            "Falling back to simple pad/trim."
                        )

                # IMPORTANT:
                # После t0-crop мы больше НЕ делаем pad/trim до SEQ_LENGTH
                # (иначе появляется обрезанное/ресэмплированное окно и/или нули).
                # Длина должна совпадать за счёт параметров t0_pre_ms/t0_post_ms.
                #
                # В fallback-сценариях длина может быть разной — это можно
                # увидеть как inconsistency, поэтому padding отключаем полностью.

                # Convert to (num_channels, seq_length)  -> currently 6 channels
                sig = sig.T   # (6, seq_length)

                # ---- optional symmetrical components (notebook stage 5, adapted) ----
                # Requirement: if pu mode -> symseq must be computed from pu-values of currents/voltages.
                if enable_symseq:
                    if norm_mode_pre == 'pu':
                        Unom_kv = cfg.LINE_UNOM_KV
                        S_base_MVA = getattr(cfg, 'S_BASE_MVA', 100.0)
                        Ibase_A = (S_base_MVA * 1_000_000) / (3 ** 0.5 * Unom_kv * 1000)
                        # currents IA,IB,IC -> p.u.
                        sig[0:3, :] /= Ibase_A
                        # voltages UA,UB,UC -> p.u.
                        sig[3:6, :] /= Unom_kv

                    # Compute sliding-window sequence magnitudes (time-varying)
                    from .preprocessing import sliding_window_symseq
                    sym_ch = sliding_window_symseq(
                        sig, fs=file_fs_hz, f0=_cfg_mains_hz, window_cycles=1
                    )  # (6, seq_length)  -> [|I1|,|I2|,|I0|,|U1|,|U2|,|U0|]

                    # Notebook channel order:
                    # IA, IB, IC, |I1|, |I2|, |I0|, UA, UB, UC, |U1|, |U2|, |U0|
                    sig = np.vstack([
                        sig[0:3, :],      # IA, IB, IC (possibly pu)
                        sym_ch[0:3, :],   # |I1|, |I2|, |I0|  (sliding-window)
                        sig[3:6, :],      # UA, UB, UC (possibly pu)
                        sym_ch[3:6, :],   # |U1|, |U2|, |U0|  (sliding-window)
                    ])   # (12, seq_length)
                    num_channels = 12

                signals_list.append(sig)
                distances_list.append(distance)

            except Exception as e:
                print(f" [SKIP] {os.path.basename(fpath)} - error: {e}")
                skipped += 1

        if len(signals_list) == 0:
            raise ValueError("No valid samples loaded. Check your CSV files.")

        print(f" Loaded {len(signals_list)} samples ({skipped} skipped)")

        self.seq_length = seq_length
        self.num_channels = num_channels
        self.num_samples = len(signals_list)

        # signals: (N, num_channels, seq_length) / distances: (N,)
        self.signals   = np.stack(signals_list, axis=0)   # (N, C, T)
        self.distances = np.array(distances_list, dtype=np.float32)  # (N,)

        print(f" Signal tensor shape : {self.signals.shape}")
        print(f" Distance range      : [{self.distances.min():.2f}, {self.distances.max():.2f}] km")

        # ============ NORMALIZATION ============
        if normalize:
            norm_mode = getattr(cfg, 'NORMALIZATION_MODE', 'standard') if cfg else 'standard'

            if norm_mode == 'pu':
                print(" Applying physical per-unit (p.u.) normalization...")
                if not cfg:
                    raise ValueError("cfg must be provided for p.u. normalization mode")

                L_km = cfg.LINE_L_KM
                # In pu mode we already converted phase quantities to p.u. inside the per-file
                # preprocessing step (so symseq is computed from p.u. values). Here we only
                # normalize the distance target.
                self.distances /= L_km  # distance [km] -> distance_pu

                self.signal_scalers = None
                self.distance_scaler = None
                print(" p.u. distance normalization complete")

            else:
                # Per-channel StandardScaler
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

                print(" Per-channel normalization applied (standard mode)")
        else:
            self.signal_scalers  = None
            self.distance_scaler = None

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (signal, distance) pair for given index."""
        signal   = torch.from_numpy(self.signals[idx])  # (C, T)
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
        """Create train and test DataLoaders.

        Args:
            data_dir : Directory with oscillogram CSV files.
            cfg      : Config object.

        Returns:
            (train_loader, test_loader, scalers_dict)
        """

        full_dataset = FaultDataset(
            data_dir=data_dir,
            seq_length=cfg.SEQ_LENGTH,
            num_channels=cfg.NUM_CHANNELS,
            normalize=cfg.NORMALIZE_DATA,
            cfg=cfg,
        )

        scalers = {
            'signal':   full_dataset.signal_scalers,
            'distance': full_dataset.distance_scaler,
        }

        train_size = int(cfg.TRAIN_SPLIT * len(full_dataset))
        test_size  = len(full_dataset) - train_size

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
            pin_memory=pin,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )

        print(f"\n Train : {train_size} samples")
        print(f" Test  : {test_size} samples")
        print(f" Batch : {cfg.BATCH_SIZE}")

        return train_loader, test_loader, scalers
