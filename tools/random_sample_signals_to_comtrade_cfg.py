#!/usr/bin/env python3
"""
random_sample_signals_to_comtrade_cfg.py
===========================================
Конвертирует таблицу-тензор `random_sample_signals.csv` (из output/thesis/runs/*)
в COMTRADE 1999 .cfg (ТОЛЬКО конфиг, без .dat).

Используется для анализа фазных величин в COMTRADE-плеерах, когда у вас есть
выборка, но нет исходного comtrade/cfg/dat.

Входной CSV ожидается в формате:
  time_idx,ch0,ch1,...,ch{N-1}

Ожидаемый тензор (после FaultDataset при SYMSEQ_ENABLED):
  ch0..ch2  -> IA, IB, IC
  ch3..ch5  -> |I1|, |I2|, |I0|    (игнорируем)
  ch6..ch8  -> UA, UB, UC
  ch9..ch11 -> |U1|, |U2|, |U0|    (игнорируем)

COMTRADE .cfg будет содержать 6 аналоговых каналов:
  Ia, Ib, Ic (A) и Ua, Ub, Uc (V)

Замечание:
- .cfg требует min/max по каналам; они вычисляются по данным CSV.
- .dat не генерируется (только cfg).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ChannelSpec:
    ch_id: str
    uu: str  # "A" or "V"
    values: np.ndarray  # shape (n_steps,)


def _write_cfg(
    *,
    out_path: Path,
    source_name: str,
    distance_km: float,
    fs_hz: float,
    channels: List[ChannelSpec],
) -> None:
    n_analog = len(channels)
    n_digital = 0
    n_total = n_analog + n_digital

    stamp = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")[:26]

    # COMTRADE 1999 cfg:
    # 1) station_name, rec_dev_id, rev_year
    lines: List[str] = []
    lines.append(f"SymSeq_{source_name},{distance_km:.2f}km,1999")

    # 2) TT,nA,nD
    lines.append(f"{n_total},{n_analog}A,{n_digital}D")

    # 3+) analog channels
    # format:
    # n,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS
    for idx, spec in enumerate(channels, start=1):
        v = spec.values
        vmin = float(np.min(v))
        vmax = float(np.max(v))

        lines.append(
            f"{idx},{spec.ch_id},,"
            f"{spec.uu},"
            f"{1.0:.9e},{0.0:.9e},"
            f"{0.0:.6f},"
            f"{vmin:.6f},{vmax:.6f},"
            f"{1.0:.6f},{1.0:.6f},"
            f"S"
        )

    # network frequency (f0) — в COMTRADE .cfg это поле
    # в нашем проекте обычно 50 Hz; если нужно иное — добавим аргумент.
    f0 = 50.0
    lines.append(f"{f0:.3f}")

    # time info
    # lines:
    # 1) nrates/ samp (we keep simple: 1)
    # 2) fs_out,endsamp
    # 3) timestamps first and last
    n_steps = int(channels[0].values.shape[0]) if channels else 0
    lines.append("1")
    lines.append(f"{fs_hz:.6f},{n_steps}")

    # timestamps: we don't know absolute; use stamp as both first and last
    lines.append(stamp)
    lines.append(stamp)

    # data file format
    lines.append("ASCII")   # irrelevant if no .dat, but valid cfg
    lines.append("1")       # timemult

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _infer_channels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      I (n_steps, 3) with columns Ia, Ib, Ic
      U (n_steps, 3) with columns Ua, Ub, Uc
    """
    # time_idx ignored
    ch_cols = [c for c in df.columns if c.startswith("ch")]
    if not ch_cols:
        raise ValueError("No ch* columns found in CSV.")

    # Ensure ordering ch0..ch{N-1}
    def _ch_key(name: str) -> int:
        return int(name.replace("ch", ""))

    ch_cols_sorted = sorted(ch_cols, key=_ch_key)
    n_ch = len(ch_cols_sorted)
    if n_ch < 9:
        raise ValueError(f"Expected at least 9 signal channels, got {n_ch}.")

    # Phase currents: ch0..ch2
    Ia = df["ch0"].to_numpy(dtype=float)
    Ib = df["ch1"].to_numpy(dtype=float)
    Ic = df["ch2"].to_numpy(dtype=float)

    # Phase voltages: ch6..ch8
    Ua = df["ch6"].to_numpy(dtype=float)
    Ub = df["ch7"].to_numpy(dtype=float)
    Uc = df["ch8"].to_numpy(dtype=float)

    I = np.column_stack([Ia, Ib, Ic])
    U = np.column_stack([Ua, Ub, Uc])
    return I, U


def main() -> None:
    p = argparse.ArgumentParser(description="random_sample_signals.csv -> COMTRADE .cfg only")
    p.add_argument("--csv", required=True, help="Path to random_sample_signals.csv")
    p.add_argument("--fs_hz", type=float, required=True, help="Sampling frequency (Hz) for COMTRADE cfg")
    p.add_argument("--distance_km", type=float, default=0.0, help="distance value to put into cfg header")
    p.add_argument("--out", default="output_comtrade_cfg", help="Output folder for cfg")
    p.add_argument("--stem", default=None, help="Output cfg filename stem (without extension)")
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))

    df = pd.read_csv(csv_path)
    I, U = _infer_channels(df)

    # build channel specs in COMTRADE order:
    # Ia, Ib, Ic, Ua, Ub, Uc
    ch_specs: List[ChannelSpec] = [
        ChannelSpec("Ia", "A", I[:, 0]),
        ChannelSpec("Ib", "A", I[:, 1]),
        ChannelSpec("Ic", "A", I[:, 2]),
        ChannelSpec("Ua", "V", U[:, 0]),
        ChannelSpec("Ub", "V", U[:, 1]),
        ChannelSpec("Uc", "V", U[:, 2]),
    ]

    if args.stem:
        stem = args.stem
    else:
        stem = csv_path.stem

    out_dir = Path(args.out)
    cfg_path = out_dir / f"{stem}.cfg"

    _write_cfg(
        out_path=cfg_path,
        source_name=csv_path.stem,
        distance_km=args.distance_km,
        fs_hz=args.fs_hz,
        channels=ch_specs,
    )

    print(f"[OK] Wrote COMTRADE cfg: {cfg_path}")


if __name__ == "__main__":
    main()
