#!/usr/bin/env python3
"""
symseq_to_comtrade.py
=====================
Экспорт симметричных составляющих (скользящее окно) в формат COMTRADE
(.cfg + .dat) для просмотра в любом осциллографическом ПО
(OMICRON Transview, PowerDB, DIgSILENT, RTDS и др.).

Архитектура модуля
------------------
CSV (сигналы)
    └─► sliding_symseq()        — скользящий расчёт I0/I1/I2, U0/U1/U2
            └─► SymSeqResult    — dataclass-контейнер результатов
                    └─► ComtradeExporter.export()  — запись .cfg / .dat

Расширяемость
-------------
* Добавить новый канал   → добавить поле в SymSeqResult + строку в
                           ComtradeExporter._channel_defs()
* Изменить формат .dat   → переопределить ComtradeExporter._write_dat()
* Экспорт батча файлов   → export_batch() внизу модуля

Зависимости: только numpy, pandas (уже в requirements.txt проекта).

Запуск:
    python tools/symseq_to_comtrade.py
    python tools/symseq_to_comtrade.py --csv data/data_training/2AB_10km.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── путь к пакетам проекта ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from fault_distance.utils.column_detector import (
    detect_signal_columns,
    detect_distance_column,
)
from symseq.power_systems import symseq_from_waveforms

# ── константы ─────────────────────────────────────────────────────────────────
DEFAULT_CSV        = "data/data_training/2AB_10km.csv"
DEFAULT_F0         = 50.0     # Гц
DEFAULT_FS_FALLBACK = 1000.0  # Гц — если в CSV нет fs_hz
OUTPUT_DIR         = Path(__file__).parent / "output_tools"


# ═══════════════════════════════════════════════════════════════════════════════
# Контейнер результатов
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SymSeqResult:
    """
    Результат скользящего расчёта симметричных составляющих.

    Все массивы имеют длину n_steps = N - win + 1.
    Временна́я ось t_ms центрирована по середине каждого окна.
    """
    source_file : str
    distance_km : float
    fs          : float          # Гц
    f0          : float          # Гц
    win         : int            # отсчётов за 1 период
    n_steps     : int

    t_ms   : np.ndarray          # (n_steps,) — центры окон, мс

    # Амплитуды, А
    I0_mag : np.ndarray
    I1_mag : np.ndarray
    I2_mag : np.ndarray

    # Углы, рад
    I0_ang : np.ndarray = field(repr=False)
    I1_ang : np.ndarray = field(repr=False)
    I2_ang : np.ndarray = field(repr=False)

    # Амплитуды, В (исходные кВ → В для COMTRADE)
    U0_mag : np.ndarray = field(repr=False)
    U1_mag : np.ndarray = field(repr=False)
    U2_mag : np.ndarray = field(repr=False)

    # Углы, рад
    U0_ang : np.ndarray = field(repr=False)
    U1_ang : np.ndarray = field(repr=False)
    U2_ang : np.ndarray = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Скользящий расчёт
# ═══════════════════════════════════════════════════════════════════════════════

def sliding_symseq(
    csv_path : str | Path,
    f0       : float = DEFAULT_F0,
    fs_fallback: float = DEFAULT_FS_FALLBACK,
) -> SymSeqResult:
    """
    Загружает CSV и вычисляет симметричные составляющие скользящим окном.

    Окно = ровно 1 период (win = round(fs / f0)) → прямоугольное окно
    корректно, спектральной утечки нет.

    Parameters
    ----------
    csv_path    : путь к CSV файлу проекта
    f0          : частота сети, Гц
    fs_fallback : fs если колонки fs_hz нет в CSV

    Returns
    -------
    SymSeqResult
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # ── частота дискретизации ─────────────────────────────────────────────────
    if "fs_hz" in df.columns:
        fs = float(df["fs_hz"].iloc[0])
    else:
        fs = fs_fallback

    win = int(round(fs / f0))

    # ── колонки ───────────────────────────────────────────────────────────────
    dist_col = detect_distance_column(list(df.columns))
    col_map  = detect_signal_columns(list(df.columns), distance_col=dist_col)
    distance_km = float(df[dist_col].iloc[0])

    sig = df[
        [col_map["Ia"], col_map["Ib"], col_map["Ic"],
         col_map["Ua"], col_map["Ub"], col_map["Uc"]]
    ].values.astype(float)                              # (N, 6)

    N = len(sig)
    if N < win:
        raise ValueError(
            f"{csv_path.name}: строк {N} < размер окна {win}. "
            "Проверьте fs и f0."
        )

    n_steps = N - win + 1

    # ── выходные буферы ───────────────────────────────────────────────────────
    buf = {k: np.zeros(n_steps) for k in
           ("I0m","I1m","I2m","I0a","I1a","I2a",
            "U0m","U1m","U2m","U0a","U1a","U2a")}

    # ── скользящий цикл ───────────────────────────────────────────────────────
    for i in range(n_steps):
        sl = slice(i, i + win)

        ri = symseq_from_waveforms(
            sig[sl, 0], sig[sl, 1], sig[sl, 2],
            fs=fs, f0=f0, window=False,
        )
        ru = symseq_from_waveforms(
            sig[sl, 3], sig[sl, 4], sig[sl, 5],
            fs=fs, f0=f0, window=False,
        )

        buf["I0m"][i] = ri["X0_mag"]; buf["I0a"][i] = ri["X0_ang"]
        buf["I1m"][i] = ri["X1_mag"]; buf["I1a"][i] = ri["X1_ang"]
        buf["I2m"][i] = ri["X2_mag"]; buf["I2a"][i] = ri["X2_ang"]

        buf["U0m"][i] = ru["X0_mag"]; buf["U0a"][i] = ru["X0_ang"]
        buf["U1m"][i] = ru["X1_mag"]; buf["U1a"][i] = ru["X1_ang"]
        buf["U2m"][i] = ru["X2_mag"]; buf["U2a"][i] = ru["X2_ang"]

    t_ms = (np.arange(n_steps) + win / 2) / fs * 1000.0

    return SymSeqResult(
        source_file = csv_path.name,
        distance_km = distance_km,
        fs          = fs,
        f0          = f0,
        win         = win,
        n_steps     = n_steps,
        t_ms        = t_ms,
        I0_mag = buf["I0m"], I1_mag = buf["I1m"], I2_mag = buf["I2m"],
        I0_ang = buf["I0a"], I1_ang = buf["I1a"], I2_ang = buf["I2a"],
        U0_mag = buf["U0m"], U1_mag = buf["U1m"], U2_mag = buf["U2m"],
        U0_ang = buf["U0a"], U1_ang = buf["U1a"], U2_ang = buf["U2a"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMTRADE-экспортёр
# ═══════════════════════════════════════════════════════════════════════════════

class ComtradeExporter:
    """
    Записывает SymSeqResult в пару файлов COMTRADE 1999:
        <stem>.cfg  — конфигурационный файл (текстовый)
        <stem>.dat  — бинарные данные (формат binary32)

    Каналы, экспортируемые по умолчанию (расширяемо через _channel_defs):
        I1_mag, I2_mag, I0_mag  — амплитуды токов последовательностей, А
        I1_ang, I2_ang, I0_ang  — углы токов, градусы
        U1_mag, U2_mag, U0_mag  — амплитуды напряжений, В
        U1_ang, U2_ang, U0_ang  — углы напряжений, градусы

    Формат COMTRADE 1999 (IEEE Std C37.111-1999):
        - .cfg: ASCII, разделитель запятая
        - .dat: бинарный, sample_number(uint32) + timestamp(uint32) +
                N×int16 отсчётов
    """

    # Описание канала: (attr_name, ch_id, ph, ccbm, uu, a, b, skew, min, max, primary, secondary, PS)
    # a  — множитель (физ. значение = a * raw + b)
    # min/max — диапазон int16: ±32767
    # primary/secondary — коэф. трансформации (1/1 для вычисленных каналов)
    _CURRENT_CHANNELS = [
        ("I1_mag", "I1_mag", "ABC", "A",   "A",  None, 0.0),
        ("I2_mag", "I2_mag", "ABC", "A",   "A",  None, 0.0),
        ("I0_mag", "I0_mag", "ABC", "A",   "A",  None, 0.0),
        ("I1_ang", "I1_ang", "ABC", "deg", "deg",None, 0.0),
        ("I2_ang", "I2_ang", "ABC", "deg", "deg",None, 0.0),
        ("I0_ang", "I0_ang", "ABC", "deg", "deg",None, 0.0),
    ]
    _VOLTAGE_CHANNELS = [
        ("U1_mag", "U1_mag", "ABC", "V",   "V",  None, 0.0),
        ("U2_mag", "U2_mag", "ABC", "V",   "V",  None, 0.0),
        ("U0_mag", "U0_mag", "ABC", "V",   "V",  None, 0.0),
        ("U1_ang", "U1_ang", "ABC", "deg", "deg",None, 0.0),
        ("U2_ang", "U2_ang", "ABC", "deg", "deg",None, 0.0),
        ("U0_ang", "U0_ang", "ABC", "deg", "deg",None, 0.0),
    ]

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── публичный интерфейс ───────────────────────────────────────────────────

    def export(self, result: SymSeqResult, stem: Optional[str] = None) -> tuple[Path, Path]:
        """
        Записывает .cfg и .dat.

        Parameters
        ----------
        result : SymSeqResult
        stem   : базовое имя файлов без расширения.
                 По умолчанию: <source_stem>_symseq

        Returns
        -------
        (cfg_path, dat_path)
        """
        if stem is None:
            stem = Path(result.source_file).stem + "_symseq"

        cfg_path = self.output_dir / f"{stem}.cfg"
        dat_path = self.output_dir / f"{stem}.dat"

        channels, data_matrix = self._build_channels(result)
        self._write_cfg(cfg_path, result, channels, dat_path.name)
        self._write_dat(dat_path, result, data_matrix)

        return cfg_path, dat_path

    # ── построение списка каналов и матрицы данных ────────────────────────────

    def _build_channels(self, r: SymSeqResult):
        """
        Собирает список каналов и матрицу сырых int16-значений.

        Для каждого канала:
            a = max_physical / 32767   (автомасштаб под int16)
            raw = round(physical / a)

        Returns
        -------
        channels    : list of dicts с полями для .cfg
        data_matrix : np.ndarray (n_samples, n_channels), dtype int16
        """
        all_defs = self._CURRENT_CHANNELS + self._VOLTAGE_CHANNELS
        channels   = []
        data_cols  = []

        for idx, (attr, ch_id, ph, ccbm, uu, _, b) in enumerate(all_defs, start=1):
            # получаем массив физических значений
            raw_data = getattr(r, attr).copy()

            # углы: радианы → градусы
            if uu == "deg":
                raw_data = np.degrees(raw_data)

            # автомасштаб
            max_abs = np.max(np.abs(raw_data))
            a = (max_abs / 32767.0) if max_abs > 0 else 1e-6

            int16_data = np.round(raw_data / a).astype(np.int16)

            channels.append({
                "n":       idx,
                "ch_id":   ch_id,
                "ph":      ph,
                "ccbm":    ccbm,
                "uu":      uu,
                "a":       a,
                "b":       b if b is not None else 0.0,
                "skew":    0.0,
                "min":    -32767,
                "max":     32767,
                "primary": 1.0,
                "secondary": 1.0,
                "PS":      "S",
            })
            data_cols.append(int16_data)

        data_matrix = np.column_stack(data_cols).astype(np.int16)  # (n_steps, n_ch)
        return channels, data_matrix

    # ── запись .cfg ───────────────────────────────────────────────────────────

    def _write_cfg(self, path: Path, r: SymSeqResult,
                   channels: list, dat_name: str) -> None:
        """
        Формат COMTRADE 1999 (.cfg)
        Спецификация: IEEE Std C37.111-1999, раздел 5.
        """
        n_analog  = len(channels)
        n_digital = 0
        n_total   = n_analog + n_digital

        # временно́й шаг скользящего окна = 1 отсчёт исходного сигнала
        dt_us = int(round(1_000_000.0 / r.fs))   # мкс
        fs_out = r.fs                              # та же fs что и у исходника

        stamp = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")[:26]

        lines = []

        # строка 1: station_name, rec_dev_id, rev_year
        lines.append(f"SymSeq_{r.source_file},{r.distance_km:.2f}km,1999")

        # строка 2: TT,nA,nD
        lines.append(f"{n_total},{n_analog}A,{n_digital}D")

        # строки каналов
        for ch in channels:
            lines.append(
                f"{ch['n']},{ch['ch_id']},{ch['ph']},{ch['ccbm']},{ch['uu']},"
                f"{ch['a']:.9e},{ch['b']:.9e},{ch['skew']:.6f},"
                f"{ch['min']},{ch['max']},"
                f"{ch['primary']:.6f},{ch['secondary']:.6f},{ch['PS']}"
            )

        # строка частоты сети
        lines.append(f"{r.f0:.3f}")

        # строки частот дискретизации: nrates, samp, endsamp
        lines.append("1")
        lines.append(f"{fs_out:.6f},{r.n_steps}")

        # временны́е метки первого и последнего отсчёта
        lines.append(stamp)   # first sample
        lines.append(stamp)   # last  sample (упрощение — одинаковые)

        # формат файла данных
        lines.append("BINARY32")

        # мультипликатор временно́го шага (timemult)
        lines.append("1")

        path.write_text("\n".join(lines) + "\n", encoding="ascii")

    # ── запись .dat ───────────────────────────────────────────────────────────

    def _write_dat(self, path: Path,
                   r: SymSeqResult,
                   data_matrix: np.ndarray) -> None:
        """
        COMTRADE BINARY32:
            каждый сэмпл = uint32 sample_number
                         + uint32 timestamp (мкс от первого отсчёта)
                         + N × int16 каналы

        struct layout: '<IIN×h'  (little-endian)
        """
        dt_us = int(round(1_000_000.0 / r.fs))
        n_ch  = data_matrix.shape[1]
        fmt   = f"<II{n_ch}h"
        record_size = struct.calcsize(fmt)

        with open(path, "wb") as f:
            for i in range(r.n_steps):
                sample_num = i + 1
                timestamp  = i * dt_us
                row        = data_matrix[i].tolist()
                f.write(struct.pack(fmt, sample_num, timestamp, *row))


# ═══════════════════════════════════════════════════════════════════════════════
# Батч-экспорт нескольких CSV
# ═══════════════════════════════════════════════════════════════════════════════

def export_batch(
    csv_paths   : list[str | Path],
    output_dir  : Path = OUTPUT_DIR,
    f0          : float = DEFAULT_F0,
    fs_fallback : float = DEFAULT_FS_FALLBACK,
) -> list[tuple[Path, Path]]:
    """
    Обрабатывает список CSV файлов и экспортирует каждый в COMTRADE.

    Parameters
    ----------
    csv_paths   : список путей к CSV
    output_dir  : куда писать .cfg / .dat
    f0          : частота сети
    fs_fallback : fs если нет в CSV

    Returns
    -------
    Список пар (cfg_path, dat_path) для каждого файла.
    """
    exporter = ComtradeExporter(output_dir=output_dir)
    results  = []

    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        print(f"  ► {csv_path.name} ", end="", flush=True)
        try:
            result = sliding_symseq(csv_path, f0=f0, fs_fallback=fs_fallback)
            cfg_p, dat_p = exporter.export(result)
            print(f"→ {cfg_p.name}  ✓")
            results.append((cfg_p, dat_p))
        except Exception as e:
            print(f"  ОШИБКА: {e}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Экспорт симметричных составляющих в COMTRADE (.cfg + .dat)"
    )
    p.add_argument(
        "--csv", default=DEFAULT_CSV,
        help=f"Путь к CSV файлу (по умолчанию: {DEFAULT_CSV})"
    )
    p.add_argument(
        "--f0", type=float, default=DEFAULT_F0,
        help=f"Частота сети Гц (по умолчанию: {DEFAULT_F0})"
    )
    p.add_argument(
        "--fs", type=float, default=DEFAULT_FS_FALLBACK,
        help=f"fs fallback Гц, если нет в CSV (по умолчанию: {DEFAULT_FS_FALLBACK})"
    )
    p.add_argument(
        "--out", default=str(OUTPUT_DIR),
        help=f"Выходная папка (по умолчанию: {OUTPUT_DIR})"
    )
    p.add_argument(
        "--batch", nargs="+", metavar="CSV",
        help="Режим батча: несколько CSV через пробел"
    )
    return p.parse_args()


def main():
    args = _parse_args()
    out  = Path(args.out)

    if args.batch:
        print(f"Батч-режим: {len(args.batch)} файлов → {out}")
        export_batch(args.batch, output_dir=out, f0=args.f0, fs_fallback=args.fs)
        return

    print(f"Загрузка  : {args.csv}")
    result = sliding_symseq(args.csv, f0=args.f0, fs_fallback=args.fs)

    print(f"fs        : {result.fs:.1f} Гц")
    print(f"Окно      : {result.win} отсч. ({1000/result.f0:.0f} мс)")
    print(f"Шагов     : {result.n_steps}")
    print(f"Дистанция : {result.distance_km:.1f} км")
    print()
    print(f"Пиковые значения:")
    print(f"  I1 = {result.I1_mag.max():.4f} А   I2 = {result.I2_mag.max():.4f} А   I0 = {result.I0_mag.max():.4f} А")
    print(f"  U1 = {result.U1_mag.max():.4f} кВ  U2 = {result.U2_mag.max():.4f} кВ  U0 = {result.U0_mag.max():.4f} кВ")
    print()

    exporter = ComtradeExporter(output_dir=out)
    cfg_p, dat_p = exporter.export(result)

    print(f"Записано  :")
    print(f"  {cfg_p}")
    print(f"  {dat_p}")


if __name__ == "__main__":
    main()
