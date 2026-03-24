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
                           ComtradeExporter._CURRENT_CHANNELS / _VOLTAGE_CHANNELS
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
DEFAULT_CSV         = "data/data_training/2AB_10km.csv"
DEFAULT_F0          = 50.0     # Гц
DEFAULT_FS_FALLBACK = 1000.0   # Гц — если в CSV нет fs_hz
OUTPUT_DIR          = Path(__file__).parent / "output_tools"


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

    # Углы, рад (конвертируются в градусы при записи)
    I0_ang : np.ndarray = field(repr=False)
    I1_ang : np.ndarray = field(repr=False)
    I2_ang : np.ndarray = field(repr=False)

    # Амплитуды напряжений, В
    U0_mag : np.ndarray = field(repr=False)
    U1_mag : np.ndarray = field(repr=False)
    U2_mag : np.ndarray = field(repr=False)

    # Углы, рад (конвертируются в градусы при записи)
    U0_ang : np.ndarray = field(repr=False)
    U1_ang : np.ndarray = field(repr=False)
    U2_ang : np.ndarray = field(repr=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Скользящий расчёт
# ═══════════════════════════════════════════════════════════════════════════════

def sliding_symseq(
    csv_path    : str | Path,
    f0          : float = DEFAULT_F0,
    fs_fallback : float = DEFAULT_FS_FALLBACK,
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
    dist_col    = detect_distance_column(list(df.columns))
    col_map     = detect_signal_columns(list(df.columns), distance_col=dist_col)
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
        <stem>.cfg  — конфигурационный файл (ASCII)
        <stem>.dat  — данные в формате ASCII

    Формат ASCII выбран намеренно: максимальная совместимость со всеми
    осциллографическими ПО (Waves, PowerDB, DIgSILENT и др.).

    Строка канала в .cfg (IEEE Std C37.111-1999, раздел 5.3.5):
        n, ch_id, ph, ccbm, uu, a, b, skew, min, max, primary, secondary, PS

        n       — номер канала (с 1)
        ch_id   — имя канала (I1_mag и т.п.)
        ph      — фаза: пустая строка для вычисленных каналов
        ccbm    — идентификатор компонента: пустая строка для вычисленных
        uu      — единица измерения (A, deg, V)
        a       — масштабный множитель: физ. = a * raw + b
        b       — смещение (0 для всех наших каналов)
        skew    — временной сдвиг канала, мкс (0)
        min/max — диапазон сырых значений (±32767 для int16, но в ASCII
                  это поле информационное — реальные данные пишутся как float)
        primary/secondary — коэф. трансформации (1/1 для вычисленных)
        PS      — primary/secondary флаг (S)

    Каналы по умолчанию (расширяется через _CURRENT_CHANNELS/_VOLTAGE_CHANNELS):
        I1_mag, I2_mag, I0_mag  — амплитуды токов, А
        I1_ang, I2_ang, I0_ang  — углы токов, градусы
        U1_mag, U2_mag, U0_mag  — амплитуды напряжений, В
        U1_ang, U2_ang, U0_ang  — углы напряжений, градусы
    """

    # Описание канала: (attr_name, ch_id, uu)
    # ph и ccbm — пустые строки для вычисленных (не измеренных) каналов
    _CURRENT_CHANNELS = [
        ("I1_mag", "I1_mag", "A"),
        ("I2_mag", "I2_mag", "A"),
        ("I0_mag", "I0_mag", "A"),
        ("I1_ang", "I1_ang", "deg"),
        ("I2_ang", "I2_ang", "deg"),
        ("I0_ang", "I0_ang", "deg"),
    ]
    _VOLTAGE_CHANNELS = [
        ("U1_mag", "U1_mag", "V"),
        ("U2_mag", "U2_mag", "V"),
        ("U0_mag", "U0_mag", "V"),
        ("U1_ang", "U1_ang", "deg"),
        ("U2_ang", "U2_ang", "deg"),
        ("U0_ang", "U0_ang", "deg"),
    ]

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── публичный интерфейс ───────────────────────────────────────────────────

    def export(
        self,
        result : SymSeqResult,
        stem   : Optional[str] = None,
    ) -> tuple[Path, Path]:
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

        channels, phys_matrix = self._build_channels(result)
        self._write_cfg(cfg_path, result, channels)
        self._write_dat(dat_path, result, phys_matrix)

        return cfg_path, dat_path

    # ── построение каналов и матрицы физических значений ─────────────────────

    def _build_channels(self, r: SymSeqResult):
        """
        Собирает список каналов и матрицу физических значений (float64).

        Для ASCII-формата масштабирование не нужно — пишем физические
        значения напрямую. Параметры a=1, b=0 в .cfg.

        Returns
        -------
        channels    : list of dicts с полями для .cfg
        phys_matrix : np.ndarray (n_steps, n_channels), dtype float64
        """
        all_defs = self._CURRENT_CHANNELS + self._VOLTAGE_CHANNELS
        channels  = []
        data_cols = []

        for idx, (attr, ch_id, uu) in enumerate(all_defs, start=1):
            phys = getattr(r, attr).copy()

            # радианы → градусы для угловых каналов
            if uu == "deg":
                phys = np.degrees(phys)

            channels.append({
                "n"         : idx,
                "ch_id"     : ch_id,
                "ph"        : "",      # пустая — вычисленный канал
                "ccbm"      : "",      # пустая — вычисленный канал
                "uu"        : uu,
                "a"         : 1.0,     # физ. = 1.0 * raw + 0  (ASCII: raw=физ.)
                "b"         : 0.0,
                "skew"      : 0.0,
                "min"       : float(phys.min()),
                "max"       : float(phys.max()),
                "primary"   : 1.0,
                "secondary" : 1.0,
                "PS"        : "S",
            })
            data_cols.append(phys)

        phys_matrix = np.column_stack(data_cols)   # (n_steps, n_ch), float64
        return channels, phys_matrix

    # ── запись .cfg ───────────────────────────────────────────────────────────

    def _write_cfg(
        self,
        path     : Path,
        r        : SymSeqResult,
        channels : list,
    ) -> None:
        """
        Формат COMTRADE 1999 (.cfg), ASCII.
        Спецификация: IEEE Std C37.111-1999, раздел 5.
        """
        n_analog  = len(channels)
        n_digital = 0
        n_total   = n_analog + n_digital

        fs_out = r.fs
        stamp  = datetime.now().strftime("%d/%m/%Y,%H:%M:%S.%f")[:26]

        lines = []

        # строка 1: station_name, rec_dev_id, rev_year
        lines.append(f"SymSeq_{r.source_file},{r.distance_km:.2f}km,1999")

        # строка 2: TT,nA,nD
        lines.append(f"{n_total},{n_analog}A,{n_digital}D")

        # строки аналоговых каналов
        # формат: n,ch_id,ph,ccbm,uu,a,b,skew,min,max,primary,secondary,PS
        for ch in channels:
            lines.append(
                f"{ch['n']},{ch['ch_id']},{ch['ph']},{ch['ccbm']},{ch['uu']},"
                f"{ch['a']:.9e},{ch['b']:.9e},{ch['skew']:.6f},"
                f"{ch['min']:.6f},{ch['max']:.6f},"
                f"{ch['primary']:.6f},{ch['secondary']:.6f},{ch['PS']}"
            )

        # частота сети
        lines.append(f"{r.f0:.3f}")

        # частота дискретизации: nrates / samp,endsamp
        lines.append("1")
        lines.append(f"{fs_out:.6f},{r.n_steps}")

        # временны́е метки первого и последнего отсчёта
        lines.append(stamp)
        lines.append(stamp)

        # формат файла данных — ASCII
        lines.append("ASCII")

        # мультипликатор временно́го шага (timemult)
        lines.append("1")

        path.write_text("\n".join(lines) + "\n", encoding="ascii")

    # ── запись .dat (ASCII) ───────────────────────────────────────────────────

    def _write_dat(
        self,
        path        : Path,
        r           : SymSeqResult,
        phys_matrix : np.ndarray,
    ) -> None:
        """
        COMTRADE ASCII .dat:
            каждая строка = sample_number,timestamp,ch1,ch2,...,chN
            timestamp — мкс от первого отсчёта (целое)
            значения каналов — физические float, разделитель запятая

        Пример строки:
            1,0,125.34,88.12,0.03,45.00,178.21,-2.10,...
        """
        dt_us = int(round(1_000_000.0 / r.fs))   # шаг по времени, мкс

        lines = []
        for i in range(r.n_steps):
            sample_num = i + 1
            timestamp  = i * dt_us
            values     = ",".join(f"{v:.6f}" for v in phys_matrix[i])
            lines.append(f"{sample_num},{timestamp},{values}")

        path.write_text("\n".join(lines) + "\n", encoding="ascii")


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
        description="Экспорт симметричных составляющих в COMTRADE (.cfg + .dat ASCII)"
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
    print("Пиковые значения:")
    print(f"  I1 = {result.I1_mag.max():.4f} А   I2 = {result.I2_mag.max():.4f} А   I0 = {result.I0_mag.max():.4f} А")
    print(f"  U1 = {result.U1_mag.max():.4f} В   U2 = {result.U2_mag.max():.4f} В   U0 = {result.U0_mag.max():.4f} В")
    print()

    exporter = ComtradeExporter(output_dir=out)
    cfg_p, dat_p = exporter.export(result)

    print("Записано  :")
    print(f"  {cfg_p}")
    print(f"  {dat_p}")


if __name__ == "__main__":
    main()
