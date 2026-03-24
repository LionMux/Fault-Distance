#!/usr/bin/env python3
"""
COMTRADE → CSV Converter
=========================
Формат выхода:
- Каждый .cfg файл → отдельный .csv
- Имя CSV: <имя без _val#>_km.csv  (пример: 1A_0.5km.csv)
- Каждая строка = одна временная точка (sample)
- Колонки: distance_km, fs_hz, <токи>, <напряжения>, <прочие>
  * distance_km — константа для всего файла (первый столбец)
  * fs_hz       — частота дискретизации Гц, константа (второй столбец)
- Время НЕ включается
"""

import re
import sys
import logging
import configparser
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import comtrade
except ImportError:
    print("[ERROR] Установите: pip install comtrade")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("comtrade_to_csv.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# ── Детектирование каналов ─────────────────────────────────────────────────────
_PHASE = r"[aAbBcCnN0-3АаБбВвСсНн]"
_CURRENT_RE = re.compile(
    r"(?:^|[\s_\-\.])([Ii]" + _PHASE + r"{0,2})(?:[\s_\-\.]|$)"
    r"|^[Ii]" + _PHASE + r"{0,2}$"
)
_VOLTAGE_RE = re.compile(
    r"(?:^|[\s_\-\.])([UuVv]" + _PHASE + r"{0,2})(?:[\s_\-\.]|$)"
    r"|^[UuVv]" + _PHASE + r"{0,2}$"
)
_CURRENT_KEYWORD_RE = re.compile(r"[Ii][AaBbCcАаБбВв]", re.UNICODE)
_VOLTAGE_KEYWORD_RE = re.compile(r"[UuVv][AaBbCcАаБбВв]", re.UNICODE)


def _is_current(name: str) -> bool:
    n = name.strip()
    if re.fullmatch(r"[Ii]" + _PHASE + r"{0,2}", n):
        return True
    if bool(_CURRENT_RE.search(n)):
        return True
    if _CURRENT_KEYWORD_RE.search(n) and not _VOLTAGE_KEYWORD_RE.search(n):
        return True
    return False


def _is_voltage(name: str) -> bool:
    n = name.strip()
    if re.fullmatch(r"[UuVv]" + _PHASE + r"{0,2}", n):
        return True
    if bool(_VOLTAGE_RE.search(n)):
        return True
    if _VOLTAGE_KEYWORD_RE.search(n):
        return True
    return False


def _sort_channels(names: list, arrays: list) -> list:
    """Порядок: токи → напряжения → прочие."""
    currents = [(n, d) for n, d in zip(names, arrays) if _is_current(n)]
    voltages = [(n, d) for n, d in zip(names, arrays) if _is_voltage(n)]
    others   = [(n, d) for n, d in zip(names, arrays)
                if not _is_current(n) and not _is_voltage(n)]
    return currents + voltages + others


# ── Извлечение частоты дискретизации из COMTRADE ──────────────────────────────
def _extract_fs(rec) -> float:
    """
    Извлекает частоту дискретизации из COMTRADE-записи.

    COMTRADE (.cfg) хранит список пар (samp_rate, last_sample_num).
    Берём первую — она описывает основной сигнал.
    Если sample_rates недоступны или равны 0, вычисляем из временного вектора.

    Returns
    -------
    float
        Частота дискретизации в Гц.
    """
    try:
        rates = rec.cfg.sample_rates   # список [(samp_rate, end_sample), ...]
        if rates:
            fs = float(rates[0][0])
            if fs > 0:
                return fs
    except AttributeError:
        pass

    # Запасной вариант: вычислить из временного вектора
    time = np.array(rec.time, dtype=float)
    if len(time) >= 2:
        dt = np.median(np.diff(time))   # медиана устойчива к выбросам
        if dt > 0:
            return round(1.0 / dt, 2)

    raise ValueError("Не удалось определить частоту дискретизации из COMTRADE-файла")


# ── VAL regex ──────────────────────────────────────────────────────────────────
_VAL_RE = re.compile(r"_val(\d+)(?:\.[^.]+)?$", re.IGNORECASE)


def _extract_val(stem: str):
    m = _VAL_RE.search(stem)
    return int(m.group(1)) if m else None


def _build_output_name(stem: str, km: float) -> str:
    clean = _VAL_RE.sub("", stem)
    km_str = f"{km:.1f}" if km != int(km) else f"{int(km)}"
    return f"{clean}_{km_str}km.csv"


# ── Config ─────────────────────────────────────────────────────────────────────
def _load_config(ini_path: str = "config.ini") -> dict:
    cfg = configparser.ConfigParser()
    cfg.read(ini_path, encoding="utf-8")
    s = cfg["settings"]
    return {
        "line_length_km": float(s.get("line_length_km", 100.0)),
        "input_folder":   s.get("input_folder",  "./comtrade_files"),
        "output_folder":  s.get("output_folder", "./csv_output"),
        "test":           s.getboolean("test",      fallback=False),
        "recursive":      s.getboolean("recursive", fallback=False),
    }


def _find_cfg_files(folder_or_file: str, recursive: bool) -> list:
    p = Path(folder_or_file)

    if p.is_file() and p.suffix.lower() == ".cfg":
        return [p]

    if not p.is_dir():
        log.error(f"Путь не существует или не является папкой/файлом: {p}")
        return []

    pattern = "**/*.cfg" if recursive else "*.cfg"
    return sorted(p.glob(pattern))


# ── Один файл → один CSV ───────────────────────────────────────────────────────
def _process_file(cfg_path: Path, line_length_km: float,
                  output_folder: Path) -> bool:
    stem = cfg_path.stem
    val  = _extract_val(stem)

    if val is None:
        log.warning(f"ПРОПУСК {cfg_path.name} — нет паттерна _val#")
        return False

    km          = round((val / 100.0) * line_length_km, 6)
    output_name = _build_output_name(stem, km)
    output_path = output_folder / output_name

    try:
        rec = comtrade.load(str(cfg_path))
    except Exception as e:
        log.error(f"ОШИБКА {cfg_path.name} — {e}")
        return False

    # Частота дискретизации
    try:
        fs = _extract_fs(rec)
    except ValueError as e:
        log.error(f"ОШИБКА {cfg_path.name} — {e}")
        return False

    names  = list(rec.analog_channel_ids)
    arrays = list(rec.analog)

    if not arrays:
        log.warning(f"ПРОПУСК {cfg_path.name} — нет аналоговых каналов")
        return False

    ordered = _sort_channels(names, arrays)
    n_cur   = sum(1 for n, _ in ordered if _is_current(n))
    n_volt  = sum(1 for n, _ in ordered if _is_voltage(n))
    n_rows  = len(ordered[0][1]) if ordered else 0

    # distance_km и fs_hz — первые два столбца, константы
    data = {
        "distance_km": km,
        "fs_hz":       fs,
    }
    for ch_name, ch_data in ordered:
        data[ch_name] = np.array(ch_data, dtype=np.float64)

    df = pd.DataFrame(data)

    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, float_format="%.6g")

    log.info(
        f"OK {cfg_path.name} → {output_name}"
        f" | km={km}"
        f" | fs={fs:.1f} Гц"
        f" | I×{n_cur} U×{n_volt}"
        f" | {n_rows} строк"
    )
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    config        = _load_config("config.ini")
    input_folder  = Path(config["input_folder"])
    output_folder = Path(config["output_folder"])

    log.info("=" * 65)
    log.info(f"Входная папка  : {input_folder}")
    log.info(f"Выходная папка : {output_folder}")
    log.info(f"Длина линии    : {config['line_length_km']} км")
    log.info(f"Test={config['test']}  Recursive={config['recursive']}")
    log.info("=" * 65)

    cfg_files = _find_cfg_files(str(input_folder), config["recursive"])
    if not cfg_files:
        log.error("Не найдено .cfg файлов в указанной папке")
        return

    if config["test"]:
        cfg_files = cfg_files[:1]
        log.info(f"TEST MODE → только: {cfg_files[0].name}")

    total   = len(cfg_files)
    success = sum(
        _process_file(f, config["line_length_km"], output_folder)
        for f in cfg_files
    )

    log.info("=" * 65)
    log.info(f"Готово: {success}/{total} файлов сконвертировано")


if __name__ == "__main__":
    main()
