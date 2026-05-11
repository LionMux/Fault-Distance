"""
Демонстрационный скрипт: визуализация всех этапов предобработки.
Загружает один CSV из data/data_training/ и последовательно рисует:
1. Исходную осциллограмму
2. Удаление DC-составляющей
3. Фильтр Баттерворта
4. Детекцию момента КЗ (t0)
5. Симметричные составляющие
6. Нормализацию
7. Временной сдвиг (аугментация)
8. Гауссов шум (аугментация)
"""

import os
import sys
import numpy as np
import pandas as pd

# Добавляем корень проекта в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import Config
from data.preprocessing import remove_dc_period, sliding_window_symseq, DataPreprocessor
from data.fault_inception import FaultInceptionParams, detect_t0_single_phase, _fourth_order_difference, _cycle_difference_index
from data.augmentation import TimeShiftAugmentation, GaussianNoiseAugmentation

from plots_pipeline.plot_signals import plot_oscillogram_6ch, plot_fft_spectrum
from plots_pipeline.plot_preprocessing import (
    plot_dc_removal, plot_t0_detection, plot_symseq, plot_normalization
)
from plots_pipeline.plot_augmentation import plot_time_shift, plot_noise_levels

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------
CFG = Config()
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'data_training')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output', 'pipeline')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Выбираем один файл для демонстрации
DEMO_FILE = os.path.join(DATA_DIR, '1A_1.5km.csv')
if not os.path.exists(DEMO_FILE):
    # Пробуем найти любой CSV
    import glob
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    if not csv_files:
        raise FileNotFoundError(f"Не найдено CSV в {DATA_DIR}")
    DEMO_FILE = csv_files[0]

print(f"Демонстрационный файл: {os.path.basename(DEMO_FILE)}")

# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------
df = pd.read_csv(DEMO_FILE)
fs = float(df['fs_hz'].iloc[0]) if 'fs_hz' in df.columns else 2000.0
T = len(df)
time_ms = np.arange(T) / fs * 1000.0

signal_cols = ['CT1IA', 'CT1IB', 'CT1IC', 'S1) BUS1UA', 'S1) BUS1UB', 'S1) BUS1UC']
sig_raw = df[signal_cols].values.astype(np.float32).T  # (6, T)

print(f"Частота дискретизации: {fs:.0f} Гц")
print(f"Длительность: {T} отсчётов = {time_ms[-1]:.1f} мс")
print(f"Форма сигнала: {sig_raw.shape}")

# ---------------------------------------------------------------------------
# Этап 0: Исходная осциллограмма
# ---------------------------------------------------------------------------
print("\n[Этап 0] Исходная осциллограмма...")
plot_oscillogram_6ch(
    time_ms, sig_raw,
    channel_labels=['I_A', 'I_B', 'I_C', 'U_A', 'U_B', 'U_C'],
    units=['А', 'А', 'А', 'кВ', 'кВ', 'кВ'],
    title='Исходная осциллограмма короткого замыкания',
    output_path=os.path.join(OUTPUT_DIR, '00_original_oscillogram.png')
)

# FFT спектр для IA
print("[Этап 0] Спектр тока фазы A...")
plot_fft_spectrum(
    time_ms, sig_raw[0], fs,
    title='Спектр тока фазы A (исходный сигнал)',
    output_path=os.path.join(OUTPUT_DIR, '00_fft_spectrum_ia.png'),
    color='#E6B800'
)

# ---------------------------------------------------------------------------
# Этап 1: Удаление DC-составляющей
# ---------------------------------------------------------------------------
print("\n[Этап 1] Удаление DC-составляющей...")
sig_dc_removed = remove_dc_period(sig_raw, fs=fs, f_net=50.0)
plot_dc_removal(
    time_ms, sig_raw, sig_dc_removed, fs,
    output_path=os.path.join(OUTPUT_DIR, '01_dc_removal.png')
)

# ---------------------------------------------------------------------------
# Этап 2: Детекция момента КЗ (t0)
# ---------------------------------------------------------------------------
print("\n[Этап 3] Детекция момента КЗ (t0)...")
params = FaultInceptionParams(fs_hz=fs, mains_hz=50.0)
current_ia = sig_dc_removed[0, :]  # ток фазы A после удаления DC

d4 = _fourth_order_difference(current_ia)
di, _ = _cycle_difference_index(current_ia, params)
t0_idx = detect_t0_single_phase(current_ia, params)
if t0_idx is None:
    t0_idx = T // 2
    print(f"  [WARN] t0 не детектирован, используем середину: {t0_idx}")
else:
    print(f"  t0 = отсчёт {t0_idx} ({time_ms[t0_idx]:.1f} мс)")

plot_t0_detection(
    time_ms, current_ia, d4, di, t0_idx, fs,
    output_path=os.path.join(OUTPUT_DIR, '02_t0_detection.png')
)

# ---------------------------------------------------------------------------
# Этап 3: Симметричные составляющие
# ---------------------------------------------------------------------------
print("\n[Этап 3] Симметричные составляющие...")
symseq = sliding_window_symseq(sig_dc_removed, fs=fs, f0=50.0, window_cycles=1)
plot_symseq(
    time_ms, symseq,
    output_path=os.path.join(OUTPUT_DIR, '03_symmetrical_components.png')
)

# ---------------------------------------------------------------------------
# Этап 4: Нормализация
# ---------------------------------------------------------------------------
print("\n[Этап 4] Нормализация (standard)...")
sig_normalized = sig_dc_removed.copy()
for ch in range(sig_normalized.shape[0]):
    sig_normalized[ch] = DataPreprocessor.normalize_signal(sig_normalized[ch], method='standard')

plot_normalization(
    time_ms, sig_dc_removed, sig_normalized, method='standard',
    output_path=os.path.join(OUTPUT_DIR, '04_normalization.png')
)

# ---------------------------------------------------------------------------
# Этап 6: Аугментация — временной сдвиг
# ---------------------------------------------------------------------------
print("\n[Этап 6] Аугментация: временной сдвиг...")
# Для time-shift нам нужен DataFrame
df_for_shift = df.copy()
shifter = TimeShiftAugmentation(seq_length=T)
df_shifted_left = shifter.shift_left(df_for_shift, shift_amount=20)
df_shifted_right = shifter.shift_right(df_for_shift, shift_amount=20)

sig_shift_left = df_shifted_left[signal_cols].values.astype(np.float32).T
sig_shift_right = df_shifted_right[signal_cols].values.astype(np.float32).T

plot_time_shift(
    time_ms, sig_raw, sig_shift_left, sig_shift_right, shift_amount=20,
    output_path=os.path.join(OUTPUT_DIR, '05_time_shift_augmentation.png')
)

# ---------------------------------------------------------------------------
# Этап 6: Аугментация — гауссов шум
# ---------------------------------------------------------------------------
print("\n[Этап 6] Аугментация: гауссов шум...")
noise_aug = GaussianNoiseAugmentation(seq_length=T, num_channels=6)
noisy_dict = {}
for snr_db in [1, 5, 10, 20, 40]:
    df_noisy = noise_aug.add_gaussian_noise(df_for_shift, snr_db=snr_db, random_state=42)
    noisy_dict[snr_db] = df_noisy[signal_cols].values.astype(np.float32).T

plot_noise_levels(
    time_ms, sig_raw, noisy_dict,
    output_path=os.path.join(OUTPUT_DIR, '06_gaussian_noise_augmentation.png')
)

# ---------------------------------------------------------------------------
# Итог
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Демонстрация завершена!")
print(f"Графики сохранены в: {OUTPUT_DIR}")
print(f"{'='*60}")

# Список созданных файлов
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith('.png'):
        fpath = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  {f} ({size_kb:.1f} КБ)")
