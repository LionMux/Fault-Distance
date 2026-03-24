"""
Визуальная проверка симметричных составляющих на реальном CSV.

Метод: скользящее окно = ровно 1 период (win = round(fs / f0) отсчётов).
Для каждого окна: FFT → фазоры Xa,Xb,Xc → матрица Фортескью → X0,X1,X2.
window=False — прямоугольное окно корректно, т.к. win = целый период.

fs читается из колонки fs_hz CSV (если есть), иначе из аргумента FS_FALLBACK.

Запуск: python inspect_symseq.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from fault_distance.utils.column_detector import (
    detect_signal_columns,
    detect_distance_column,
)
from symseq.power_systems import symseq_from_waveforms

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
CSV_PATH    = "data/data_training/2AB_10km.csv"
F0          = 50.0    # Гц — частота сети (фиксирована для задачи)
FS_FALLBACK = 1000.0  # Гц — используется если в CSV нет колонки fs_hz
# ──────────────────────────────────────────────────────────────────────────────


# ─── Загрузка ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)

# Частота дискретизации: из CSV или fallback
if "fs_hz" in df.columns:
    fs = float(df["fs_hz"].iloc[0])
    print(f"fs взята из CSV       : {fs:.1f} Гц")
else:
    fs = FS_FALLBACK
    print(f"fs не найдена в CSV, использую fallback: {fs:.1f} Гц")

# Размер окна = целый период → нет утечки без оконной функции
win = int(round(fs / F0))
print(f"Окно (1 период)       : {win} отсчётов = {1000/F0:.1f} мс")

dist_col = detect_distance_column(list(df.columns))
col_map  = detect_signal_columns(list(df.columns), distance_col=dist_col)

# Служебные колонки — не идут в сигнал
service_cols = {dist_col, "fs_hz"}

sig_cols = [
    col_map["Ia"], col_map["Ib"], col_map["Ic"],
    col_map["Ua"], col_map["Ub"], col_map["Uc"],
]

sig = df[sig_cols].values.astype(float)   # (N, 6)
N   = len(sig)
dist_val = float(df[dist_col].iloc[0])

print(f"Файл                  : {os.path.basename(CSV_PATH)}")
print(f"Дистанция             : {dist_val:.1f} км")
print(f"Строк в CSV           : {N}")
print(f"Каналы                : {col_map}")
print()

if N < win:
    print(f"[ERROR] Слишком мало отсчётов ({N}) для окна {win}")
    sys.exit(1)


# ─── Скользящее FFT → симметричные составляющие ───────────────────────────────
n_steps = N - win + 1   # число позиций окна

I0_mag = np.zeros(n_steps)
I1_mag = np.zeros(n_steps)
I2_mag = np.zeros(n_steps)
U0_mag = np.zeros(n_steps)
U1_mag = np.zeros(n_steps)
U2_mag = np.zeros(n_steps)

for i in range(n_steps):
    sl = slice(i, i + win)   # окно [i : i+win]

    ri = symseq_from_waveforms(
        sig[sl, 0], sig[sl, 1], sig[sl, 2],
        fs=fs, f0=F0, window=False,   # целый период → прямоугольное окно
    )
    ru = symseq_from_waveforms(
        sig[sl, 3], sig[sl, 4], sig[sl, 5],
        fs=fs, f0=F0, window=False,
    )

    I0_mag[i] = ri["X0_mag"]
    I1_mag[i] = ri["X1_mag"]
    I2_mag[i] = ri["X2_mag"]
    U0_mag[i] = ru["X0_mag"]
    U1_mag[i] = ru["X1_mag"]
    U2_mag[i] = ru["X2_mag"]

# Временная ось: центр каждого окна, в мс
# i=0 → центр окна = win/2 отсчётов от начала
t_center_ms = (np.arange(n_steps) + win / 2) / fs * 1000.0
t_full_ms   = np.arange(N) / fs * 1000.0


# ─── Контрольный тест (синтетика) ─────────────────────────────────────────────
print("=== Контрольный тест (синтетика, прямая последовательность) ===")
t_syn = np.arange(win) / fs
A_syn = 100.0
rs = symseq_from_waveforms(
    A_syn * np.sin(2 * np.pi * F0 * t_syn),
    A_syn * np.sin(2 * np.pi * F0 * t_syn - 2 * np.pi / 3),
    A_syn * np.sin(2 * np.pi * F0 * t_syn + 2 * np.pi / 3),
    fs=fs, f0=F0, window=False,
)
ok1 = abs(rs["X1_mag"] - A_syn) < 0.5
ok2 = rs["X2_mag"] < 1.0
ok0 = rs["X0_mag"] < 1.0
print(f"  I1 = {rs['X1_mag']:.4f}  (ожидание {A_syn:.1f})  {'✓' if ok1 else '✗ ОШИБКА'}")
print(f"  I2 = {rs['X2_mag']:.6f}  (ожидание ≈ 0)     {'✓' if ok2 else '✗ ОШИБКА'}")
print(f"  I0 = {rs['X0_mag']:.6f}  (ожидание ≈ 0)     {'✓' if ok0 else '✗ ОШИБКА'}")
print()

# Итоговые значения по всему скользящему ряду (послеаварийный максимум)
print("=== Скользящее окно — пиковые значения ===")
print(f"  max I1 = {I1_mag.max():.4f} А   |  max I2 = {I2_mag.max():.4f} А   |  max I0 = {I0_mag.max():.4f} А")
print(f"  max U1 = {U1_mag.max():.4f} кВ  |  max U2 = {U2_mag.max():.4f} кВ  |  max U0 = {U0_mag.max():.4f} кВ")
print(f"  I2/I1 в максимуме = {(I2_mag / (I1_mag + 1e-12)).max():.4f}  (для 2AB ожидание ≈ 1.0)")


# ─── Графики ──────────────────────────────────────────────────────────────────
fname = os.path.basename(CSV_PATH)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(
    f"Симметричные составляющие (скользящее окно {win} отсч. = {1000/F0:.0f} мс)"
    f"  |  {fname}  |  {dist_val:.1f} км",
    fontsize=11,
)

# ── Строка 0: исходные сигналы ────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(t_full_ms, sig[:, 0], label="Ia", color="tab:blue",   lw=0.9)
ax.plot(t_full_ms, sig[:, 1], label="Ib", color="tab:orange", lw=0.9)
ax.plot(t_full_ms, sig[:, 2], label="Ic", color="tab:green",  lw=0.9)
ax.set_title("Исходные токи (А)")
ax.set_xlabel("Время (мс)")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[0, 1]
ax.plot(t_full_ms, sig[:, 3], label="Ua", color="tab:blue",   lw=0.9)
ax.plot(t_full_ms, sig[:, 4], label="Ub", color="tab:orange", lw=0.9)
ax.plot(t_full_ms, sig[:, 5], label="Uc", color="tab:green",  lw=0.9)
ax.set_title("Исходные напряжения (кВ)")
ax.set_xlabel("Время (мс)")
ax.legend(fontsize=8)
ax.grid(True)

# ── Строка 1: составляющие токов ──────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(t_center_ms, I1_mag, label="I1 прямая",   color="tab:blue",   lw=1.2)
ax.plot(t_center_ms, I2_mag, label="I2 обратная", color="tab:red",    lw=1.2)
ax.plot(t_center_ms, I0_mag, label="I0 нулевая",  color="tab:purple", lw=1.2)
ax.set_title("Амплитуды симм. составляющих токов (А)")
ax.set_xlabel("Время (мс)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(True)

# ── Строка 1: составляющие напряжений ─────────────────────────────────────────
ax = axes[1, 1]
ax.plot(t_center_ms, U1_mag, label="U1 прямая",   color="tab:blue",   lw=1.2)
ax.plot(t_center_ms, U2_mag, label="U2 обратная", color="tab:red",    lw=1.2)
ax.plot(t_center_ms, U0_mag, label="U0 нулевая",  color="tab:purple", lw=1.2)
ax.set_title("Амплитуды симм. составляющих напряжений (кВ)")
ax.set_xlabel("Время (мс)")
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)
ax.grid(True)

# ── Строка 2: коэффициенты несимметрии ────────────────────────────────────────
eps = 1e-12
I_unbal = I2_mag / (I1_mag + eps)
I0_ratio = I0_mag / (I1_mag + eps)

ax = axes[2, 0]
ax.plot(t_center_ms, I_unbal,  label="I2/I1 несимметрия", color="tab:red",    lw=1.2)
ax.plot(t_center_ms, I0_ratio, label="I0/I1 нулевая",     color="tab:purple", lw=1.2)
ax.axhline(1.0, color="gray", lw=0.8, ls="--", label="уровень 1.0")
ax.set_title("Коэффициенты несимметрии тока")
ax.set_xlabel("Время (мс)")
ax.set_ylim(0, max(3.0, I_unbal.max() * 1.1))
ax.legend(fontsize=8)
ax.grid(True)

# ── Строка 2: правая — пустая, используем для текстовой сводки ───────────────
ax = axes[2, 1]
ax.axis("off")
summary = (
    f"Файл: {fname}\n"
    f"Дистанция: {dist_val:.1f} км\n"
    f"fs = {fs:.1f} Гц\n"
    f"Окно: {win} отсч. ({1000/F0:.0f} мс)\n"
    f"Строк: {N}   Шагов: {n_steps}\n\n"
    f"Пиковые значения:\n"
    f"  I1 = {I1_mag.max():.3f} А\n"
    f"  I2 = {I2_mag.max():.3f} А\n"
    f"  I0 = {I0_mag.max():.3f} А\n"
    f"  U1 = {U1_mag.max():.4f} кВ\n"
    f"  U2 = {U2_mag.max():.4f} кВ\n"
    f"  U0 = {U0_mag.max():.4f} кВ\n\n"
    f"Контрольный тест:\n"
    f"  I1={'✓' if ok1 else '✗'}  I2={'✓' if ok2 else '✗'}  I0={'✓' if ok0 else '✗'}"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
out = "tools/output_tools/symseq_inspect.png"
plt.savefig(out, dpi=120)
print(f"\nГрафик сохранён: {out}")
