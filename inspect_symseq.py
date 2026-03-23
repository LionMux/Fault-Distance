"""
Визуальная проверка симметричных составляющих на реальном CSV.
Запуск: python inspect_symseq.py
"""

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fault_distance.features.symseq_adapter import compute_symseq_batch
from fault_distance.utils.column_detector import detect_signal_columns, detect_distance_column
from symseq.power_systems import instantaneous_symseq

# ─── НАСТРОЙКИ ────────────────────────────────────────────────────────────────
CSV_PATH = "data/data_training/2AB_10km.csv"  # <-- поменяй если нужно
FS  = 1000.0
F0  = 50.0
SEQ = 400
# ──────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV_PATH)

dist_col  = detect_distance_column(list(df.columns))
col_map   = detect_signal_columns(list(df.columns), distance_col=dist_col)
sig_cols  = [col_map['Ia'], col_map['Ib'], col_map['Ic'],
             col_map['Ua'], col_map['Ub'], col_map['Uc']]

print(f"CSV shape    : {df.shape}")
print(f"Distance     : {df[dist_col].iloc[0]:.2f} km")
print(f"Detected     : {col_map}")
print()

sig = df[sig_cols].values[:SEQ, :].astype(float)   # (400, 6)
x   = sig.T[np.newaxis, :, :]                      # (1, 6, 400)

# ─── FFT-фазоры ───────────────────────────────────────────────────────────────
r = compute_symseq_batch(x, FS, F0, window=True)

print("=== Симметричные составляющие (FFT) ===")
print(f"  I1 (прямая)   : {r['I1_mag'][0]:8.4f} А   ∠{np.degrees(r['I1_ang'][0]):7.2f}°")
print(f"  I2 (обратная) : {r['I2_mag'][0]:8.4f} А   ∠{np.degrees(r['I2_ang'][0]):7.2f}°")
print(f"  I0 (нулевая)  : {r['I0_mag'][0]:8.4f} А   ∠{np.degrees(r['I0_ang'][0]):7.2f}°")
print(f"  U1 (прямая)   : {r['U1_mag'][0]:8.4f} кВ  ∠{np.degrees(r['U1_ang'][0]):7.2f}°")
print(f"  U2 (обратная) : {r['U2_mag'][0]:8.4f} кВ  ∠{np.degrees(r['U2_ang'][0]):7.2f}°")
print(f"  U0 (нулевая)  : {r['U0_mag'][0]:8.4f} кВ  ∠{np.degrees(r['U0_ang'][0]):7.2f}°")
print()
print(f"  Несимметрия тока    I2/I1 = {r['I_unbalance'][0]:.4f}")
print(f"  Нулевая / прямая    I0/I1 = {r['I0_ratio'][0]:.4f}")

# ─── Мгновенные составляющие ──────────────────────────────────────────────────
t = np.arange(SEQ) / FS * 1000  # мс

i0, i1, i2 = instantaneous_symseq(sig[:, 0], sig[:, 1], sig[:, 2])
u0, u1, u2 = instantaneous_symseq(sig[:, 3], sig[:, 4], sig[:, 5])

# ─── Графики ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fname = os.path.basename(CSV_PATH)
fig.suptitle(f"Симметричные составляющие  |  {fname}  |  {df[dist_col].iloc[0]:.1f} км", fontsize=12)

# Исходные токи
axes[0, 0].plot(t, sig[:, 0], label='Ia'); axes[0, 0].plot(t, sig[:, 1], label='Ib'); axes[0, 0].plot(t, sig[:, 2], label='Ic')
axes[0, 0].set_title('Исходные токи (А)'); axes[0, 0].legend(); axes[0, 0].grid(True)

# Исходные напряжения
axes[0, 1].plot(t, sig[:, 3], label='Ua'); axes[0, 1].plot(t, sig[:, 4], label='Ub'); axes[0, 1].plot(t, sig[:, 5], label='Uc')
axes[0, 1].set_title('Исходные напряжения (кВ)'); axes[0, 1].legend(); axes[0, 1].grid(True)

# Модули мгновенных токов
axes[1, 0].plot(t, np.abs(i1), label='|I1| прямая',   color='tab:blue')
axes[1, 0].plot(t, np.abs(i2), label='|I2| обратная', color='tab:red')
axes[1, 0].plot(t, np.abs(i0), label='|I0| нулевая',  color='tab:purple')
axes[1, 0].set_title('|Мгновенные симм. составляющие| токов (А)'); axes[1, 0].legend(); axes[1, 0].grid(True)

# Модули мгновенных напряжений
axes[1, 1].plot(t, np.abs(u1), label='|U1| прямая',   color='tab:blue')
axes[1, 1].plot(t, np.abs(u2), label='|U2| обратная', color='tab:red')
axes[1, 1].plot(t, np.abs(u0), label='|U0| нулевая',  color='tab:purple')
axes[1, 1].set_title('|Мгновенные симм. составляющие| напряжений (кВ)'); axes[1, 1].legend(); axes[1, 1].grid(True)

# Re-части токов
axes[2, 0].plot(t, i1.real, label='Re(I1)', color='tab:blue')
axes[2, 0].plot(t, i2.real, label='Re(I2)', color='tab:red')
axes[2, 0].plot(t, i0.real, label='Re(I0)', color='tab:purple')
axes[2, 0].set_title('Re(симм. составляющие) токов (А)'); axes[2, 0].set_xlabel('Время (мс)'); axes[2, 0].legend(); axes[2, 0].grid(True)

# Re-части напряжений
axes[2, 1].plot(t, u1.real, label='Re(U1)', color='tab:blue')
axes[2, 1].plot(t, u2.real, label='Re(U2)', color='tab:red')
axes[2, 1].plot(t, u0.real, label='Re(U0)', color='tab:purple')
axes[2, 1].set_title('Re(симм. составляющие) напряжений (кВ)'); axes[2, 1].set_xlabel('Время (мс)'); axes[2, 1].legend(); axes[2, 1].grid(True)

plt.tight_layout()
plt.savefig("symseq_inspect.png", dpi=120)
print("\nГрафик сохранён: symseq_inspect.png")
plt.show()
