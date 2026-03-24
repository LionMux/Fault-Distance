"""
Минимальная проверка матрицы Фортескью.
Синтетические данные с известным ответом — без CSV, без FFT.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from symseq.core import abc_to_seq

print("=" * 60)
print("ТЕСТ 1: Чистая ПРЯМАЯ последовательность")
print("  Ia = 1∠0°,  Ib = 1∠-120°,  Ic = 1∠+120°")
print("  Ожидание: I1=1, I2≈0, I0≈0")
Va = 1.0 + 0j
Vb = np.exp(-1j * 2 * np.pi / 3)
Vc = np.exp(+1j * 2 * np.pi / 3)
V0, V1, V2 = abc_to_seq(Va, Vb, Vc)
print(f"  I0 = {abs(V0):.6f}  (ожидание ≈ 0)")
print(f"  I1 = {abs(V1):.6f}  (ожидание ≈ 1)")
print(f"  I2 = {abs(V2):.6f}  (ожидание ≈ 0)")

print()
print("=" * 60)
print("ТЕСТ 2: Чистая ОБРАТНАЯ последовательность")
print("  Ia = 1∠0°,  Ib = 1∠+120°,  Ic = 1∠-120°")
print("  Ожидание: I2=1, I1≈0, I0≈0")
Va = 1.0 + 0j
Vb = np.exp(+1j * 2 * np.pi / 3)
Vc = np.exp(-1j * 2 * np.pi / 3)
V0, V1, V2 = abc_to_seq(Va, Vb, Vc)
print(f"  I0 = {abs(V0):.6f}  (ожидание ≈ 0)")
print(f"  I1 = {abs(V1):.6f}  (ожидание ≈ 0)")
print(f"  I2 = {abs(V2):.6f}  (ожидание ≈ 1)")

print()
print("=" * 60)
print("ТЕСТ 3: Чистая НУЛЕВАЯ последовательность")
print("  Ia = Ib = Ic = 1∠0°")
print("  Ожидание: I0=1, I1≈0, I2≈0")
Va = Vb = Vc = 1.0 + 0j
V0, V1, V2 = abc_to_seq(Va, Vb, Vc)
print(f"  I0 = {abs(V0):.6f}  (ожидание ≈ 1)")
print(f"  I1 = {abs(V1):.6f}  (ожидание ≈ 0)")
print(f"  I2 = {abs(V2):.6f}  (ожидание ≈ 0)")

print()
print("=" * 60)
print("ТЕСТ 4: Двухфазное КЗ AB (2AB)")
print("  При 2AB: Ia≠0, Ib≠0, Ic=0 → I1≈I2, I0≈0")
Ia = 1.0 + 0j
Ib = np.exp(1j * np.pi)   # противофаза к Ia
Ic = 0.0 + 0j
V0, V1, V2 = abc_to_seq(Ia, Ib, Ic)
print(f"  I0 = {abs(V0):.6f}  (ожидание ≈ 0)")
print(f"  I1 = {abs(V1):.6f}  (ожидание = I2)")
print(f"  I2 = {abs(V2):.6f}  (ожидание = I1)")
