"""
Symmetrical components adapter for Fault-Distance batch format.

Input tensor shape: (B, 6, N)
Channel order:       [Ia, Ib, Ic, Ua, Ub, Uc]  (indices 0-5)

For each sample in the batch, computes positive (1), negative (2),
and zero (0) sequence phasors for both current and voltage groups.

Returns a dict of numpy arrays, all shape (B,).
No PyTorch dependency — pure numpy, ready for preprocessing pipelines.
"""

import numpy as np
from symseq.power_systems import symseq_from_waveforms

# Зафиксированный порядок каналов в батче
CHANNEL_ORDER = ["Ia", "Ib", "Ic", "Ua", "Ub", "Uc"]

# Индексы каналов
_I_IDX = (0, 1, 2)  # Ia, Ib, Ic
_U_IDX = (3, 4, 5)  # Ua, Ub, Uc


def compute_symseq_batch(
    x: np.ndarray,
    fs: float,
    f0: float = 50.0,
    window: bool = True,
) -> dict[str, np.ndarray]:
    """
    Compute symmetrical sequence phasors for a batch of oscillograms.

    Parameters
    ----------
    x : np.ndarray, shape (B, 6, N)
        Batch of oscillograms. Channels must follow CHANNEL_ORDER:
        [Ia, Ib, Ic, Ua, Ub, Uc].
    fs : float
        Sampling frequency in Hz (e.g. 1000.0 for 1 kHz).
    f0 : float
        Fundamental frequency in Hz (default 50.0).
    window : bool
        Apply Hann window before FFT. Default True.
        Set False only if N contains an exact integer number of periods.

    Returns
    -------
    dict with keys (all arrays have shape (B,), dtype complex128 or float64):

        Phasors (complex):
            'I0', 'I1', 'I2'  — current sequence phasors
            'U0', 'U1', 'U2'  — voltage sequence phasors

        Magnitudes (float):
            'I0_mag', 'I1_mag', 'I2_mag'
            'U0_mag', 'U1_mag', 'U2_mag'

        Angles in radians (float):
            'I0_ang', 'I1_ang', 'I2_ang'
            'U0_ang', 'U1_ang', 'U2_ang'

        Derived features (float) — useful as direct ML inputs:
            'I_unbalance'  — |I2| / |I1|, обратная / прямая (ток)
            'U_unbalance'  — |U2| / |U1|, обратная / прямая (напряжение)
            'I0_ratio'     — |I0| / |I1|, нулевая / прямая (ток)
            'U0_ratio'     — |U0| / |U1|, нулевая / прямая (напряжение)

    Raises
    ------
    ValueError
        If x.ndim != 3 or x.shape[1] != 6.

    Examples
    --------
    >>> import numpy as np
    >>> fs, f0, N = 1000.0, 50.0, 400
    >>> B = 4
    >>> x = np.zeros((B, 6, N))
    >>> t = np.arange(N) / fs
    >>> a = np.exp(1j * 2 * np.pi / 3)
    >>> # Fill balanced positive-sequence currents
    >>> x[:, 0, :] = 100 * np.sin(2 * np.pi * f0 * t)           # Ia
    >>> x[:, 1, :] = 100 * np.sin(2 * np.pi * f0 * t - 2*np.pi/3)  # Ib
    >>> x[:, 2, :] = 100 * np.sin(2 * np.pi * f0 * t + 2*np.pi/3)  # Ic
    >>> result = compute_symseq_batch(x, fs, f0, window=False)
    >>> assert result['I1_mag'].shape == (B,)
    >>> assert np.all(result['I1_mag'] > 90)   # ~100 A
    >>> assert np.all(result['I2_mag'] < 1)    # balanced → I2 ≈ 0
    """
    x = np.asarray(x, dtype=float)

    if x.ndim != 3:
        raise ValueError(f"Expected x.ndim == 3, got {x.ndim}")
    if x.shape[1] != 6:
        raise ValueError(
            f"Expected 6 channels [Ia,Ib,Ic,Ua,Ub,Uc], got {x.shape[1]}"
        )

    B = x.shape[0]

    # Инициализация выходных массивов
    keys_complex = ['I0', 'I1', 'I2', 'U0', 'U1', 'U2']
    result: dict[str, np.ndarray] = {k: np.zeros(B, dtype=complex) for k in keys_complex}

    for i in range(B):
        # --- Токи ---
        i_res = symseq_from_waveforms(
            x[i, 0, :], x[i, 1, :], x[i, 2, :],
            fs=fs, f0=f0, window=window,
        )
        result['I0'][i] = i_res['X0']
        result['I1'][i] = i_res['X1']
        result['I2'][i] = i_res['X2']

        # --- Напряжения ---
        u_res = symseq_from_waveforms(
            x[i, 3, :], x[i, 4, :], x[i, 5, :],
            fs=fs, f0=f0, window=window,
        )
        result['U0'][i] = u_res['X0']
        result['U1'][i] = u_res['X1']
        result['U2'][i] = u_res['X2']

    # Модули и углы
    for prefix in ('I', 'U'):
        for seq in ('0', '1', '2'):
            k = f'{prefix}{seq}'
            result[f'{k}_mag'] = np.abs(result[k])
            result[f'{k}_ang'] = np.angle(result[k])

    # Производные признаки (защита от деления на ноль)
    eps = 1e-12
    result['I_unbalance'] = result['I2_mag'] / (result['I1_mag'] + eps)
    result['U_unbalance'] = result['U2_mag'] / (result['U1_mag'] + eps)
    result['I0_ratio']    = result['I0_mag'] / (result['I1_mag'] + eps)
    result['U0_ratio']    = result['U0_mag'] / (result['U1_mag'] + eps)

    return result


def compute_symseq_feature_vector(
    x: np.ndarray,
    fs: float,
    f0: float = 50.0,
    window: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: returns a float feature matrix (B, 16).

    Feature columns (16 total):
        I1_mag, I2_mag, I0_mag, I1_ang, I2_ang, I0_ang,
        U1_mag, U2_mag, U0_mag, U1_ang, U2_ang, U0_ang,
        I_unbalance, U_unbalance, I0_ratio, U0_ratio

    Parameters
    ----------
    x : np.ndarray, shape (B, 6, N)
    fs, f0, window : same as compute_symseq_batch

    Returns
    -------
    np.ndarray, shape (B, 16), dtype float64
    """
    d = compute_symseq_batch(x, fs, f0, window)

    cols = [
        'I1_mag', 'I2_mag', 'I0_mag',
        'I1_ang', 'I2_ang', 'I0_ang',
        'U1_mag', 'U2_mag', 'U0_mag',
        'U1_ang', 'U2_ang', 'U0_ang',
        'I_unbalance', 'U_unbalance',
        'I0_ratio', 'U0_ratio',
    ]
    return np.column_stack([d[c] for c in cols]).astype(np.float64)
