"""
Fortescue symmetrical components transformation.
Pure numpy, no external dependencies beyond numpy itself.

Theory:
    [V0]   1 [1  1    1   ] [Va]
    [V1] = - [1  a    a^2 ] [Vb]
    [V2]   3 [1  a^2  a   ] [Vc]

    where a = exp(j * 2*pi/3)

Inverse:
    [Va]   [1  1   1  ] [V0]
    [Vb] = [1  a^2 a  ] [V1]
    [Vc]   [1  a   a^2] [V2]
"""

import numpy as np

# Оператор Фортескью: a = e^(j*2π/3)
_A = np.exp(1j * 2 * np.pi / 3)

# Прямая матрица (abc → 012), масштаб 1/3 уже включён
_F = (1 / 3) * np.array([
    [1,      1,       1      ],
    [1,      _A,      _A**2  ],
    [1,      _A**2,   _A     ],
], dtype=complex)

# Обратная матрица (012 → abc)
_F_INV = np.array([
    [1,  1,      1     ],
    [1,  _A**2,  _A    ],
    [1,  _A,     _A**2 ],
], dtype=complex)


def abc_to_seq(
    va: complex | np.ndarray,
    vb: complex | np.ndarray,
    vc: complex | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert three-phase phasors (or arrays of phasors) to
    symmetrical sequence components (zero, positive, negative).

    Parameters
    ----------
    va, vb, vc : complex scalar or np.ndarray
        Phase phasors. Can be any broadcastable shape.

    Returns
    -------
    v0, v1, v2 : np.ndarray
        Zero, positive (direct), negative (inverse) sequence phasors.
        Same shape as input.

    Examples
    --------
    >>> # Balanced positive-sequence system
    >>> a = np.exp(1j * 2 * np.pi / 3)
    >>> va, vb, vc = 1+0j, a**2, a
    >>> v0, v1, v2 = abc_to_seq(va, vb, vc)
    >>> assert abs(v1 - 1) < 1e-9
    >>> assert abs(v0) < 1e-9
    >>> assert abs(v2) < 1e-9
    """
    va = np.asarray(va, dtype=complex)
    vb = np.asarray(vb, dtype=complex)
    vc = np.asarray(vc, dtype=complex)

    v0 = _F[0, 0] * va + _F[0, 1] * vb + _F[0, 2] * vc
    v1 = _F[1, 0] * va + _F[1, 1] * vb + _F[1, 2] * vc
    v2 = _F[2, 0] * va + _F[2, 1] * vb + _F[2, 2] * vc

    return v0, v1, v2


def seq_to_abc(
    v0: complex | np.ndarray,
    v1: complex | np.ndarray,
    v2: complex | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert symmetrical sequence components back to three-phase phasors.

    Parameters
    ----------
    v0, v1, v2 : complex scalar or np.ndarray
        Zero, positive, negative sequence phasors.

    Returns
    -------
    va, vb, vc : np.ndarray
        Reconstructed phase phasors. Same shape as input.
    """
    v0 = np.asarray(v0, dtype=complex)
    v1 = np.asarray(v1, dtype=complex)
    v2 = np.asarray(v2, dtype=complex)

    va = _F_INV[0, 0] * v0 + _F_INV[0, 1] * v1 + _F_INV[0, 2] * v2
    vb = _F_INV[1, 0] * v0 + _F_INV[1, 1] * v1 + _F_INV[1, 2] * v2
    vc = _F_INV[2, 0] * v0 + _F_INV[2, 1] * v1 + _F_INV[2, 2] * v2

    return va, vb, vc


def abc_to_seq_batch(x: np.ndarray) -> np.ndarray:
    """
    Vectorised Fortescue transform for batched data.

    Parameters
    ----------
    x : np.ndarray, shape (..., 3)
        Last axis must be [Va, Vb, Vc] as complex values.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Last axis is [V0, V1, V2].
    """
    x = np.asarray(x, dtype=complex)
    return x @ _F.T  # (..., 3) @ (3, 3) → (..., 3)


def seq_to_abc_batch(x: np.ndarray) -> np.ndarray:
    """
    Inverse Fortescue transform for batched data.

    Parameters
    ----------
    x : np.ndarray, shape (..., 3)
        Last axis must be [V0, V1, V2] as complex values.

    Returns
    -------
    np.ndarray, shape (..., 3)
        Last axis is [Va, Vb, Vc].
    """
    x = np.asarray(x, dtype=complex)
    return x @ _F_INV.T  # (..., 3) @ (3, 3) → (..., 3)
