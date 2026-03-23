"""
Integration tests for symseq_adapter — batch (B, 6, N) format.
Run with: pytest symseq/tests/test_adapter.py -v
"""

import numpy as np
import pytest
import sys, os

# Позволяем импортировать src/fault_distance из любого места
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from fault_distance.features.symseq_adapter import (
    compute_symseq_batch,
    compute_symseq_feature_vector,
    CHANNEL_ORDER,
)

FS  = 1000.0
F0  = 50.0
N   = 400      # 20 периодов → нет утечки без окна
B   = 4


def make_batch_balanced(amplitude: float = 100.0) -> np.ndarray:
    """Сбалансированная прямая последовательность токов, нулевые напряжения."""
    t = np.arange(N) / FS
    x = np.zeros((B, 6, N))
    x[:, 0, :] = amplitude * np.sin(2 * np.pi * F0 * t)
    x[:, 1, :] = amplitude * np.sin(2 * np.pi * F0 * t - 2 * np.pi / 3)
    x[:, 2, :] = amplitude * np.sin(2 * np.pi * F0 * t + 2 * np.pi / 3)
    return x


def make_batch_zero_seq(amplitude: float = 100.0) -> np.ndarray:
    """Чистая нулевая последовательность: Ia = Ib = Ic (однофазное КЗ)."""
    t = np.arange(N) / FS
    x = np.zeros((B, 6, N))
    sig = amplitude * np.sin(2 * np.pi * F0 * t)
    x[:, 0, :] = sig
    x[:, 1, :] = sig
    x[:, 2, :] = sig
    return x


class TestOutputShapes:
    def test_dict_keys(self):
        x = make_batch_balanced()
        result = compute_symseq_batch(x, FS, F0, window=False)
        expected_keys = {
            'I0','I1','I2','U0','U1','U2',
            'I0_mag','I1_mag','I2_mag',
            'U0_mag','U1_mag','U2_mag',
            'I0_ang','I1_ang','I2_ang',
            'U0_ang','U1_ang','U2_ang',
            'I_unbalance','U_unbalance','I0_ratio','U0_ratio',
        }
        assert expected_keys == set(result.keys())

    def test_shapes(self):
        x = make_batch_balanced()
        result = compute_symseq_batch(x, FS, F0, window=False)
        for k, v in result.items():
            assert v.shape == (B,), f"Key '{k}': expected ({B},), got {v.shape}"

    def test_feature_vector_shape(self):
        x = make_batch_balanced()
        fv = compute_symseq_feature_vector(x, FS, F0, window=False)
        assert fv.shape == (B, 16)
        assert fv.dtype == np.float64


class TestBalancedPositiveSequence:
    """Для сбалансированной прямой последовательности: I1 ≈ A, I0 ≈ I2 ≈ 0."""

    def test_I1_magnitude(self):
        x = make_batch_balanced(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        np.testing.assert_allclose(r['I1_mag'], 100.0, atol=0.5)

    def test_I2_near_zero(self):
        x = make_batch_balanced(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        assert np.all(r['I2_mag'] < 1.0), f"I2_mag should be ~0, got {r['I2_mag']}"

    def test_I0_near_zero(self):
        x = make_batch_balanced(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        assert np.all(r['I0_mag'] < 1.0), f"I0_mag should be ~0, got {r['I0_mag']}"

    def test_unbalance_near_zero(self):
        x = make_batch_balanced(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        assert np.all(r['I_unbalance'] < 0.01)


class TestZeroSequence:
    """Ia = Ib = Ic → I0 ≈ Ia, I1 ≈ I2 ≈ 0."""

    def test_I0_magnitude(self):
        x = make_batch_zero_seq(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        # I0 = (Ia + Ib + Ic) / 3 = Ia при Ia=Ib=Ic
        np.testing.assert_allclose(r['I0_mag'], 100.0, atol=0.5)

    def test_I1_near_zero(self):
        x = make_batch_zero_seq(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        assert np.all(r['I1_mag'] < 1.0)

    def test_I2_near_zero(self):
        x = make_batch_zero_seq(amplitude=100.0)
        r = compute_symseq_batch(x, FS, F0, window=False)
        assert np.all(r['I2_mag'] < 1.0)


class TestValidation:
    def test_wrong_ndim(self):
        with pytest.raises(ValueError, match="ndim"):
            compute_symseq_batch(np.zeros((6, 400)), FS, F0)

    def test_wrong_channels(self):
        with pytest.raises(ValueError, match="6 channels"):
            compute_symseq_batch(np.zeros((4, 3, 400)), FS, F0)
