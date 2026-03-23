"""
Unit tests for symseq.core — Fortescue matrix transforms.
Run with: pytest symseq/tests/test_core.py -v
"""

import numpy as np
import pytest
from symseq.core import abc_to_seq, seq_to_abc, abc_to_seq_batch, seq_to_abc_batch

ATOL = 1e-9
_A = np.exp(1j * 2 * np.pi / 3)


class TestPositiveSequence:
    """Balanced positive-sequence system: only V1 ≠ 0."""

    def test_phasors(self):
        va = 1.0 + 0j
        vb = _A**2   # lags by 120°
        vc = _A      # leads by 120° (= lags by 240°)
        v0, v1, v2 = abc_to_seq(va, vb, vc)
        assert abs(v1 - 1.0) < ATOL, f"V1 should be 1, got {v1}"
        assert abs(v0) < ATOL,       f"V0 should be 0, got {v0}"
        assert abs(v2) < ATOL,       f"V2 should be 0, got {v2}"


class TestZeroSequence:
    """Pure zero-sequence: Va = Vb = Vc = V, so V0 = V, V1 = V2 = 0."""

    def test_phasors(self):
        v = 2.5 + 1j
        v0, v1, v2 = abc_to_seq(v, v, v)
        assert abs(v0 - v) < ATOL,  f"V0 should equal V={v}, got {v0}"
        assert abs(v1) < ATOL,      f"V1 should be 0, got {v1}"
        assert abs(v2) < ATOL,      f"V2 should be 0, got {v2}"


class TestNegativeSequence:
    """Balanced negative-sequence: Va=1, Vb=a, Vc=a² → only V2 ≠ 0."""

    def test_phasors(self):
        va = 1.0 + 0j
        vb = _A      # phase B leads A (reversed rotation)
        vc = _A**2
        v0, v1, v2 = abc_to_seq(va, vb, vc)
        assert abs(v2 - 1.0) < ATOL, f"V2 should be 1, got {v2}"
        assert abs(v0) < ATOL,       f"V0 should be 0, got {v0}"
        assert abs(v1) < ATOL,       f"V1 should be 0, got {v1}"


class TestRoundTrip:
    """abc → seq → abc must return original values."""

    @pytest.mark.parametrize("va,vb,vc", [
        (1+0j, -0.5+0.866j, -0.5-0.866j),
        (3+2j, 1-1j, 0+4j),
        (0+0j, 0+0j, 0+0j),
    ])
    def test_roundtrip(self, va, vb, vc):
        v0, v1, v2 = abc_to_seq(va, vb, vc)
        ra, rb, rc = seq_to_abc(v0, v1, v2)
        assert abs(ra - va) < ATOL
        assert abs(rb - vb) < ATOL
        assert abs(rc - vc) < ATOL


class TestBatchTransform:
    """Batch variant must give same results as scalar variant."""

    def test_batch_matches_scalar(self):
        rng = np.random.default_rng(42)
        batch = rng.standard_normal((8, 3)) + 1j * rng.standard_normal((8, 3))
        result_batch = abc_to_seq_batch(batch)   # (8, 3) → (8, 3)

        for i in range(8):
            v0, v1, v2 = abc_to_seq(batch[i, 0], batch[i, 1], batch[i, 2])
            assert abs(result_batch[i, 0] - v0) < ATOL
            assert abs(result_batch[i, 1] - v1) < ATOL
            assert abs(result_batch[i, 2] - v2) < ATOL
