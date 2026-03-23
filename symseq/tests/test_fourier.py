"""
Unit tests for symseq.fourier — phasor estimation via FFT.
Run with: pytest symseq/tests/test_fourier.py -v
"""

import numpy as np
import pytest
from symseq.fourier import estimate_phasor, estimate_phasors_batch

FS = 1000.0   # Hz
F0 = 50.0     # Hz
N  = 400      # samples = exactly 20 periods → no spectral leakage


def make_sine(A: float, phi: float) -> np.ndarray:
    """Generate A * sin(2π·f0·t + phi) of length N at fs."""
    t = np.arange(N) / FS
    return A * np.sin(2 * np.pi * F0 * t + phi)


class TestEstimatePhasor:
    ATOL_AMP = 0.5   # допуск по амплитуде, А или В
    ATOL_ANG = 0.01  # допуск по фазе, рад

    @pytest.mark.parametrize("A,phi", [
        (100.0, 0.0),
        (100.0, np.pi / 4),
        (50.0, -np.pi / 3),
        (1.0, np.pi / 2),
    ])
    def test_amplitude(self, A, phi):
        sig = make_sine(A, phi)
        p = estimate_phasor(sig, FS, F0, window=False)
        assert abs(abs(p) - A) < self.ATOL_AMP, \
            f"Expected amp={A}, got {abs(p):.4f}"

    @pytest.mark.parametrize("A,phi", [
        (100.0, 0.0),
        (100.0, np.pi / 4),
    ])
    def test_phase(self, A, phi):
        # sin(ωt + φ) = cos(ωt + φ - π/2)
        # FFT phasor of sin is -j·A·e^(jφ), so angle = φ - π/2
        sig = make_sine(A, phi)
        p = estimate_phasor(sig, FS, F0, window=False)
        expected_angle = phi - np.pi / 2
        diff = abs(np.angle(p) - expected_angle)
        diff = min(diff, 2 * np.pi - diff)   # wrap
        assert diff < self.ATOL_ANG, \
            f"Expected angle≈{expected_angle:.3f}, got {np.angle(p):.3f}"


class TestEstimatePhasorsBatch:
    def test_batch_shape(self):
        sigs = np.stack([make_sine(100, 0), make_sine(50, np.pi/3)], axis=0)
        result = estimate_phasors_batch(sigs, FS, F0, window=False)
        assert result.shape == (2,)
        assert result.dtype == complex

    def test_batch_matches_scalar(self):
        params = [(100.0, 0.0), (80.0, np.pi/6), (60.0, -np.pi/4)]
        sigs = np.stack([make_sine(A, phi) for A, phi in params])
        batch_res = estimate_phasors_batch(sigs, FS, F0, window=False)
        for i, (A, phi) in enumerate(params):
            scalar_res = estimate_phasor(make_sine(A, phi), FS, F0, window=False)
            assert abs(batch_res[i] - scalar_res) < 1e-10
