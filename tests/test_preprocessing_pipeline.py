"""
Unit tests for the notebook-aligned preprocessing pipeline:
    - remove_dc_period
    - 12-channel tensor formation
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import remove_dc_period


# ---------------------------------------------------------------------------
# remove_dc_period
# ---------------------------------------------------------------------------

class TestRemoveDCPeriod:
    def test_removes_pure_dc(self):
        """A constant signal should become all zeros."""
        fs, f0 = 2000.0, 50.0
        sig = np.ones((3, 400), dtype=np.float32) * 5.0
        out = remove_dc_period(sig, fs, f0)
        # After subtracting the rolling mean, a constant becomes ~0
        np.testing.assert_allclose(out, 0, atol=1e-5)

    def test_preserves_sinusoid(self):
        """A pure sinusoid should be nearly unchanged (mean over integer periods is 0)."""
        fs, f0 = 2000.0, 50.0
        t = np.arange(400) / fs
        x = np.sin(2 * np.pi * f0 * t)
        sig = x[np.newaxis, :].astype(np.float32)
        out = remove_dc_period(sig, fs, f0)
        # Should preserve shape and be very close to original
        assert out.shape == sig.shape
        np.testing.assert_allclose(out, sig, atol=1e-3)

    def test_1d_input(self):
        """Accept 1-D array."""
        fs, f0 = 2000.0, 50.0
        x = np.ones(400, dtype=np.float32) * 3.0
        out = remove_dc_period(x, fs, f0)
        np.testing.assert_allclose(out, 0, atol=1e-5)

    def test_rejects_3d(self):
        """Should raise on 3-D input."""
        with pytest.raises(ValueError):
            remove_dc_period(np.zeros((2, 3, 400)), 2000.0, 50.0)


# ---------------------------------------------------------------------------
# 12-channel dataset integration (requires real CSV files)
# ---------------------------------------------------------------------------

class TestTwelveChannelDataset:
    @pytest.fixture(scope='class')
    def csv_dir(self):
        d = os.path.join(os.path.dirname(__file__), '..', 'data', 'csv_all')
        if not os.path.isdir(d):
            pytest.skip(f'CSV directory not found: {d}')
        return d

    def test_12ch_shape(self, csv_dir):
        from config import Config
        from data.dataset import FaultDataset
        cfg = Config(SYMSEQ_ENABLED=True, REMOVE_DC_ENABLED=False,
                     T0_ENABLED=False, NUM_CHANNELS=12, NORMALIZE_DATA=False)
        ds = FaultDataset(csv_dir, seq_length=400, cfg=cfg, normalize=False)
        assert ds.signals.shape[1] == 12
        assert ds.signals.shape[2] == 400
        assert ds.num_channels == 12

    def test_6ch_backward_compatible(self, csv_dir):
        from config import Config
        from data.dataset import FaultDataset
        cfg = Config(SYMSEQ_ENABLED=False, REMOVE_DC_ENABLED=False,
                     T0_ENABLED=False, NUM_CHANNELS=6, NORMALIZE_DATA=False)
        ds = FaultDataset(csv_dir, seq_length=400, cfg=cfg, normalize=False)
        assert ds.signals.shape[1] == 6
        assert ds.num_channels == 6

    def test_symseq_channels_are_time_varying(self, csv_dir):
        """Sliding-window phasor channels (3-5, 9-11) should vary over time."""
        from config import Config
        from data.dataset import FaultDataset
        cfg = Config(SYMSEQ_ENABLED=True, REMOVE_DC_ENABLED=False,
                     T0_ENABLED=False, NUM_CHANNELS=12, NORMALIZE_DATA=False)
        ds = FaultDataset(csv_dir, seq_length=400, cfg=cfg, normalize=False)
        # Check a few samples — time-varying channels should NOT be constant
        for idx in [0, 10, 20]:
            for ch in [3, 4, 5, 9, 10, 11]:
                ts = ds.signals[idx, ch, :]
                assert np.std(ts) > 0.01, \
                    f'Sample {idx} channel {ch} std={np.std(ts):.4f} is too flat'
                assert np.all(ts >= 0), \
                    f'Sample {idx} channel {ch} contains negative magnitude'

    def test_remove_dc_lowers_mean(self, csv_dir):
        from config import Config
        from data.dataset import FaultDataset
        cfg_on = Config(SYMSEQ_ENABLED=False, REMOVE_DC_ENABLED=True,
                        T0_ENABLED=False, NUM_CHANNELS=6, NORMALIZE_DATA=False)
        ds_on = FaultDataset(csv_dir, seq_length=400, cfg=cfg_on, normalize=False)

        cfg_off = Config(SYMSEQ_ENABLED=False, REMOVE_DC_ENABLED=False,
                         T0_ENABLED=False, NUM_CHANNELS=6, NORMALIZE_DATA=False)
        ds_off = FaultDataset(csv_dir, seq_length=400, cfg=cfg_off, normalize=False)

        # Mean absolute value of DC component should be smaller with filter
        mean_dc_on = np.abs(ds_on.signals[:, 0, :].mean(axis=1)).mean()
        mean_dc_off = np.abs(ds_off.signals[:, 0, :].mean(axis=1)).mean()
        assert mean_dc_on < mean_dc_off
