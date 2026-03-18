#!/usr/bin/env python3
"""
Unit Tests for Data Augmentation Module
========================================

Run with: python -m pytest tests/test_augmentation.py -v
"""

import os
import sys
import pytest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.augmentation import (
    TimeShiftAugmentation,
    GaussianNoiseAugmentation,
    AugmentationPipeline
)


class TestTimeShiftAugmentation:
    """Test time shift augmentation."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with 401 rows."""
        data = {
            'distance_km': [0.5] * 401,
            'CT1IA': np.linspace(0, 260, 401),
            'CT1IB': np.linspace(0, 260, 401),
            'CT1IC': np.linspace(0, 260, 401),
            'S1) BUS1UA': np.linspace(0, 100, 401),
            'S1) BUS1UB': np.linspace(0, 100, 401),
            'S1) BUS1UC': np.linspace(0, 100, 401),
        }
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test TimeShiftAugmentation initialization."""
        ts = TimeShiftAugmentation(seq_length=401)
        assert ts.seq_length == 401
    
    def test_shift_left(self, sample_df):
        """Test left shift preserves row count."""
        ts = TimeShiftAugmentation(seq_length=401)
        
        df_shifted = ts.shift_left(sample_df, shift_amount=10)
        
        # Should maintain 401 rows
        assert len(df_shifted) == 401
        
        # Should have correct columns
        assert set(df_shifted.columns) == set(sample_df.columns)
        
        # First 10 values should be different (shifted)
        assert not np.allclose(
            df_shifted['CT1IA'].values[:10],
            sample_df['CT1IA'].values[:10],
            atol=1.0
        )
    
    def test_shift_right(self, sample_df):
        """Test right shift preserves row count."""
        ts = TimeShiftAugmentation(seq_length=401)
        
        df_shifted = ts.shift_right(sample_df, shift_amount=10)
        
        # Should maintain 401 rows
        assert len(df_shifted) == 401
        
        # Should have correct columns
        assert set(df_shifted.columns) == set(sample_df.columns)
        
        # First values should be repeated (padded)
        assert np.allclose(
            df_shifted['CT1IA'].values[:10],
            df_shifted['CT1IA'].values[10],
            atol=1e-6
        )
    
    def test_shift_zero(self, sample_df):
        """Test zero shift returns unchanged."""
        ts = TimeShiftAugmentation(seq_length=401)
        
        df_shifted = ts.shift_left(sample_df, shift_amount=0)
        
        # Should be identical to original
        pd.testing.assert_frame_equal(df_shifted, sample_df)
    
    def test_invalid_length(self):
        """Test error on wrong input length."""
        ts = TimeShiftAugmentation(seq_length=401)
        
        # Create 200-row DataFrame
        df_short = pd.DataFrame({
            'distance_km': [0.5] * 200,
            'CT1IA': np.zeros(200),
            'CT1IB': np.zeros(200),
            'CT1IC': np.zeros(200),
            'S1) BUS1UA': np.zeros(200),
            'S1) BUS1UB': np.zeros(200),
            'S1) BUS1UC': np.zeros(200),
        })
        
        with pytest.raises(ValueError):
            ts.shift_left(df_short, shift_amount=10)


class TestGaussianNoiseAugmentation:
    """Test Gaussian noise augmentation."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        data = {
            'distance_km': [0.5] * 401,
            'CT1IA': np.sin(np.linspace(0, 4*np.pi, 401)) * 100,
            'CT1IB': np.sin(np.linspace(0, 4*np.pi, 401)) * 100,
            'CT1IC': np.sin(np.linspace(0, 4*np.pi, 401)) * 100,
            'S1) BUS1UA': np.ones(401) * 100,
            'S1) BUS1UB': np.ones(401) * 100,
            'S1) BUS1UC': np.ones(401) * 100,
        }
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test GaussianNoiseAugmentation initialization."""
        gn = GaussianNoiseAugmentation(seq_length=401, num_channels=6)
        assert gn.seq_length == 401
        assert gn.num_channels == 6
    
    def test_snr_db_calculation(self, sample_df):
        """Test SNR calculation."""
        gn = GaussianNoiseAugmentation()
        
        signal = sample_df['CT1IA'].values
        noise_std = gn.calculate_noise_std(signal, snr_db=20)
        
        # Noise std should be positive
        assert noise_std > 0
        
        # Verify SNR formula
        signal_rms = np.sqrt(np.mean(signal ** 2))
        calculated_snr = 20 * np.log10(signal_rms / noise_std)
        assert np.isclose(calculated_snr, 20, atol=0.1)
    
    def test_noise_addition(self, sample_df):
        """Test noise is actually added."""
        gn = GaussianNoiseAugmentation()
        
        df_noisy = gn.add_gaussian_noise(sample_df, snr_db=5, random_state=42)
        
        # Should have same shape
        assert df_noisy.shape == sample_df.shape
        
        # Values should differ
        assert not df_noisy['CT1IA'].equals(sample_df['CT1IA'])
        
        # Distance should remain unchanged
        pd.testing.assert_series_equal(
            df_noisy['distance_km'],
            sample_df['distance_km'],
            check_exact=True
        )
    
    def test_snr_levels(self):
        """Test SNR level labels."""
        gn = GaussianNoiseAugmentation()
        
        assert gn.snr_db_to_label(1) == 'very_noisy'
        assert gn.snr_db_to_label(5) == 'noisy'
        assert gn.snr_db_to_label(10) == 'moderate'
        assert gn.snr_db_to_label(20) == 'clean'
        assert gn.snr_db_to_label(40) == 'very_clean'


class TestAugmentationPipeline:
    """Test full augmentation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    @pytest.fixture
    def sample_csv_file(self, temp_dir):
        """Create sample CSV file."""
        data = {
            'distance_km': [0.5] * 401,
            'CT1IA': np.linspace(0, 260, 401),
            'CT1IB': np.linspace(0, 260, 401),
            'CT1IC': np.linspace(0, 260, 401),
            'S1) BUS1UA': np.linspace(0, 100, 401),
            'S1) BUS1UB': np.linspace(0, 100, 401),
            'S1) BUS1UC': np.linspace(0, 100, 401),
        }
        df = pd.DataFrame(data)
        
        csv_path = os.path.join(temp_dir, 'test_0.5km.csv')
        df.to_csv(csv_path, index=False)
        
        return csv_path
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = AugmentationPipeline()
        
        assert len(pipeline.TIME_SHIFTS) == 5
        assert len(pipeline.SNR_LEVELS) == 5
    
    def test_augment_single_file(self, sample_csv_file, temp_dir):
        """Test augmentation of single file."""
        pipeline = AugmentationPipeline()
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir)
        
        created_files = pipeline.augment_single_file(
            sample_csv_file,
            output_dir,
            file_stem='test_0.5km'
        )
        
        # Should create 50 files
        # (5 time shifts left + 5 right) × 5 SNR levels
        assert len(created_files) == 50
        
        # All files should exist
        for fpath in created_files:
            assert os.path.exists(fpath)
            
            # Each file should be valid CSV with 401 rows
            df = pd.read_csv(fpath)
            assert len(df) == 401
    
    def test_augment_dataset(self, temp_dir):
        """Test augmentation of entire dataset."""
        # Create 5 sample files
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir)
        
        for i, distance in enumerate([0.5, 1.0, 1.5, 2.0, 2.5]):
            data = {
                'distance_km': [distance] * 401,
                'CT1IA': np.linspace(0, 260, 401),
                'CT1IB': np.linspace(0, 260, 401),
                'CT1IC': np.linspace(0, 260, 401),
                'S1) BUS1UA': np.linspace(0, 100, 401),
                'S1) BUS1UB': np.linspace(0, 100, 401),
                'S1) BUS1UC': np.linspace(0, 100, 401),
            }
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(input_dir, f'1A_{distance:.1f}km.csv'), index=False)
        
        # Augment
        pipeline = AugmentationPipeline()
        stats = pipeline.augment_dataset(input_dir, output_dir)
        
        # Check stats
        assert stats['original_count'] == 5
        assert stats['total_created'] == 250  # 5 files × 50 augmentations
        assert stats['expected_total'] == 250
    
    def test_augmentation_preserves_distance(self, sample_csv_file, temp_dir):
        """Test that augmentation preserves distance value."""
        pipeline = AugmentationPipeline()
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(output_dir)
        
        # Load original
        df_original = pd.read_csv(sample_csv_file)
        original_distance = df_original['distance_km'].iloc[0]
        
        # Augment
        created_files = pipeline.augment_single_file(
            sample_csv_file,
            output_dir,
            file_stem='test_0.5km'
        )
        
        # Check all augmented files have same distance
        for fpath in created_files:
            df_aug = pd.read_csv(fpath)
            assert all(df_aug['distance_km'] == original_distance)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
