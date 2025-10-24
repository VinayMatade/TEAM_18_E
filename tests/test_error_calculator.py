"""
Tests for GPS error calculation processor.

Tests error vector computation, statistics calculation, and validation functionality.
"""

import pytest
import pandas as pd
import numpy as np
import logging

from uav_log_processor.processors.error_calculator import ErrorCalculator


class TestErrorCalculator:
    """Test cases for ErrorCalculator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'max_error_threshold': 50.0,
            'temporal_consistency_window': 5
        }
        self.calculator = ErrorCalculator(self.config)
    
    def create_test_data(self, num_samples=100, add_noise=True):
        """Create test data with GPS and ground truth positions."""
        timestamps = np.linspace(0, 10, num_samples)
        
        # Create ground truth trajectory (circular motion)
        t = timestamps
        radius = 10.0
        truth_x = radius * np.cos(0.5 * t)
        truth_y = radius * np.sin(0.5 * t)
        truth_z = np.ones_like(t) * 100.0
        
        # Create GPS positions with known errors
        if add_noise:
            # Add systematic bias and random noise
            gps_x = truth_x + 2.0 + np.random.normal(0, 1.0, len(t))  # 2m bias + 1m noise
            gps_y = truth_y - 1.5 + np.random.normal(0, 0.8, len(t))  # -1.5m bias + 0.8m noise
            gps_z = truth_z + 0.5 + np.random.normal(0, 0.5, len(t))  # 0.5m bias + 0.5m noise
        else:
            # Perfect GPS for controlled tests
            gps_x = truth_x.copy()
            gps_y = truth_y.copy()
            gps_z = truth_z.copy()
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'gps_x': gps_x,
            'gps_y': gps_y,
            'gps_z': gps_z,
            'ground_truth_x': truth_x,
            'ground_truth_y': truth_y,
            'ground_truth_z': truth_z
        })
        
        return data
    
    def create_synthetic_error_data(self):
        """Create data with known error patterns for validation."""
        num_samples = 50
        
        # Create simple linear trajectory
        truth_x = np.linspace(0, 10, num_samples)
        truth_y = np.zeros(num_samples)
        truth_z = np.ones(num_samples) * 100.0
        
        # Add known errors
        gps_x = truth_x + 1.0  # Constant 1m error in x
        gps_y = truth_y + 2.0  # Constant 2m error in y
        gps_z = truth_z - 0.5  # Constant -0.5m error in z
        
        data = pd.DataFrame({
            'gps_x': gps_x,
            'gps_y': gps_y,
            'gps_z': gps_z,
            'ground_truth_x': truth_x,
            'ground_truth_y': truth_y,
            'ground_truth_z': truth_z
        })
        
        return data
    
    def test_initialization(self):
        """Test ErrorCalculator initialization."""
        calculator = ErrorCalculator()
        assert calculator.config == {}
        assert calculator.max_error_threshold == 100.0  # default
        assert calculator.temporal_consistency_window == 5  # default
        
        calculator_with_config = ErrorCalculator(self.config)
        assert calculator_with_config.max_error_threshold == 50.0
        assert calculator_with_config.temporal_consistency_window == 5
    
    def test_validate_input_valid_data(self):
        """Test input validation with valid data."""
        data = self.create_test_data()
        assert self.calculator.validate_input(data) == True
    
    def test_validate_input_missing_gps_columns(self):
        """Test input validation with missing GPS columns."""
        data = self.create_test_data()
        data = data.drop(['gps_x', 'gps_y'], axis=1)
        assert self.calculator.validate_input(data) == False
    
    def test_validate_input_missing_ground_truth_columns(self):
        """Test input validation with missing ground truth columns."""
        data = self.create_test_data()
        data = data.drop(['ground_truth_x', 'ground_truth_z'], axis=1)
        assert self.calculator.validate_input(data) == False
    
    def test_validate_input_empty_dataframe(self):
        """Test input validation with empty DataFrame."""
        data = pd.DataFrame()
        assert self.calculator.validate_input(data) == False
    
    def test_calculate_basic_errors(self):
        """Test basic error calculation with synthetic data."""
        data = self.create_synthetic_error_data()
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        # Check that all error columns are present
        assert 'gps_error_x' in errors.columns
        assert 'gps_error_y' in errors.columns
        assert 'gps_error_z' in errors.columns
        assert 'gps_error_norm' in errors.columns
        
        # Check error values (should be constant for synthetic data)
        np.testing.assert_array_almost_equal(errors['gps_error_x'], 1.0, decimal=6)
        np.testing.assert_array_almost_equal(errors['gps_error_y'], 2.0, decimal=6)
        np.testing.assert_array_almost_equal(errors['gps_error_z'], -0.5, decimal=6)
        
        # Check error magnitude calculation
        expected_norm = np.sqrt(1.0**2 + 2.0**2 + (-0.5)**2)
        np.testing.assert_array_almost_equal(errors['gps_error_norm'], expected_norm, decimal=6)
    
    def test_calculate_mismatched_lengths(self):
        """Test error calculation with mismatched data lengths."""
        data = self.create_test_data(num_samples=50)
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']].iloc[:40]  # Shorter
        
        with pytest.raises(ValueError, match="GPS data and ground truth must have same length"):
            self.calculator.calculate(gps_data, ground_truth)
    
    def test_process_method(self):
        """Test the main process method."""
        data = self.create_test_data()
        
        result = self.calculator.process(data)
        
        # Should contain all original columns plus error columns
        assert len(result.columns) >= len(data.columns)
        assert 'gps_error_x' in result.columns
        assert 'gps_error_y' in result.columns
        assert 'gps_error_z' in result.columns
        assert 'gps_error_norm' in result.columns
        
        # Should have same number of rows
        assert len(result) == len(data)
        
        # Check that errors are reasonable (not NaN or infinite)
        assert not result['gps_error_x'].isna().any()
        assert not result['gps_error_y'].isna().any()
        assert not result['gps_error_z'].isna().any()
        assert not result['gps_error_norm'].isna().any()
        assert np.isfinite(result['gps_error_norm']).all()
    
    def test_temporal_consistency_outlier_detection(self):
        """Test temporal consistency with outliers."""
        data = self.create_test_data(num_samples=50, add_noise=False)
        
        # Add a single large outlier
        data.loc[25, 'gps_x'] += 20.0  # 20m error spike
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        # The outlier should be detected and handled
        assert errors['gps_error_norm'].max() <= self.calculator.max_error_threshold
    
    def test_compute_error_statistics_basic(self):
        """Test basic error statistics computation."""
        data = self.create_synthetic_error_data()
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        stats = self.calculator.compute_error_statistics(errors)
        
        # Check that basic statistics are present
        assert 'gps_error_x_mean' in stats
        assert 'gps_error_y_mean' in stats
        assert 'gps_error_z_mean' in stats
        assert 'gps_error_norm_mean' in stats
        
        assert 'gps_error_norm_std' in stats
        assert 'gps_error_norm_min' in stats
        assert 'gps_error_norm_max' in stats
        assert 'gps_error_norm_median' in stats
        
        # Check percentile statistics
        assert 'error_p50' in stats
        assert 'error_p90' in stats
        assert 'error_p95' in stats
        assert 'error_p99' in stats
        
        # Check error bounds analysis
        assert 'errors_under_1m' in stats
        assert 'errors_under_2m' in stats
        assert 'errors_under_5m' in stats
        assert 'errors_over_10m' in stats
        
        # Verify values for synthetic data
        expected_norm = np.sqrt(1.0**2 + 2.0**2 + (-0.5)**2)
        assert abs(stats['gps_error_norm_mean'] - expected_norm) < 0.01
        assert abs(stats['gps_error_x_mean'] - 1.0) < 0.01
        assert abs(stats['gps_error_y_mean'] - 2.0) < 0.01
        assert abs(stats['gps_error_z_mean'] - (-0.5)) < 0.01
    
    def test_compute_error_statistics_empty_data(self):
        """Test error statistics with empty data."""
        empty_errors = pd.DataFrame()
        stats = self.calculator.compute_error_statistics(empty_errors)
        assert stats == {}
    
    def test_temporal_consistency_metrics(self):
        """Test temporal consistency metrics calculation."""
        data = self.create_test_data(num_samples=50)
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        metrics = self.calculator._compute_temporal_consistency_metrics(errors)
        
        # Check that temporal metrics are present
        assert 'error_rate_of_change_mean' in metrics
        assert 'error_rate_of_change_std' in metrics
        assert 'error_rate_of_change_max' in metrics
        assert 'sudden_error_jumps' in metrics
        assert 'sudden_jump_rate' in metrics
        assert 'error_autocorr_lag1' in metrics
        
        # Values should be reasonable
        assert metrics['error_rate_of_change_mean'] >= 0
        assert metrics['sudden_error_jumps'] >= 0
        assert metrics['sudden_jump_rate'] >= 0
        assert -1 <= metrics['error_autocorr_lag1'] <= 1
    
    def test_validate_error_quality_good_data(self):
        """Test error quality validation with good data."""
        # Create data with small, consistent errors
        data = self.create_test_data(num_samples=100, add_noise=False)
        # Add small consistent errors
        data['gps_x'] += 0.5  # 0.5m error
        data['gps_y'] += 0.3  # 0.3m error
        data['gps_z'] += 0.2  # 0.2m error
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        is_valid, messages = self.calculator.validate_error_quality(errors)
        
        assert is_valid == True
        assert 'status' in messages
        assert messages['status'] == "Error quality validation passed"
    
    def test_validate_error_quality_high_mean_error(self):
        """Test error quality validation with high mean error."""
        data = self.create_test_data(num_samples=100, add_noise=False)
        # Add large consistent errors
        data['gps_x'] += 10.0  # 10m error (exceeds default threshold of 5m)
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        is_valid, messages = self.calculator.validate_error_quality(errors)
        
        assert is_valid == False
        assert 'high_mean_error' in messages
    
    def test_validate_error_quality_high_p95_error(self):
        """Test error quality validation with high 95th percentile error."""
        data = self.create_test_data(num_samples=100, add_noise=False)
        
        # Add large errors to more samples to ensure they survive outlier capping
        data.loc[90:99, 'gps_x'] += 25.0  # Large errors in last 10% of data
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        is_valid, messages = self.calculator.validate_error_quality(
            errors, max_p95_error=10.0  # Lower threshold for this test
        )
        
        assert is_valid == False
        assert 'high_p95_error' in messages
    
    def test_validate_error_quality_empty_data(self):
        """Test error quality validation with empty data."""
        empty_errors = pd.DataFrame()
        
        is_valid, messages = self.calculator.validate_error_quality(empty_errors)
        
        assert is_valid == False
        assert 'empty_data' in messages
    
    def test_error_magnitude_calculation(self):
        """Test error magnitude calculation accuracy."""
        # Create simple test case with known values
        gps_data = pd.DataFrame({
            'gps_x': [3.0, 0.0, 4.0],
            'gps_y': [4.0, 3.0, 0.0],
            'gps_z': [0.0, 4.0, 3.0]
        })
        
        ground_truth = pd.DataFrame({
            'ground_truth_x': [0.0, 0.0, 0.0],
            'ground_truth_y': [0.0, 0.0, 0.0],
            'ground_truth_z': [0.0, 0.0, 0.0]
        })
        
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        # Expected magnitudes: sqrt(3²+4²+0²)=5, sqrt(0²+3²+4²)=5, sqrt(4²+0²+3²)=5
        expected_magnitudes = [5.0, 5.0, 5.0]
        np.testing.assert_array_almost_equal(errors['gps_error_norm'], expected_magnitudes, decimal=6)
    
    def test_error_calculation_with_nan_values(self):
        """Test error calculation handling of NaN values."""
        data = self.create_test_data(num_samples=20)
        
        # Introduce some NaN values
        data.loc[5, 'gps_x'] = np.nan
        data.loc[10, 'ground_truth_y'] = np.nan
        
        result = self.calculator.process(data)
        
        # Should handle NaN values gracefully
        assert len(result) == len(data)
        # NaN in input should propagate to error calculations
        assert pd.isna(result.loc[5, 'gps_error_x'])
        assert pd.isna(result.loc[10, 'gps_error_y'])
    
    def test_extreme_outlier_capping(self):
        """Test that extreme outliers are capped at threshold."""
        data = self.create_test_data(num_samples=20, add_noise=False)
        
        # Add extreme outlier that exceeds max_error_threshold
        data.loc[10, 'gps_x'] += 100.0  # 100m error (exceeds 50m threshold)
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        # Error magnitude should be capped at threshold
        assert errors['gps_error_norm'].max() <= self.calculator.max_error_threshold
        
        # The capped error should be exactly at the threshold
        max_error_idx = errors['gps_error_norm'].idxmax()
        assert abs(errors.loc[max_error_idx, 'gps_error_norm'] - self.calculator.max_error_threshold) < 0.01
    
    def test_error_statistics_percentiles(self):
        """Test error statistics percentile calculations."""
        # Create data with known distribution
        num_samples = 100
        data = self.create_test_data(num_samples=num_samples, add_noise=False)
        
        # Create linearly increasing errors from 0 to 10m
        error_values = np.linspace(0, 10, num_samples)
        data['gps_x'] = data['ground_truth_x'] + error_values
        data['gps_y'] = data['ground_truth_y']  # No y error
        data['gps_z'] = data['ground_truth_z']  # No z error
        
        gps_data = data[['gps_x', 'gps_y', 'gps_z']]
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']]
        errors = self.calculator.calculate(gps_data, ground_truth)
        
        stats = self.calculator.compute_error_statistics(errors)
        
        # Check percentiles (should match the linear distribution)
        assert abs(stats['error_p50'] - 5.0) < 0.2  # 50th percentile ~5m
        assert abs(stats['error_p90'] - 9.0) < 0.2  # 90th percentile ~9m
        assert abs(stats['error_p95'] - 9.5) < 0.2  # 95th percentile ~9.5m
        
        # Check error bounds
        assert stats['errors_under_1m'] > 0  # Some errors under 1m
        assert stats['errors_under_5m'] > 40  # More than 40% under 5m
        assert stats['errors_over_10m'] < 10  # Less than 10% over 10m