"""
Tests for ground truth generation processor.

Tests anchor point calculation, IMU velocity integration, and sensor fusion algorithms.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import logging

from uav_log_processor.processors.ground_truth_generator import GroundTruthGenerator


class TestGroundTruthGenerator:
    """Test cases for GroundTruthGenerator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'rtk_fix_types': [4, 5, 6],
            'high_confidence_hdop_threshold': 1.0,
            'anchor_validation_threshold': 5.0,  # More lenient for tests
            'min_anchor_samples': 10,
            'max_velocity_threshold': 50.0,
            'integration_method': 'trapezoidal',
            'drift_correction_method': 'linear',
            'smoothing_window': 5,
            'fusion_method': 'complementary',
            'complementary_alpha': 0.98
        }
        self.generator = GroundTruthGenerator(self.config)
    
    def create_test_data(self, num_samples=100, with_motion_labels=True):
        """Create test data for ground truth generation."""
        timestamps = np.linspace(0, 10, num_samples)  # 10 seconds of data
        
        # Create synthetic GPS trajectory (circular motion)
        t = timestamps
        radius = 10.0
        gps_x = radius * np.cos(0.5 * t)
        gps_y = radius * np.sin(0.5 * t)
        gps_z = np.ones_like(t) * 100.0  # Constant altitude
        
        # Add some noise to GPS
        gps_x += np.random.normal(0, 0.5, len(t))
        gps_y += np.random.normal(0, 0.5, len(t))
        gps_z += np.random.normal(0, 0.2, len(t))
        
        # Create synthetic IMU data
        # Acceleration from circular motion: a = v²/r
        velocity = 0.5 * radius  # v = ω * r
        centripetal_accel = velocity**2 / radius
        
        imu_ax = -centripetal_accel * np.cos(0.5 * t) + np.random.normal(0, 0.1, len(t))
        imu_ay = -centripetal_accel * np.sin(0.5 * t) + np.random.normal(0, 0.1, len(t))
        imu_az = np.ones_like(t) * 9.81 + np.random.normal(0, 0.1, len(t))  # Gravity + noise
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'gps_x': gps_x,
            'gps_y': gps_y,
            'gps_z': gps_z,
            'imu_ax': imu_ax,
            'imu_ay': imu_ay,
            'imu_az': imu_az,
            'gps_hdop': np.ones_like(t) * 1.5,  # Moderate GPS quality
            'gps_fix_type': np.ones_like(t) * 3  # 3D fix
        })
        
        if with_motion_labels:
            # Create motion labels (stationary at start and end)
            motion_labels = ['moving'] * len(t)
            motion_labels[:15] = ['stationary'] * 15  # First 15 samples stationary
            motion_labels[-15:] = ['stationary'] * 15  # Last 15 samples stationary
            data['motion_label'] = motion_labels
        
        return data
    
    def test_initialization(self):
        """Test GroundTruthGenerator initialization."""
        generator = GroundTruthGenerator()
        assert generator.config == {}
        
        generator_with_config = GroundTruthGenerator(self.config)
        assert generator_with_config.rtk_fix_types == [4, 5, 6]
        assert generator_with_config.high_confidence_hdop_threshold == 1.0
    
    def test_validate_input_valid_data(self):
        """Test input validation with valid data."""
        data = self.create_test_data()
        assert self.generator.validate_input(data) == True
    
    def test_validate_input_missing_timestamp(self):
        """Test input validation with missing timestamp."""
        data = self.create_test_data()
        data = data.drop('timestamp', axis=1)
        assert self.generator.validate_input(data) == False
    
    def test_validate_input_insufficient_gps_columns(self):
        """Test input validation with insufficient GPS columns."""
        data = self.create_test_data()
        data = data.drop(['gps_x', 'gps_y'], axis=1)
        assert self.generator.validate_input(data) == False
    
    def test_validate_input_insufficient_imu_columns(self):
        """Test input validation with insufficient IMU columns."""
        data = self.create_test_data()
        data = data.drop(['imu_ax', 'imu_ay'], axis=1)
        assert self.generator.validate_input(data) == False
    
    def test_find_gps_columns(self):
        """Test GPS column detection."""
        data = self.create_test_data()
        gps_cols = self.generator._find_gps_columns(data)
        assert len(gps_cols) == 3
        assert 'gps_x' in gps_cols
        assert 'gps_y' in gps_cols
        assert 'gps_z' in gps_cols
    
    def test_find_imu_columns(self):
        """Test IMU column detection."""
        data = self.create_test_data()
        imu_cols = self.generator._find_imu_columns(data)
        assert len(imu_cols) == 3
        assert 'imu_ax' in imu_cols
        assert 'imu_ay' in imu_cols
        assert 'imu_az' in imu_cols
    
    def test_calculate_stationary_anchors_basic(self):
        """Test basic anchor point calculation."""
        # Create data with less noise for more predictable anchors
        data = self.create_controlled_test_data()
        anchors = self.generator.calculate_stationary_anchors(data)
        
        # Should find at least 1 anchor point (may be filtered due to validation)
        assert len(anchors) >= 1
        
        # Check anchor properties
        for anchor in anchors:
            assert 'x' in anchor
            assert 'y' in anchor
            assert 'z' in anchor
            assert 'timestamp' in anchor
            assert 'confidence' in anchor
            assert anchor['sample_count'] >= self.generator.min_anchor_samples
    
    def create_controlled_test_data(self, num_samples=100):
        """Create test data with controlled noise for predictable results."""
        timestamps = np.linspace(0, 10, num_samples)
        
        # Create simple trajectory with low noise
        gps_x = np.zeros_like(timestamps)
        gps_y = np.zeros_like(timestamps)
        gps_z = np.ones_like(timestamps) * 100.0
        
        # Add minimal noise
        gps_x += np.random.normal(0, 0.1, len(timestamps))
        gps_y += np.random.normal(0, 0.1, len(timestamps))
        gps_z += np.random.normal(0, 0.05, len(timestamps))
        
        # Simple IMU data
        imu_ax = np.random.normal(0, 0.05, len(timestamps))
        imu_ay = np.random.normal(0, 0.05, len(timestamps))
        imu_az = np.ones_like(timestamps) * 9.81 + np.random.normal(0, 0.05, len(timestamps))
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'gps_x': gps_x,
            'gps_y': gps_y,
            'gps_z': gps_z,
            'imu_ax': imu_ax,
            'imu_ay': imu_ay,
            'imu_az': imu_az,
            'gps_hdop': np.ones_like(timestamps) * 1.0,  # Good GPS quality
            'gps_fix_type': np.ones_like(timestamps) * 3
        })
        
        # Create motion labels (stationary at start and end)
        motion_labels = ['moving'] * len(timestamps)
        motion_labels[:15] = ['stationary'] * 15
        motion_labels[-15:] = ['stationary'] * 15
        data['motion_label'] = motion_labels
        
        return data
    
    def test_calculate_stationary_anchors_no_motion_labels(self):
        """Test anchor calculation without motion labels."""
        data = self.create_test_data(with_motion_labels=False)
        
        with pytest.raises(ValueError, match="Motion classification required"):
            self.generator.calculate_stationary_anchors(data)
    
    def test_calculate_stationary_anchors_no_stationary_segments(self):
        """Test anchor calculation with no stationary segments."""
        data = self.create_test_data()
        data['motion_label'] = 'moving'  # All moving
        
        anchors = self.generator.calculate_stationary_anchors(data)
        assert len(anchors) == 0
    
    def test_calculate_stationary_anchors_without_validation(self):
        """Test anchor calculation before validation filtering."""
        data = self.create_test_data()
        
        # Get stationary segments directly
        stationary_mask = data['motion_label'] == 'stationary'
        segments = self.generator._find_continuous_segments(stationary_mask, data)
        
        # Should find 2 segments (start and end)
        assert len(segments) == 2
        
        # Each segment should have sufficient duration
        for segment in segments:
            assert segment['duration'] > 0
            assert segment['end_index'] > segment['start_index']
    
    def test_is_high_confidence_segment_rtk(self):
        """Test high confidence detection with RTK fix types."""
        data = pd.DataFrame({
            'gps_fix_type': [4, 5, 6, 4],  # RTK fix types
            'gps_hdop': [2.0, 2.0, 2.0, 2.0]
        })
        
        assert self.generator._is_high_confidence_segment(data) == True
    
    def test_is_high_confidence_segment_low_hdop(self):
        """Test high confidence detection with low HDOP."""
        data = pd.DataFrame({
            'gps_fix_type': [3, 3, 3, 3],  # Regular 3D fix
            'gps_hdop': [0.8, 0.9, 0.7, 0.8]  # Low HDOP
        })
        
        assert self.generator._is_high_confidence_segment(data) == True
    
    def test_is_high_confidence_segment_low_confidence(self):
        """Test high confidence detection with low confidence GPS."""
        data = pd.DataFrame({
            'gps_fix_type': [3, 3, 3, 3],  # Regular 3D fix
            'gps_hdop': [2.5, 3.0, 2.8, 2.7]  # High HDOP
        })
        
        assert self.generator._is_high_confidence_segment(data) == False
    
    def test_integrate_imu_velocity_basic(self):
        """Test basic IMU velocity integration."""
        data = self.create_test_data(num_samples=50)
        
        # Create simple anchor points
        anchors = [
            {'timestamp': 0.0, 'index': 0, 'x': 10.0, 'y': 0.0, 'z': 100.0},
            {'timestamp': 5.0, 'index': 25, 'x': 0.0, 'y': 10.0, 'z': 100.0},
            {'timestamp': 10.0, 'index': 49, 'x': -10.0, 'y': 0.0, 'z': 100.0}
        ]
        
        integrated = self.generator.integrate_imu_velocity(data, anchors)
        
        assert len(integrated) == len(data)
        assert 'x' in integrated.columns
        assert 'y' in integrated.columns
        assert 'z' in integrated.columns
        
        # Check that positions start and end at anchor points
        assert abs(integrated.iloc[0]['x'] - 10.0) < 0.1
        assert abs(integrated.iloc[25]['x'] - 0.0) < 0.1
        assert abs(integrated.iloc[49]['x'] - (-10.0)) < 0.1
    
    def test_integrate_imu_velocity_no_anchors(self):
        """Test IMU integration with no anchor points."""
        data = self.create_test_data(num_samples=50)
        anchors = []
        
        integrated = self.generator.integrate_imu_velocity(data, anchors)
        
        # Should fallback to GPS positions
        assert len(integrated) == len(data)
        np.testing.assert_array_almost_equal(integrated['x'].values, data['gps_x'].values, decimal=1)
    
    def test_integrate_trapezoidal(self):
        """Test trapezoidal integration method."""
        # Simple test: integrate constant acceleration
        acceleration = np.array([1.0, 1.0, 1.0, 1.0])  # 1 m/s² constant
        dt = np.array([0.1, 0.1, 0.1, 0.1])  # 0.1 second intervals
        
        velocity = self.generator._integrate_trapezoidal(acceleration, dt)
        
        # Expected: v = a*t, so v = [0, 0.1, 0.2, 0.3]
        expected = np.array([0.0, 0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(velocity, expected, decimal=2)
    
    def test_apply_drift_correction(self):
        """Test drift correction algorithm."""
        # Create positions with drift
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # Linear trajectory
        y = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # No y movement
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # No z movement
        
        start_pos = [0.0, 0.0, 0.0]
        end_pos = [3.0, 0.0, 0.0]  # Should end at x=3, but we have x=4 (drift=1)
        timestamps = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        
        x_corr, y_corr, z_corr = self.generator._apply_drift_correction(
            x, y, z, start_pos, end_pos, timestamps
        )
        
        # Should correct the drift linearly
        assert abs(x_corr[-1] - 3.0) < 0.01  # End position should be corrected
        assert abs(x_corr[0] - 0.0) < 0.01   # Start position should remain
    
    def test_apply_complementary_filter(self):
        """Test complementary filter sensor fusion."""
        data = self.create_test_data(num_samples=50)
        
        # Create integrated positions (slightly different from GPS)
        integrated = pd.DataFrame({
            'x': data['gps_x'] + 0.1,  # Small offset
            'y': data['gps_y'] + 0.1,
            'z': data['gps_z'] + 0.1
        }, index=data.index)
        
        anchors = []  # No anchors for this test
        
        fused = self.generator._apply_complementary_filter(data, integrated, anchors)
        
        assert len(fused) == len(data)
        assert 'x' in fused.columns
        assert 'y' in fused.columns
        assert 'z' in fused.columns
        
        # Check that fusion produces reasonable results
        # The fused result should be close to GPS (since alpha is high ~0.98)
        # but may have some influence from IMU integration
        x_diff_gps = np.abs(fused['x'] - data['gps_x']).mean()
        x_diff_imu = np.abs(fused['x'] - integrated['x']).mean()
        
        # Fused should be closer to GPS than to pure IMU integration
        assert x_diff_gps < x_diff_imu
    
    def test_generate_full_pipeline(self):
        """Test complete ground truth generation pipeline."""
        data = self.create_test_data(num_samples=100)
        
        result = self.generator.generate(data)
        
        assert len(result) == len(data)
        assert 'ground_truth_x' in result.columns
        assert 'ground_truth_y' in result.columns
        assert 'ground_truth_z' in result.columns
        assert 'timestamp' in result.columns
        
        # Check that ground truth positions are reasonable
        assert not result['ground_truth_x'].isna().any()
        assert not result['ground_truth_y'].isna().any()
        assert not result['ground_truth_z'].isna().any()
        
        # Ground truth should be close to GPS but potentially smoother
        gps_x_std = data['gps_x'].std()
        gt_x_std = result['ground_truth_x'].std()
        # Ground truth might be smoother (lower std) due to filtering
        assert gt_x_std <= gps_x_std * 1.5  # Allow some variation
    
    def test_process_method(self):
        """Test the main process method."""
        data = self.create_test_data()
        
        result = self.generator.process(data)
        
        # Should contain all original columns plus ground truth
        assert len(result.columns) >= len(data.columns)
        assert 'ground_truth_x' in result.columns
        assert 'ground_truth_y' in result.columns
        assert 'ground_truth_z' in result.columns
        
        # Should have same number of rows
        assert len(result) == len(data)
    
    def test_anchor_validation(self):
        """Test anchor point validation and filtering."""
        # Create anchors with one outlier
        anchors = [
            {'x': 0.0, 'y': 0.0, 'z': 100.0, 'position_std': 0.5, 'confidence': 'normal'},
            {'x': 1.0, 'y': 1.0, 'z': 100.0, 'position_std': 0.3, 'confidence': 'high'},
            {'x': 50.0, 'y': 50.0, 'z': 100.0, 'position_std': 0.4, 'confidence': 'normal'},  # Outlier
            {'x': 2.0, 'y': 2.0, 'z': 100.0, 'position_std': 0.6, 'confidence': 'normal'}
        ]
        
        validated = self.generator._validate_anchor_points(anchors)
        
        # Should filter out the outlier
        assert len(validated) < len(anchors)
        
        # Remaining anchors should be consistent
        for anchor in validated:
            assert anchor['position_std'] <= self.generator.anchor_validation_threshold
    
    def test_velocity_limiting(self):
        """Test velocity limiting for unrealistic speeds."""
        # Create positions with unrealistic velocity spike
        x = np.array([0.0, 1.0, 100.0, 3.0, 4.0])  # Spike at index 2
        y = np.array([0.0, 1.0, 1.0, 1.0, 1.0])
        z = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        dt = np.array([1.0, 1.0, 1.0, 1.0])
        
        x_limited, y_limited, z_limited = self.generator._limit_velocities(x, y, z, dt)
        
        # Should smooth out the velocity spike
        assert abs(x_limited[2] - 100.0) > 10.0  # Should be significantly different
        assert x_limited[2] < 50.0  # Should be reduced
    
    def test_cubic_spline_smoothing(self):
        """Test cubic spline smoothing around anchor points."""
        # Create positions with some noise
        positions = pd.DataFrame({
            'x': [0.0, 1.1, 1.9, 3.1, 4.0],
            'y': [0.0, 0.1, -0.1, 0.1, 0.0],
            'z': [0.0, 0.0, 0.0, 0.0, 0.0]
        })
        
        anchors = [{'index': 2, 'x': 2.0, 'y': 0.0, 'z': 0.0}]  # Anchor at middle
        
        smoothed = self.generator._apply_cubic_spline_smoothing(positions, anchors)
        
        assert len(smoothed) == len(positions)
        # Should be smoother around the anchor point
        assert abs(smoothed.iloc[1]['x'] - 1.1) < 0.5  # Some smoothing applied