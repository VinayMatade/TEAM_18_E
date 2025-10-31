"""
Tests for motion classification system.

Tests motion detection algorithms, segment classification, and threshold sensitivity.
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, List

from uav_log_processor.processors.motion_classifier import MotionClassifier


class TestMotionClassifier(unittest.TestCase):
    """Test motion classification functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'accel_threshold': 0.5,  # m/s²
            'gyro_threshold': 0.1,   # rad/s
            'window_size_seconds': 2.0,  # Shorter for testing
            'min_segment_duration': 1.0  # Shorter for testing
        }
        self.classifier = MotionClassifier(self.config)
    
    def test_calculate_acceleration_magnitude(self):
        """Test acceleration magnitude calculation."""
        # Create test data with known acceleration values
        data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0],
            'imu_ax': [0.0, 1.0, 0.0, 2.0],
            'imu_ay': [0.0, 0.0, 1.0, 0.0],
            'imu_az': [9.81, 9.81, 9.81, 9.81]  # Gravity
        })
        
        magnitude = self.classifier.calculate_acceleration_magnitude(data)
        
        # Should calculate motion magnitude combining horizontal and vertical components
        # horizontal: sqrt(ax² + ay²), vertical: |az - 9.81|, total: sqrt(horizontal² + vertical²)
        expected = np.array([0.0, 1.0, 1.0, 2.0])  # sqrt((sqrt(ax²+ay²))² + (|az-9.81|)²)
        
        self.assertEqual(len(magnitude), len(data))
        np.testing.assert_array_almost_equal(magnitude.values, expected, decimal=2)
    
    def test_calculate_gyroscope_magnitude(self):
        """Test gyroscope magnitude calculation."""
        # Create test data with known gyroscope values
        data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0],
            'imu_gx': [0.0, 0.1, 0.0, 0.3],
            'imu_gy': [0.0, 0.0, 0.1, 0.4],
            'imu_gz': [0.0, 0.0, 0.0, 0.0]
        })
        
        magnitude = self.classifier.calculate_gyroscope_magnitude(data)
        
        # Should calculate magnitude: sqrt(gx² + gy² + gz²)
        expected = np.array([0.0, 0.1, 0.1, 0.5])
        
        self.assertEqual(len(magnitude), len(data))
        np.testing.assert_array_almost_equal(magnitude.values, expected, decimal=2)
    
    def test_apply_sliding_window_smoothing(self):
        """Test sliding window smoothing for noise reduction."""
        # Create noisy signal
        np.random.seed(42)
        signal = pd.Series([1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0])
        data = pd.DataFrame({
            'timestamp': np.arange(0, 0.7, 0.1)  # 10 Hz
        })
        
        smoothed = self.classifier.apply_sliding_window_smoothing(signal, data)
        
        # Smoothed signal should have less variation
        self.assertEqual(len(smoothed), len(signal))
        self.assertLess(smoothed.std(), signal.std())
    
    def test_classify_stationary_motion(self):
        """Test classification of stationary motion."""
        # Create data representing stationary drone
        data = pd.DataFrame({
            'timestamp': np.arange(0, 5, 0.1),  # 5 seconds at 10 Hz
            'imu_ax': np.random.normal(0, 0.1, 50),  # Low acceleration noise
            'imu_ay': np.random.normal(0, 0.1, 50),
            'imu_az': np.random.normal(9.81, 0.1, 50),  # Gravity + noise
            'imu_gx': np.random.normal(0, 0.02, 50),  # Low gyro noise
            'imu_gy': np.random.normal(0, 0.02, 50),
            'imu_gz': np.random.normal(0, 0.02, 50)
        })
        
        motion_labels = self.classifier.classify(data)
        
        # Most samples should be classified as stationary
        stationary_percentage = (motion_labels == 'stationary').mean()
        self.assertGreater(stationary_percentage, 0.7)  # At least 70% stationary
    
    def test_classify_moving_motion(self):
        """Test classification of moving motion."""
        # Create data representing moving drone
        data = pd.DataFrame({
            'timestamp': np.arange(0, 5, 0.1),  # 5 seconds at 10 Hz
            'imu_ax': np.random.normal(2.0, 0.5, 50),  # High acceleration
            'imu_ay': np.random.normal(1.5, 0.5, 50),
            'imu_az': np.random.normal(9.81, 0.5, 50),
            'imu_gx': np.random.normal(0.5, 0.2, 50),  # High gyro values
            'imu_gy': np.random.normal(0.3, 0.2, 50),
            'imu_gz': np.random.normal(0.2, 0.2, 50)
        })
        
        motion_labels = self.classifier.classify(data)
        
        # Most samples should be classified as moving
        moving_percentage = (motion_labels == 'moving').mean()
        self.assertGreater(moving_percentage, 0.7)  # At least 70% moving
    
    def test_threshold_sensitivity(self):
        """Test sensitivity to threshold changes."""
        # Create borderline data
        data = pd.DataFrame({
            'timestamp': np.arange(0, 3, 0.1),
            'imu_ax': np.full(30, 0.4),  # Just below default threshold
            'imu_ay': np.full(30, 0.0),
            'imu_az': np.full(30, 9.81),
            'imu_gx': np.full(30, 0.08),  # Just below default threshold
            'imu_gy': np.full(30, 0.0),
            'imu_gz': np.full(30, 0.0)
        })
        
        # Test with default thresholds
        labels_default = self.classifier.classify(data)
        
        # Test with stricter thresholds
        strict_classifier = MotionClassifier({
            'accel_threshold': 0.3,
            'gyro_threshold': 0.05,
            'window_size_seconds': 2.0,
            'min_segment_duration': 1.0
        })
        labels_strict = strict_classifier.classify(data)
        
        # Stricter thresholds should result in more "moving" classifications
        stationary_default = (labels_default == 'stationary').sum()
        stationary_strict = (labels_strict == 'stationary').sum()
        
        self.assertLessEqual(stationary_strict, stationary_default)
    
    def test_get_stationary_segments(self):
        """Test stationary segment detection."""
        # Create data with known stationary periods (longer duration to meet minimum)
        motion_labels = pd.Series([
            'moving', 'moving', 'stationary', 'stationary', 'stationary',
            'stationary', 'stationary', 'stationary', 'stationary', 'stationary',
            'moving', 'stationary', 'stationary', 'stationary', 'stationary'
        ])
        data = pd.DataFrame({
            'timestamp': np.arange(0, 1.5, 0.1)  # 15 samples, 0.1s apart
        })
        
        segments = self.classifier.get_stationary_segments(motion_labels, data)
        
        # Should find stationary segments
        self.assertGreater(len(segments), 0)
        
        # Check segment structure
        for start_idx, end_idx in segments:
            self.assertIsInstance(start_idx, int)
            self.assertIsInstance(end_idx, int)
            self.assertLessEqual(start_idx, end_idx)
            
            # Verify these indices correspond to stationary labels
            for i in range(start_idx, end_idx + 1):
                if i < len(motion_labels):
                    self.assertEqual(motion_labels.iloc[i], 'stationary')
    
    def test_segment_boundary_detection(self):
        """Test detection of segment boundaries."""
        # Create data with clear transitions
        motion_labels = pd.Series([
            'stationary', 'stationary', 'moving', 'moving', 'moving',
            'stationary', 'stationary', 'moving', 'moving', 'stationary'
        ])
        data = pd.DataFrame({
            'timestamp': np.arange(0, 1.0, 0.1)
        })
        
        segments = self.classifier.get_motion_segments(motion_labels, data)
        
        # Should detect multiple segments
        self.assertGreater(len(segments), 2)
        
        # Check segment continuity
        for i in range(len(segments) - 1):
            current_end = segments[i]['end_index']
            next_start = segments[i + 1]['start_index']
            self.assertEqual(current_end + 1, next_start)
    
    def test_detect_transitions(self):
        """Test transition detection between states."""
        motion_labels = pd.Series([
            'stationary', 'stationary', 'moving', 'moving', 'stationary'
        ])
        data = pd.DataFrame({
            'timestamp': [0.0, 0.1, 0.2, 0.3, 0.4]
        })
        
        transitions = self.classifier.detect_transitions(motion_labels, data)
        
        # Should detect 2 transitions: stationary->moving and moving->stationary
        self.assertEqual(len(transitions), 2)
        
        # Check transition details
        self.assertEqual(transitions[0]['from_state'], 'stationary')
        self.assertEqual(transitions[0]['to_state'], 'moving')
        self.assertEqual(transitions[1]['from_state'], 'moving')
        self.assertEqual(transitions[1]['to_state'], 'stationary')
    
    def test_classification_statistics(self):
        """Test motion classification statistics calculation."""
        motion_labels = pd.Series([
            'stationary', 'stationary', 'moving', 'moving', 'moving',
            'stationary', 'moving', 'moving', 'stationary', 'stationary'
        ])
        data = pd.DataFrame({
            'timestamp': np.arange(0, 1.0, 0.1)
        })
        
        stats = self.classifier.get_classification_statistics(motion_labels, data)
        
        # Check basic statistics
        self.assertEqual(stats['total_samples'], 10)
        self.assertEqual(stats['stationary_samples'], 5)
        self.assertEqual(stats['moving_samples'], 5)
        self.assertAlmostEqual(stats['stationary_percentage'], 50.0)
        self.assertAlmostEqual(stats['moving_percentage'], 50.0)
        
        # Check duration statistics
        self.assertIn('total_duration_seconds', stats)
        self.assertIn('num_stationary_segments', stats)
        self.assertIn('num_moving_segments', stats)
    
    def test_process_method(self):
        """Test the main process method."""
        data = pd.DataFrame({
            'timestamp': np.arange(0, 2, 0.1),
            'imu_ax': np.random.normal(0, 0.1, 20),
            'imu_ay': np.random.normal(0, 0.1, 20),
            'imu_az': np.random.normal(9.81, 0.1, 20),
            'imu_gx': np.random.normal(0, 0.02, 20),
            'imu_gy': np.random.normal(0, 0.02, 20),
            'imu_gz': np.random.normal(0, 0.02, 20)
        })
        
        result = self.classifier.process(data)
        
        # Should return DataFrame with motion labels added
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('motion_label', result.columns)
        self.assertEqual(len(result), len(data))
        
        # All original columns should be preserved
        for col in data.columns:
            self.assertIn(col, result.columns)
    
    def test_prefixed_column_handling(self):
        """Test handling of prefixed IMU columns from synchronization."""
        # Simulate data from synchronizer with prefixed columns
        data = pd.DataFrame({
            'timestamp': np.arange(0, 2, 0.1),
            'stream1_imu_ax': np.random.normal(0, 0.1, 20),
            'stream1_imu_ay': np.random.normal(0, 0.1, 20),
            'stream1_imu_az': np.random.normal(9.81, 0.1, 20),
            'stream1_imu_gx': np.random.normal(0, 0.02, 20),
            'stream1_imu_gy': np.random.normal(0, 0.02, 20),
            'stream1_imu_gz': np.random.normal(0, 0.02, 20)
        })
        
        motion_labels = self.classifier.classify(data)
        
        # Should handle prefixed columns correctly
        self.assertEqual(len(motion_labels), len(data))
        self.assertTrue(all(label in ['stationary', 'moving'] for label in motion_labels))
    
    def test_missing_imu_data_handling(self):
        """Test handling of missing IMU data."""
        # Data with missing IMU columns
        data = pd.DataFrame({
            'timestamp': np.arange(0, 2, 0.1),
            'gps_lat': np.full(20, 37.7749),
            'gps_lon': np.full(20, -122.4194)
        })
        
        # Should handle gracefully
        self.assertFalse(self.classifier.validate_input(data))
        
        # Classification should still work but return zeros
        motion_labels = self.classifier.classify(data)
        self.assertEqual(len(motion_labels), len(data))
    
    def test_known_stationary_periods(self):
        """Test with known stationary periods."""
        # Create data with clear stationary period in the middle
        timestamps = np.arange(0, 10, 0.1)  # 10 seconds
        n_samples = len(timestamps)
        
        # High motion at start and end, low motion in middle
        accel_pattern = np.concatenate([
            np.full(30, 2.0),      # Moving (0-3s)
            np.full(40, 0.1),      # Stationary (3-7s)
            np.full(30, 2.0)       # Moving (7-10s)
        ])
        
        gyro_pattern = np.concatenate([
            np.full(30, 0.5),      # Moving
            np.full(40, 0.02),     # Stationary
            np.full(30, 0.5)       # Moving
        ])
        
        data = pd.DataFrame({
            'timestamp': timestamps,
            'imu_ax': accel_pattern + np.random.normal(0, 0.05, n_samples),
            'imu_ay': np.random.normal(0, 0.05, n_samples),
            'imu_az': np.random.normal(9.81, 0.05, n_samples),
            'imu_gx': gyro_pattern + np.random.normal(0, 0.01, n_samples),
            'imu_gy': np.random.normal(0, 0.01, n_samples),
            'imu_gz': np.random.normal(0, 0.01, n_samples)
        })
        
        motion_labels = self.classifier.classify(data)
        
        # Check that middle section is mostly stationary
        middle_start = 30  # 3 seconds
        middle_end = 70    # 7 seconds
        middle_labels = motion_labels.iloc[middle_start:middle_end]
        
        stationary_in_middle = (middle_labels == 'stationary').mean()
        self.assertGreater(stationary_in_middle, 0.6)  # At least 60% stationary in middle
    
    def test_empty_data_handling(self):
        """Test handling of empty data."""
        empty_data = pd.DataFrame()
        
        motion_labels = self.classifier.classify(empty_data)
        self.assertTrue(motion_labels.empty)
        
        segments = self.classifier.get_stationary_segments(motion_labels, empty_data)
        self.assertEqual(len(segments), 0)


if __name__ == '__main__':
    unittest.main()