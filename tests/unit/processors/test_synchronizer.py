"""
Integration tests for data synchronization system.

Tests multi-stream synchronization, coordinate conversion, and data quality.
"""

import unittest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import tempfile
import os

from uav_log_processor.processors.synchronizer import DataSynchronizer
from uav_log_processor.utils.coordinates import CoordinateConverter


class TestDataSynchronizer(unittest.TestCase):
    """Test data synchronization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'target_frequency': 10.0,  # 10 Hz for faster testing
            'interpolation_method': 'linear',
            'max_gap_seconds': 0.5,
            'min_data_threshold': 0.5
        }
        self.synchronizer = DataSynchronizer(self.config)
    
    def test_normalize_timestamps_seconds(self):
        """Test timestamp normalization from seconds."""
        data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0],
            'value': [10, 20, 30, 40]
        })
        
        normalized = self.synchronizer._normalize_timestamps(data)
        
        self.assertTrue(np.allclose(normalized['timestamp'], [1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(normalized['timestamp'].is_monotonic_increasing)
    
    def test_normalize_timestamps_microseconds(self):
        """Test timestamp normalization from microseconds."""
        data = pd.DataFrame({
            'timestamp': [1000000, 2000000, 3000000, 4000000],  # Microseconds
            'value': [10, 20, 30, 40]
        })
        
        normalized = self.synchronizer._normalize_timestamps(data)
        
        
        self.assertTrue(np.allclose(normalized['timestamp'], [1.0, 2.0, 3.0, 4.0]))
    
    def test_normalize_timestamps_datetime(self):
        """Test timestamp normalization from datetime."""
        timestamps = pd.to_datetime(['2023-01-01 12:00:00', '2023-01-01 12:00:01', 
                                   '2023-01-01 12:00:02', '2023-01-01 12:00:03'])
        data = pd.DataFrame({
            'timestamp': timestamps,
            'value': [10, 20, 30, 40]
        })
        
        normalized = self.synchronizer._normalize_timestamps(data)
        
        # Should be 1-second intervals
        time_diffs = np.diff(normalized['timestamp'])
        self.assertTrue(np.allclose(time_diffs, [1.0, 1.0, 1.0]))
    
    def test_find_common_time_range(self):
        """Test finding common time range across streams."""
        stream1 = pd.DataFrame({'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0]})
        stream2 = pd.DataFrame({'timestamp': [2.5, 3.0, 3.5, 4.0, 4.5]})
        stream3 = pd.DataFrame({'timestamp': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]})
        
        streams = {'stream1': stream1, 'stream2': stream2, 'stream3': stream3}
        
        time_range = self.synchronizer._find_common_time_range(streams)
        
        self.assertIsNotNone(time_range)
        start_time, end_time = time_range
        
        # Common range should be from max(start_times) to min(end_times)
        self.assertAlmostEqual(start_time, 2.5)  # max(1.0, 2.5, 0.5)
        self.assertAlmostEqual(end_time, 3.0)    # min(5.0, 4.5, 3.0)
    
    def test_create_time_axis(self):
        """Test uniform time axis creation."""
        start_time = 1.0
        end_time = 3.0
        
        time_axis = self.synchronizer._create_time_axis(start_time, end_time)
        
        # Should have 10 Hz frequency (0.1 second intervals)
        expected_dt = 1.0 / self.config['target_frequency']
        expected_samples = int((end_time - start_time) / expected_dt) + 1
        
        self.assertEqual(len(time_axis), expected_samples)
        self.assertAlmostEqual(time_axis[0], start_time)
        self.assertAlmostEqual(time_axis[-1], start_time + (expected_samples - 1) * expected_dt)
    
    def test_interpolate_to_time_axis(self):
        """Test data interpolation to target time axis."""
        # Create test data with irregular timestamps
        data = pd.DataFrame({
            'timestamp': [1.0, 1.5, 2.2, 3.0],
            'value': [10.0, 15.0, 22.0, 30.0]
        })
        
        # Create uniform time axis
        time_axis = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
        
        interpolated = self.synchronizer._interpolate_to_time_axis(data, time_axis)
        
        self.assertEqual(len(interpolated), len(time_axis))
        self.assertTrue(np.allclose(interpolated['timestamp'], time_axis))
        
        # Check that original values are preserved at exact timestamps
        self.assertAlmostEqual(interpolated.loc[interpolated['timestamp'] == 1.0, 'value'].iloc[0], 10.0)
        self.assertAlmostEqual(interpolated.loc[interpolated['timestamp'] == 2.2, 'value'].iloc[0], 22.0)
        self.assertAlmostEqual(interpolated.loc[interpolated['timestamp'] == 3.0, 'value'].iloc[0], 30.0)
    
    def test_detect_data_gaps(self):
        """Test data gap detection."""
        # Create data with gaps
        data = pd.DataFrame({
            'timestamp': [1.0, 1.1, 1.2, 2.0, 2.1, 3.5, 3.6],  # Gaps at 1.2->2.0 and 2.1->3.5
            'value': [10, 11, 12, 20, 21, 35, 36]
        })
        
        gaps = self.synchronizer.detect_data_gaps(data, max_gap_seconds=0.5)
        
        self.assertEqual(len(gaps), 2)
        
        # First gap: 1.2 to 2.0 (0.8 seconds)
        self.assertAlmostEqual(gaps[0][0], 1.2)
        self.assertAlmostEqual(gaps[0][1], 2.0)
        
        # Second gap: 2.1 to 3.5 (1.4 seconds)
        self.assertAlmostEqual(gaps[1][0], 2.1)
        self.assertAlmostEqual(gaps[1][1], 3.5)
    
    def test_synchronize_streams_basic(self):
        """Test basic multi-stream synchronization."""
        # Create test streams with different sampling rates and time ranges
        stream1 = pd.DataFrame({
            'timestamp': [1.0, 1.5, 2.0, 2.5, 3.0],
            'gps_lat': [37.123, 37.124, 37.125, 37.126, 37.127],
            'gps_lon': [-122.123, -122.124, -122.125, -122.126, -122.127],
            'gps_alt': [100.0, 101.0, 102.0, 103.0, 104.0]
        })
        
        stream2 = pd.DataFrame({
            'timestamp': [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            'imu_ax': [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18],
            'imu_ay': [0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28],
            'imu_az': [9.8, 9.81, 9.82, 9.83, 9.84, 9.85, 9.86, 9.87, 9.88]
        })
        
        data_streams = {'gps': stream1, 'imu': stream2}
        
        synchronized = self.synchronizer.synchronize_streams(data_streams)
        
        # Check basic properties
        self.assertIsInstance(synchronized, pd.DataFrame)
        self.assertGreater(len(synchronized), 0)
        self.assertIn('timestamp', synchronized.columns)
        
        # Check that data from both streams is present
        gps_cols = [col for col in synchronized.columns if col.startswith('gps_')]
        imu_cols = [col for col in synchronized.columns if col.startswith('imu_')]
        
        self.assertGreater(len(gps_cols), 0)
        self.assertGreater(len(imu_cols), 0)
        
        # Check timestamp uniformity (should be at target frequency)
        time_diffs = np.diff(synchronized['timestamp'])
        expected_dt = 1.0 / self.config['target_frequency']
        self.assertTrue(np.allclose(time_diffs, expected_dt, rtol=0.01))
    
    def test_synchronize_streams_with_missing_data(self):
        """Test synchronization with missing data handling."""
        # Create streams with NaN values
        stream1 = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0],
            'gps_lat': [37.123, np.nan, 37.125, 37.126],
            'gps_lon': [-122.123, -122.124, np.nan, -122.126]
        })
        
        stream2 = pd.DataFrame({
            'timestamp': [1.5, 2.5, 3.5, 4.5],
            'imu_ax': [0.1, 0.2, np.nan, 0.4],
            'imu_ay': [np.nan, 0.21, 0.22, 0.23]
        })
        
        data_streams = {'gps': stream1, 'imu': stream2}
        
        synchronized = self.synchronizer.synchronize_streams(data_streams)
        
        # Should handle missing data gracefully
        self.assertIsInstance(synchronized, pd.DataFrame)
        self.assertGreater(len(synchronized), 0)
        
        # Check that some data is preserved
        self.assertTrue(synchronized.notna().any().any())
    
    def test_resample_with_quality_check(self):
        """Test resampling with quality metrics."""
        # Create test data at 20 Hz
        timestamps = np.arange(0, 5, 0.05)  # 5 seconds at 20 Hz
        data = pd.DataFrame({
            'timestamp': timestamps,
            'value': np.sin(timestamps)  # Sine wave for testing
        })
        
        resampled, quality_metrics = self.synchronizer.resample_with_quality_check(data, target_freq=10.0)
        
        # Check quality metrics
        self.assertIn('original_samples', quality_metrics)
        self.assertIn('resampled_samples', quality_metrics)
        self.assertIn('target_frequency_hz', quality_metrics)
        self.assertIn('actual_frequency_hz', quality_metrics)
        
        self.assertEqual(quality_metrics['original_samples'], len(data))
        self.assertAlmostEqual(quality_metrics['target_frequency_hz'], 10.0)
        self.assertAlmostEqual(quality_metrics['actual_frequency_hz'], 10.0, places=0)
        
        # Check resampled data
        self.assertIsInstance(resampled, pd.DataFrame)
        self.assertGreater(len(resampled), 0)


class TestCoordinateConverter(unittest.TestCase):
    """Test coordinate conversion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = CoordinateConverter()
        
        # Test coordinates (San Francisco area)
        self.test_lat = 37.7749
        self.test_lon = -122.4194
        self.test_alt = 100.0
        
        self.home_lat = 37.7750
        self.home_lon = -122.4195
        self.home_alt = 50.0
    
    def test_calculate_home_point_first_valid(self):
        """Test home point calculation using first valid method."""
        gps_data = pd.DataFrame({
            'gps_lat': [np.nan, 37.123, 37.124, 37.125],
            'gps_lon': [np.nan, -122.123, -122.124, -122.125],
            'gps_alt': [np.nan, 100.0, 101.0, 102.0]
        })
        
        home_lat, home_lon, home_alt = self.converter.calculate_home_point(gps_data, method='first_valid')
        
        self.assertAlmostEqual(home_lat, 37.123)
        self.assertAlmostEqual(home_lon, -122.123)
        self.assertAlmostEqual(home_alt, 100.0)
    
    def test_calculate_home_point_mean(self):
        """Test home point calculation using mean method."""
        gps_data = pd.DataFrame({
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.123, -122.124, -122.125],
            'gps_alt': [100.0, 101.0, 102.0]
        })
        
        home_lat, home_lon, home_alt = self.converter.calculate_home_point(gps_data, method='mean')
        
        self.assertAlmostEqual(home_lat, 37.124)
        self.assertAlmostEqual(home_lon, -122.124)
        self.assertAlmostEqual(home_alt, 101.0)
    
    def test_wgs84_to_enu_single_point(self):
        """Test WGS84 to ENU conversion for single point."""
        east, north, up = self.converter.wgs84_to_enu(
            self.test_lat, self.test_lon, self.test_alt,
            self.home_lat, self.home_lon, self.home_alt
        )
        
        # Should return numpy arrays
        self.assertIsInstance(east, np.ndarray)
        self.assertIsInstance(north, np.ndarray)
        self.assertIsInstance(up, np.ndarray)
        
        # Check that conversion produces reasonable values
        # Small displacement should result in small ENU coordinates
        east_val = east.item() if east.ndim == 0 else east[0]
        north_val = north.item() if north.ndim == 0 else north[0]
        up_val = up.item() if up.ndim == 0 else up[0]
        
        self.assertLess(abs(east_val), 1000)  # Less than 1 km
        self.assertLess(abs(north_val), 1000)
        self.assertAlmostEqual(up_val, self.test_alt - self.home_alt, places=1)
    
    def test_wgs84_to_enu_array(self):
        """Test WGS84 to ENU conversion for array of points."""
        lats = np.array([37.7749, 37.7750, 37.7751])
        lons = np.array([-122.4194, -122.4195, -122.4196])
        alts = np.array([100.0, 101.0, 102.0])
        
        east, north, up = self.converter.wgs84_to_enu(
            lats, lons, alts,
            self.home_lat, self.home_lon, self.home_alt
        )
        
        # Should return arrays of same length
        self.assertEqual(len(east), len(lats))
        self.assertEqual(len(north), len(lats))
        self.assertEqual(len(up), len(lats))
        
        # Check that arrays are properly ordered
        self.assertTrue(np.all(np.isfinite(east)))
        self.assertTrue(np.all(np.isfinite(north)))
        self.assertTrue(np.all(np.isfinite(up)))
    
    def test_enu_to_wgs84_roundtrip(self):
        """Test ENU to WGS84 conversion roundtrip accuracy."""
        # Convert to ENU
        east, north, up = self.converter.wgs84_to_enu(
            self.test_lat, self.test_lon, self.test_alt,
            self.home_lat, self.home_lon, self.home_alt
        )
        
        # Convert back to WGS84
        lat_back, lon_back, alt_back = self.converter.enu_to_wgs84(
            east, north, up,
            self.home_lat, self.home_lon, self.home_alt
        )
        
        # Should be close to original values
        self.assertAlmostEqual(lat_back.item() if lat_back.ndim == 0 else lat_back[0], self.test_lat, places=6)
        self.assertAlmostEqual(lon_back.item() if lon_back.ndim == 0 else lon_back[0], self.test_lon, places=6)
        self.assertAlmostEqual(alt_back.item() if alt_back.ndim == 0 else alt_back[0], self.test_alt, places=1)
    
    def test_convert_dataframe_to_enu(self):
        """Test DataFrame coordinate conversion."""
        gps_data = pd.DataFrame({
            'gps_lat': [37.7749, 37.7750, 37.7751],
            'gps_lon': [-122.4194, -122.4195, -122.4196],
            'gps_alt': [100.0, 101.0, 102.0],
            'other_data': [1, 2, 3]
        })
        
        converted = self.converter.convert_dataframe_to_enu(gps_data)
        
        # Should have ENU columns added
        self.assertIn('enu_x', converted.columns)
        self.assertIn('enu_y', converted.columns)
        self.assertIn('enu_z', converted.columns)
        
        # Should preserve original columns
        self.assertIn('gps_lat', converted.columns)
        self.assertIn('other_data', converted.columns)
        
        # Should have home point in metadata
        self.assertIn('home_point', converted.attrs)
        
        # Check that ENU values are reasonable
        self.assertTrue(converted['enu_x'].notna().all())
        self.assertTrue(converted['enu_y'].notna().all())
        self.assertTrue(converted['enu_z'].notna().all())
    
    def test_validate_coordinates(self):
        """Test coordinate validation."""
        # Valid coordinates
        self.assertTrue(self.converter._validate_coordinates(37.7749, -122.4194, 100.0))
        
        # Invalid latitude
        self.assertFalse(self.converter._validate_coordinates(91.0, -122.4194, 100.0))
        self.assertFalse(self.converter._validate_coordinates(-91.0, -122.4194, 100.0))
        
        # Invalid longitude
        self.assertFalse(self.converter._validate_coordinates(37.7749, 181.0, 100.0))
        self.assertFalse(self.converter._validate_coordinates(37.7749, -181.0, 100.0))
        
        # Invalid altitude
        self.assertFalse(self.converter._validate_coordinates(37.7749, -122.4194, 60000.0))
        
        # NaN values
        self.assertFalse(self.converter._validate_coordinates(np.nan, -122.4194, 100.0))


class TestSynchronizerIntegration(unittest.TestCase):
    """Integration tests for synchronizer with coordinate conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synchronizer = DataSynchronizer({
            'target_frequency': 5.0,  # 5 Hz for testing
            'max_gap_seconds': 1.0
        })
    
    def test_synchronize_with_coordinate_conversion(self):
        """Test full synchronization pipeline with coordinate conversion."""
        # Create GPS stream with better overlap
        gps_stream = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0],
            'gps_lat': [37.7749, 37.7750, 37.7751, 37.7752, 37.7753],
            'gps_lon': [-122.4194, -122.4195, -122.4196, -122.4197, -122.4198],
            'gps_alt': [100.0, 101.0, 102.0, 103.0, 104.0],
            'hdop': [1.5, 1.4, 1.3, 1.2, 1.1],
            'fix_type': [3, 3, 3, 3, 3]
        })
        
        # Create IMU stream with better overlap
        imu_stream = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0],  # Same timestamps for better overlap
            'imu_ax': [0.1, 0.2, 0.3, 0.4, 0.5],
            'imu_ay': [0.11, 0.21, 0.31, 0.41, 0.51],
            'imu_az': [9.8, 9.81, 9.82, 9.83, 9.84],
            'imu_gx': [0.01, 0.02, 0.03, 0.04, 0.05],
            'imu_gy': [0.011, 0.021, 0.031, 0.041, 0.051],
            'imu_gz': [0.001, 0.002, 0.003, 0.004, 0.005]
        })
        
        data_streams = {'gps': gps_stream, 'imu': imu_stream}
        
        # Test synchronization with coordinate conversion
        synchronized = self.synchronizer.synchronize_with_coordinate_conversion(data_streams)
        
        # Check basic properties
        self.assertIsInstance(synchronized, pd.DataFrame)
        self.assertGreater(len(synchronized), 0)
        
        # Check that both GPS and IMU data are present
        gps_cols = [col for col in synchronized.columns if 'gps_' in col]
        imu_cols = [col for col in synchronized.columns if 'imu_' in col]
        enu_cols = [col for col in synchronized.columns if 'enu_' in col]
        
        self.assertGreater(len(gps_cols), 0)
        self.assertGreater(len(imu_cols), 0)
        self.assertGreater(len(enu_cols), 0)
        
        # Check that ENU coordinates are present and valid
        self.assertIn('enu_x', synchronized.columns)
        self.assertIn('enu_y', synchronized.columns)
        self.assertIn('enu_z', synchronized.columns)
        
        # Check that ENU values are reasonable (not all NaN)
        self.assertTrue(synchronized['enu_x'].notna().any())
        self.assertTrue(synchronized['enu_y'].notna().any())
        self.assertTrue(synchronized['enu_z'].notna().any())
        
        # Check timestamp uniformity
        time_diffs = np.diff(synchronized['timestamp'])
        expected_dt = 1.0 / 5.0  # 5 Hz
        self.assertTrue(np.allclose(time_diffs, expected_dt, rtol=0.1))
    
    def test_end_to_end_with_real_data_structure(self):
        """Test end-to-end synchronization with realistic data structure."""
        # Simulate data from multiple parsers with better overlap
        np.random.seed(42)  # For reproducible tests
        
        tlog_data = pd.DataFrame({
            'timestamp': np.arange(2.0, 8.0, 0.2),  # 5 Hz GPS, shorter range
            'gps_lat': 37.7749 + np.random.normal(0, 0.0001, 30),
            'gps_lon': -122.4194 + np.random.normal(0, 0.0001, 30),
            'gps_alt': 100.0 + np.random.normal(0, 1.0, 30),
            'hdop': 1.5 + np.random.normal(0, 0.1, 30),
            'fix_type': np.full(30, 3)
        })
        
        bin_data = pd.DataFrame({
            'timestamp': np.arange(2.0, 8.0, 0.2),  # Same frequency and range for better overlap
            'imu_ax': np.random.normal(0, 0.1, 30),
            'imu_ay': np.random.normal(0, 0.1, 30),
            'imu_az': 9.81 + np.random.normal(0, 0.05, 30),
            'imu_gx': np.random.normal(0, 0.01, 30),
            'imu_gy': np.random.normal(0, 0.01, 30),
            'imu_gz': np.random.normal(0, 0.01, 30)
        })
        
        data_streams = {'tlog': tlog_data, 'bin': bin_data}
        
        # Synchronize
        synchronized = self.synchronizer.synchronize_with_coordinate_conversion(data_streams)
        
        # Validate results
        self.assertIsInstance(synchronized, pd.DataFrame)
        self.assertGreater(len(synchronized), 10)  # Should have reasonable amount of data
        
        # Check data quality
        missing_percentage = synchronized.isnull().sum() / len(synchronized)
        
        # Most columns should have reasonable data coverage
        for col in synchronized.columns:
            if col != 'timestamp':
                self.assertLess(missing_percentage[col], 0.8, 
                               f"Column {col} has too much missing data: {missing_percentage[col]:.2%}")


if __name__ == '__main__':
    unittest.main()