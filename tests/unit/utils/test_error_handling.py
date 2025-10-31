"""
Unit tests for error handling utilities.

Tests robust error handling, memory management, and data quality validation.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from unittest.mock import Mock, patch, MagicMock
import psutil

from uav_log_processor.utils.error_handling import (
    RobustErrorHandler, ChunkedProcessor, CoordinateConverter,
    ProcessingError, CorruptedFileError, MemoryError, CoordinateConversionError,
    safe_operation, validate_dataframe_integrity
)
from uav_log_processor.utils.gps_filter import GPSReliabilityFilter, prioritize_gps_units
from uav_log_processor.utils.data_quality import (
    DataQualityValidator, DataQualityReporter, DataQualityMetrics,
    validate_processing_pipeline, check_data_consistency
)


class TestRobustErrorHandler(unittest.TestCase):
    """Test robust error handling functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = RobustErrorHandler(max_memory_gb=1.0, enable_recovery=True)
    
    def test_handle_processing_errors_success(self):
        """Test successful operation handling."""
        with self.handler.handle_processing_errors("test_operation"):
            result = 42
        
        self.assertEqual(len(self.handler.error_log), 0)
    
    def test_handle_processing_errors_non_critical(self):
        """Test non-critical error handling."""
        with self.handler.handle_processing_errors("test_operation", critical=False):
            raise ValueError("Test error")
        
        self.assertEqual(len(self.handler.error_log), 1)
        self.assertEqual(self.handler.error_log[0]['operation'], 'test_operation')
        self.assertEqual(self.handler.error_log[0]['error_type'], 'ValueError')
    
    def test_handle_processing_errors_critical(self):
        """Test critical error handling."""
        with self.assertRaises(ProcessingError):
            with self.handler.handle_processing_errors("test_operation", critical=True):
                raise ValueError("Critical test error")
        
        self.assertEqual(len(self.handler.error_log), 1)
    
    @patch('psutil.Process')
    def test_check_memory_usage_normal(self, mock_process):
        """Test normal memory usage check."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 0.5 * (1024**3)  # 0.5 GB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory_gb = self.handler.check_memory_usage()
        self.assertAlmostEqual(memory_gb, 0.5, places=1)
    
    @patch('psutil.Process')
    def test_check_memory_usage_high(self, mock_process):
        """Test high memory usage detection."""
        mock_memory_info = Mock()
        mock_memory_info.rss = 2.0 * (1024**3)  # 2.0 GB (exceeds 1.0 GB limit)
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        with self.assertRaises(MemoryError):
            self.handler.check_memory_usage()
    
    def test_safe_file_read_nonexistent(self):
        """Test safe file reading with non-existent file."""
        def dummy_parser(path):
            raise FileNotFoundError("File not found")
        
        result = self.handler.safe_file_read("nonexistent.txt", dummy_parser)
        self.assertIsNone(result)
    
    def test_safe_file_read_success(self):
        """Test successful safe file reading."""
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            def dummy_parser(path):
                return test_df
            
            result = self.handler.safe_file_read(temp_path, dummy_parser)
            pd.testing.assert_frame_equal(result, test_df)
        finally:
            os.unlink(temp_path)
    
    def test_get_error_summary_empty(self):
        """Test error summary with no errors."""
        summary = self.handler.get_error_summary()
        
        self.assertEqual(summary['total_errors'], 0)
        self.assertEqual(summary['error_types'], {})
        self.assertEqual(summary['operations'], {})
    
    def test_get_error_summary_with_errors(self):
        """Test error summary with multiple errors."""
        # Add some test errors
        self.handler.error_log = [
            {'operation': 'op1', 'error_type': 'ValueError', 'error_message': 'msg1'},
            {'operation': 'op2', 'error_type': 'ValueError', 'error_message': 'msg2'},
            {'operation': 'op1', 'error_type': 'TypeError', 'error_message': 'msg3'}
        ]
        
        summary = self.handler.get_error_summary()
        
        self.assertEqual(summary['total_errors'], 3)
        self.assertEqual(summary['error_types']['ValueError'], 2)
        self.assertEqual(summary['error_types']['TypeError'], 1)
        self.assertEqual(summary['operations']['op1'], 2)
        self.assertEqual(summary['operations']['op2'], 1)


class TestChunkedProcessor(unittest.TestCase):
    """Test chunked processing functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = ChunkedProcessor(chunk_size=100, max_memory_gb=1.0)
    
    def test_process_in_chunks_empty(self):
        """Test processing empty DataFrame."""
        df = pd.DataFrame()
        
        def dummy_process(chunk):
            return chunk
        
        result = self.processor.process_in_chunks(df, dummy_process)
        self.assertTrue(result.empty)
    
    def test_process_in_chunks_success(self):
        """Test successful chunked processing."""
        df = pd.DataFrame({'col1': range(250)})  # 250 rows, chunk_size=100
        
        def add_one(chunk):
            result = chunk.copy()
            result['col1'] = result['col1'] + 1
            return result
        
        result = self.processor.process_in_chunks(df, add_one)
        
        self.assertEqual(len(result), 250)
        self.assertEqual(result['col1'].iloc[0], 1)  # 0 + 1
        self.assertEqual(result['col1'].iloc[-1], 250)  # 249 + 1
    
    def test_estimate_chunk_size(self):
        """Test chunk size estimation."""
        df = pd.DataFrame({
            'col1': range(1000),
            'col2': np.random.random(1000),
            'col3': ['test'] * 1000
        })
        
        chunk_size = self.processor._estimate_chunk_size(df)
        
        self.assertGreater(chunk_size, 0)
        self.assertLessEqual(chunk_size, 100000)  # Max bound
        self.assertGreaterEqual(chunk_size, 1000)  # Min bound


class TestCoordinateConverter(unittest.TestCase):
    """Test coordinate conversion with error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = CoordinateConverter()
    
    def test_validate_home_point_valid(self):
        """Test valid home point validation."""
        valid_points = [
            (37.123, -122.456, 100.0),
            (0.0, 0.0, 0.0),
            (-45.0, 180.0, 1000.0)
        ]
        
        for point in valid_points:
            self.assertTrue(self.converter._validate_home_point(point))
    
    def test_validate_home_point_invalid(self):
        """Test invalid home point validation."""
        invalid_points = [
            (91.0, 0.0, 0.0),  # Invalid latitude
            (0.0, 181.0, 0.0),  # Invalid longitude
            (0.0, 0.0, 20000.0),  # Invalid altitude
            (37.123, -122.456),  # Wrong number of elements
        ]
        
        for point in invalid_points:
            self.assertFalse(self.converter._validate_home_point(point))
    
    def test_safe_coordinate_conversion_empty(self):
        """Test coordinate conversion with empty DataFrame."""
        df = pd.DataFrame()
        home_point = (37.123, -122.456, 100.0)
        
        result = self.converter.safe_coordinate_conversion(df, home_point)
        self.assertTrue(result.empty)
    
    @patch('pyproj.Transformer')
    @patch('pyproj.CRS')
    def test_safe_coordinate_conversion_success(self, mock_crs, mock_transformer):
        """Test successful coordinate conversion."""
        # Mock pyproj components
        mock_transformer_instance = Mock()
        mock_transformer_instance.transform.return_value = ([100.0, 101.0], [200.0, 201.0])
        mock_transformer.from_crs.return_value = mock_transformer_instance
        
        df = pd.DataFrame({
            'gps_lat': [37.123, 37.124],
            'gps_lon': [-122.456, -122.457],
            'gps_alt': [150.0, 151.0]
        })
        home_point = (37.123, -122.456, 100.0)
        
        result = self.converter.safe_coordinate_conversion(df, home_point)
        
        self.assertIn('gps_x', result.columns)
        self.assertIn('gps_y', result.columns)
        self.assertIn('gps_z', result.columns)
        self.assertEqual(result['gps_x'].iloc[0], 100.0)
        self.assertEqual(result['gps_y'].iloc[0], 200.0)
        self.assertEqual(result['gps_z'].iloc[0], 50.0)  # 150 - 100


class TestGPSReliabilityFilter(unittest.TestCase):
    """Test GPS reliability filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = GPSReliabilityFilter(min_fix_type=3, max_hdop=5.0, max_vdop=10.0)
    
    def test_filter_gps_data_empty(self):
        """Test filtering empty GPS data."""
        df = pd.DataFrame()
        result = self.filter.filter_gps_data(df)
        self.assertTrue(result.empty)
    
    def test_filter_gps_data_fix_type(self):
        """Test GPS filtering by fix type."""
        df = pd.DataFrame({
            'fix_type': [1, 2, 3, 4, 5],
            'hdop': [1.0, 1.0, 1.0, 1.0, 1.0],
            'vdop': [2.0, 2.0, 2.0, 2.0, 2.0]
        })
        
        result = self.filter.filter_gps_data(df)
        
        # Should keep only fix_type >= 3
        self.assertEqual(len(result), 3)
        self.assertTrue((result['fix_type'] >= 3).all())
    
    def test_filter_gps_data_hdop(self):
        """Test GPS filtering by HDOP."""
        df = pd.DataFrame({
            'fix_type': [3, 3, 3, 3, 3],
            'hdop': [1.0, 3.0, 5.0, 7.0, 10.0],
            'vdop': [2.0, 2.0, 2.0, 2.0, 2.0]
        })
        
        result = self.filter.filter_gps_data(df)
        
        # Should keep only hdop <= 5.0
        self.assertEqual(len(result), 3)
        self.assertTrue((result['hdop'] <= 5.0).all())
    
    def test_select_best_gps_source_single(self):
        """Test GPS source selection with single source."""
        gps_data = pd.DataFrame({
            'fix_type': [3, 3, 3],
            'hdop': [1.0, 1.0, 1.0],
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        sources = {'gps1': gps_data}
        
        best_name, best_data = self.filter.select_best_gps_source(sources)
        
        self.assertEqual(best_name, 'gps1')
        self.assertEqual(len(best_data), 3)
    
    def test_select_best_gps_source_multiple(self):
        """Test GPS source selection with multiple sources."""
        gps1 = pd.DataFrame({
            'fix_type': [2, 2, 2],  # Poor fix quality
            'hdop': [10.0, 10.0, 10.0],  # High HDOP
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        gps2 = pd.DataFrame({
            'fix_type': [3, 3, 3],  # Good fix quality
            'hdop': [1.0, 1.0, 1.0],  # Low HDOP
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        sources = {'gps1': gps1, 'gps2': gps2}
        
        best_name, best_data = self.filter.select_best_gps_source(sources)
        
        self.assertEqual(best_name, 'gps2')  # Should select better quality source
    
    def test_get_gps_quality_report(self):
        """Test GPS quality report generation."""
        df = pd.DataFrame({
            'fix_type': [3, 3, 2, 3],
            'hdop': [1.0, 2.0, 10.0, 1.5],
            'vdop': [2.0, 3.0, 15.0, 2.5],
            'gps_lat': [37.123, 37.124, 37.125, 37.126],
            'gps_lon': [-122.456, -122.457, -122.458, -122.459]
        })
        
        report = self.filter.get_gps_quality_report(df)
        
        self.assertEqual(report['total_samples'], 4)
        self.assertEqual(report['reliable_samples'], 3)  # Only 3 pass all filters
        self.assertAlmostEqual(report['reliability_ratio'], 0.75)
        self.assertAlmostEqual(report['avg_hdop'], 3.625)  # (1+2+10+1.5)/4
        self.assertAlmostEqual(report['avg_fix_type'], 2.75)  # (3+3+2+3)/4


class TestDataQualityValidator(unittest.TestCase):
    """Test data quality validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator()
    
    def test_validate_dataset_empty(self):
        """Test validation of empty dataset."""
        df = pd.DataFrame()
        
        metrics = self.validator.validate_dataset(df)
        
        self.assertEqual(metrics.total_samples, 0)
        self.assertGreater(len(metrics.processing_errors), 0)
    
    def test_validate_dataset_good_quality(self):
        """Test validation of good quality dataset."""
        df = pd.DataFrame({
            'timestamp': np.arange(0, 10, 0.1),  # 100 samples, 10 Hz
            'gps_lat': 37.123 + np.random.normal(0, 0.0001, 100),
            'gps_lon': -122.456 + np.random.normal(0, 0.0001, 100),
            'gps_alt': 100.0 + np.random.normal(0, 1.0, 100),
            'fix_type': [3] * 100,
            'hdop': [1.0] * 100,
            'vdop': [2.0] * 100,
            'imu_ax': np.random.normal(0, 0.1, 100),
            'imu_ay': np.random.normal(0, 0.1, 100),
            'imu_az': np.random.normal(9.8, 0.1, 100),
            'imu_gx': np.random.normal(0, 0.01, 100),
            'imu_gy': np.random.normal(0, 0.01, 100),
            'imu_gz': np.random.normal(0, 0.01, 100)
        })
        
        metrics = self.validator.validate_dataset(df)
        
        self.assertEqual(metrics.total_samples, 100)
        self.assertGreater(metrics.overall_quality_score, 80)  # Should be high quality
        self.assertAlmostEqual(metrics.sampling_rate_hz, 10.0, places=0)
        self.assertEqual(metrics.time_gaps_count, 0)
    
    def test_validate_dataset_poor_quality(self):
        """Test validation of poor quality dataset."""
        df = pd.DataFrame({
            'timestamp': [0, 1, 10, 11, 20],  # Large gaps
            'gps_lat': [37.123, 37.124, np.nan, 37.126, 37.127],  # Missing data
            'gps_lon': [-122.456, -122.457, np.nan, -122.459, -122.460],
            'fix_type': [1, 2, 1, 2, 1],  # Poor fix quality
            'hdop': [10.0, 15.0, 20.0, 12.0, 8.0],  # High HDOP
        })
        
        metrics = self.validator.validate_dataset(df, original_count=10)
        
        self.assertEqual(metrics.total_samples, 5)
        self.assertEqual(metrics.data_loss_ratio, 0.5)  # 50% data loss
        self.assertLess(metrics.overall_quality_score, 50)  # Should be poor quality
        self.assertGreater(metrics.time_gaps_count, 0)  # Should detect gaps
        self.assertGreater(len(metrics.processing_warnings), 0)


class TestDataQualityReporter(unittest.TestCase):
    """Test data quality reporting."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = DataQualityReporter()
    
    def test_generate_report(self):
        """Test report generation."""
        metrics = DataQualityMetrics(
            total_samples=100,
            valid_samples=95,
            data_loss_ratio=0.05,
            duration_seconds=10.0,
            sampling_rate_hz=10.0,
            gps_fix_ratio=0.9,
            avg_hdop=2.0,
            overall_quality_score=85.0,
            processing_warnings=['Test warning'],
            processing_errors=[]
        )
        
        report = self.reporter.generate_report(metrics)
        
        self.assertIn('report_metadata', report)
        self.assertIn('summary', report)
        self.assertIn('temporal_quality', report)
        self.assertIn('gps_quality', report)
        self.assertIn('recommendations', report)
        
        self.assertEqual(report['summary']['overall_quality_score'], 85.0)
        self.assertEqual(report['summary']['quality_grade'], 'B')
        self.assertEqual(report['temporal_quality']['sampling_rate_hz'], 10.0)
        self.assertEqual(report['gps_quality']['gps_fix_ratio'], 0.9)
    
    def test_get_quality_grade(self):
        """Test quality grade assignment."""
        self.assertEqual(self.reporter._get_quality_grade(95), 'A')
        self.assertEqual(self.reporter._get_quality_grade(85), 'B')
        self.assertEqual(self.reporter._get_quality_grade(75), 'C')
        self.assertEqual(self.reporter._get_quality_grade(65), 'D')
        self.assertEqual(self.reporter._get_quality_grade(45), 'F')


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_safe_operation_success(self):
        """Test successful safe operation."""
        def add_numbers(a, b):
            return a + b
        
        result = safe_operation(add_numbers, 2, 3)
        self.assertEqual(result, 5)
    
    def test_safe_operation_failure(self):
        """Test failed safe operation."""
        def failing_function():
            raise ValueError("Test error")
        
        result = safe_operation(failing_function, default_return="default")
        self.assertEqual(result, "default")
    
    def test_validate_dataframe_integrity_valid(self):
        """Test DataFrame integrity validation with valid data."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'col3': [1.0, 2.0, 3.0]
        })
        
        result = validate_dataframe_integrity(df, required_columns=['col1', 'col2'])
        
        self.assertTrue(result['valid'])
        self.assertEqual(result['row_count'], 3)
        self.assertEqual(result['column_count'], 3)
        self.assertEqual(result['missing_columns'], [])
    
    def test_validate_dataframe_integrity_missing_columns(self):
        """Test DataFrame integrity validation with missing columns."""
        df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = validate_dataframe_integrity(df, required_columns=['col1', 'col2', 'col3'])
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['missing_columns'], ['col3'])
    
    def test_validate_dataframe_integrity_none(self):
        """Test DataFrame integrity validation with None input."""
        result = validate_dataframe_integrity(None, required_columns=['col1'])
        
        self.assertFalse(result['valid'])
        self.assertEqual(result['row_count'], 0)
        self.assertEqual(result['missing_columns'], ['col1'])
    
    def test_check_data_consistency(self):
        """Test data consistency checking."""
        df = pd.DataFrame({
            'timestamp': [1.0, 1.0, 2.0],  # Duplicate timestamp
            'gps_lat': [37.123, 91.0, 37.125],  # Invalid latitude
            'gps_lon': [-122.456, -122.457, 181.0],  # Invalid longitude
            'gps_alt': [100.0, 60000.0, 102.0],  # Extreme altitude
            'constant_col': [5.0, 5.0, 5.0]  # Constant values
        })
        
        issues = check_data_consistency(df)
        
        self.assertGreater(len(issues), 0)
        # Should detect duplicate timestamps, invalid coordinates, extreme altitude, and constant values
        issue_text = ' '.join(issues)
        self.assertIn('duplicate', issue_text.lower())
        self.assertIn('latitude', issue_text.lower())
        self.assertIn('longitude', issue_text.lower())
        self.assertIn('altitude', issue_text.lower())
        self.assertIn('constant', issue_text.lower())


if __name__ == '__main__':
    unittest.main()