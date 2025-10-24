"""
Unit tests for data quality validation and reporting.

Tests data quality metrics, validation, and comprehensive reporting.
"""

import unittest
import tempfile
import os
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from uav_log_processor.utils.data_quality import (
    DataQualityValidator, DataQualityReporter, DataQualityMetrics,
    validate_processing_pipeline, check_data_consistency
)


class TestDataQualityMetrics(unittest.TestCase):
    """Test DataQualityMetrics dataclass."""
    
    def test_default_initialization(self):
        """Test default metrics initialization."""
        metrics = DataQualityMetrics()
        
        self.assertEqual(metrics.total_samples, 0)
        self.assertEqual(metrics.valid_samples, 0)
        self.assertEqual(metrics.data_loss_ratio, 0.0)
        self.assertEqual(metrics.overall_quality_score, 0.0)
        self.assertEqual(metrics.processing_warnings, [])
        self.assertEqual(metrics.processing_errors, [])
    
    def test_custom_initialization(self):
        """Test custom metrics initialization."""
        warnings = ['Test warning']
        errors = ['Test error']
        
        metrics = DataQualityMetrics(
            total_samples=100,
            valid_samples=95,
            data_loss_ratio=0.05,
            overall_quality_score=85.0,
            processing_warnings=warnings,
            processing_errors=errors
        )
        
        self.assertEqual(metrics.total_samples, 100)
        self.assertEqual(metrics.valid_samples, 95)
        self.assertEqual(metrics.data_loss_ratio, 0.05)
        self.assertEqual(metrics.overall_quality_score, 85.0)
        self.assertEqual(metrics.processing_warnings, warnings)
        self.assertEqual(metrics.processing_errors, errors)


class TestDataQualityValidator(unittest.TestCase):
    """Test data quality validation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = DataQualityValidator({
            'max_data_loss_warning': 0.1,
            'min_sampling_rate': 5.0,
            'max_time_gap': 5.0,
            'min_gps_fix_ratio': 0.8,
            'max_hdop_threshold': 5.0,
            'min_position_stability': 100.0
        })
    
    def test_validate_dataset_none(self):
        """Test validation with None dataset."""
        metrics = self.validator.validate_dataset(None)
        
        self.assertEqual(metrics.total_samples, 0)
        self.assertGreater(len(metrics.processing_errors), 0)
        self.assertIn('empty', metrics.processing_errors[0].lower())
    
    def test_validate_dataset_empty(self):
        """Test validation with empty dataset."""
        df = pd.DataFrame()
        
        metrics = self.validator.validate_dataset(df)
        
        self.assertEqual(metrics.total_samples, 0)
        self.assertGreater(len(metrics.processing_errors), 0)
    
    def test_validate_temporal_quality_good(self):
        """Test temporal quality validation with good data."""
        # Create 10 seconds of data at 10 Hz
        timestamps = np.arange(0, 10, 0.1)
        df = pd.DataFrame({'timestamp': timestamps})
        
        metrics = DataQualityMetrics()
        self.validator._validate_temporal_quality(df, metrics)
        
        self.assertAlmostEqual(metrics.duration_seconds, 9.9, places=1)
        self.assertAlmostEqual(metrics.sampling_rate_hz, 10.0, places=0)
        self.assertEqual(metrics.time_gaps_count, 0)
        self.assertEqual(len(self.validator.warnings), 0)
    
    def test_validate_temporal_quality_gaps(self):
        """Test temporal quality validation with time gaps."""
        # Create data with large gaps
        timestamps = [0.0, 0.1, 0.2, 10.0, 10.1, 20.0, 20.1]
        df = pd.DataFrame({'timestamp': timestamps})
        
        metrics = DataQualityMetrics()
        self.validator._validate_temporal_quality(df, metrics)
        
        self.assertGreater(metrics.time_gaps_count, 0)
        self.assertGreater(metrics.max_gap_seconds, 5.0)
        self.assertGreater(len(self.validator.warnings), 0)
    
    def test_validate_temporal_quality_low_rate(self):
        """Test temporal quality validation with low sampling rate."""
        # Create 10 seconds of data at 1 Hz (below 5 Hz threshold)
        timestamps = np.arange(0, 10, 1.0)
        df = pd.DataFrame({'timestamp': timestamps})
        
        metrics = DataQualityMetrics()
        self.validator._validate_temporal_quality(df, metrics)
        
        self.assertAlmostEqual(metrics.sampling_rate_hz, 1.0, places=0)
        self.assertGreater(len(self.validator.warnings), 0)
        self.assertIn('sampling rate', self.validator.warnings[0].lower())
    
    def test_validate_gps_quality_good(self):
        """Test GPS quality validation with good data."""
        df = pd.DataFrame({
            'gps_lat': [37.123456, 37.123457, 37.123458],
            'gps_lon': [-122.456789, -122.456790, -122.456791],
            'gps_alt': [100.0, 100.1, 100.2],
            'fix_type': [3, 3, 4],
            'hdop': [1.0, 1.2, 0.8],
            'vdop': [2.0, 2.2, 1.8]
        })
        
        metrics = DataQualityMetrics()
        self.validator._validate_gps_quality(df, metrics)
        
        self.assertEqual(metrics.valid_samples, 3)
        self.assertEqual(metrics.gps_fix_ratio, 1.0)  # All fixes are >= 3
        self.assertLess(metrics.avg_hdop, 5.0)
        self.assertLess(metrics.position_std_m, 100.0)
        self.assertEqual(len(self.validator.warnings), 0)
    
    def test_validate_gps_quality_poor(self):
        """Test GPS quality validation with poor data."""
        df = pd.DataFrame({
            'gps_lat': [37.123, 37.124, 37.125, 37.126],  # High variability
            'gps_lon': [-122.456, -122.457, -122.458, -122.459],
            'fix_type': [1, 2, 3, 2],  # Poor fix quality
            'hdop': [10.0, 12.0, 8.0, 15.0],  # High HDOP
            'vdop': [20.0, 25.0, 18.0, 30.0]
        })
        
        metrics = DataQualityMetrics()
        self.validator._validate_gps_quality(df, metrics)
        
        self.assertEqual(metrics.valid_samples, 4)
        self.assertLess(metrics.gps_fix_ratio, 0.8)  # Only 1/4 fixes are >= 3
        self.assertGreater(metrics.avg_hdop, 5.0)
        self.assertGreater(len(self.validator.warnings), 0)
    
    def test_validate_imu_quality_good(self):
        """Test IMU quality validation with good data."""
        df = pd.DataFrame({
            'imu_ax': np.random.normal(0, 0.1, 100),
            'imu_ay': np.random.normal(0, 0.1, 100),
            'imu_az': np.random.normal(9.8, 0.1, 100),
            'imu_gx': np.random.normal(0, 0.01, 100),
            'imu_gy': np.random.normal(0, 0.01, 100),
            'imu_gz': np.random.normal(0, 0.01, 100)
        })
        
        metrics = DataQualityMetrics()
        metrics.total_samples = 100
        self.validator._validate_imu_quality(df, metrics)
        
        self.assertEqual(metrics.imu_data_ratio, 1.0)  # All samples have IMU data
        self.assertIsInstance(metrics.accel_std, float)
        self.assertIsInstance(metrics.gyro_std, float)
        self.assertEqual(len(self.validator.warnings), 0)
    
    def test_validate_imu_quality_poor(self):
        """Test IMU quality validation with poor data."""
        # Create data with many NaN values and unrealistic readings
        df = pd.DataFrame({
            'imu_ax': [np.nan, 0.1, np.nan, 100.0, 0.2],  # Missing data and extreme values
            'imu_ay': [0.1, np.nan, 0.2, 0.3, np.nan],
            'imu_az': [9.8, 9.9, np.nan, 9.7, 9.8],
            'imu_gx': [0.01, 0.02, np.nan, 100.0, 0.01],  # Extreme gyro value
            'imu_gy': [np.nan, 0.01, 0.02, 0.01, np.nan],
            'imu_gz': [0.01, np.nan, 0.01, 0.02, 0.01]
        })
        
        metrics = DataQualityMetrics()
        metrics.total_samples = 5
        self.validator._validate_imu_quality(df, metrics)
        
        self.assertLess(metrics.imu_data_ratio, 0.8)  # Low data availability
        self.assertGreater(len(self.validator.warnings), 0)
    
    def test_calculate_quality_score_perfect(self):
        """Test quality score calculation for perfect data."""
        metrics = DataQualityMetrics(
            data_loss_ratio=0.0,
            sampling_rate_hz=15.0,
            gps_fix_ratio=1.0,
            avg_hdop=1.0,
            imu_data_ratio=1.0,
            time_gaps_count=0,
            processing_warnings=[],
            processing_errors=[]
        )
        
        score = self.validator._calculate_quality_score(metrics)
        
        self.assertEqual(score, 100.0)
    
    def test_calculate_quality_score_poor(self):
        """Test quality score calculation for poor data."""
        metrics = DataQualityMetrics(
            data_loss_ratio=0.5,  # 50% data loss
            sampling_rate_hz=2.0,  # Below minimum
            gps_fix_ratio=0.3,  # Poor GPS
            avg_hdop=10.0,  # High HDOP
            imu_data_ratio=0.5,  # Low IMU availability
            time_gaps_count=10,  # Many gaps
            processing_warnings=['warning1', 'warning2'],
            processing_errors=['error1']
        )
        
        score = self.validator._calculate_quality_score(metrics)
        
        self.assertLess(score, 30.0)  # Should be very low
    
    def test_validate_dataset_integration(self):
        """Test complete dataset validation integration."""
        # Create a realistic dataset with mixed quality
        np.random.seed(42)  # For reproducible results
        
        df = pd.DataFrame({
            'timestamp': np.arange(0, 20, 0.1),  # 20 seconds at 10 Hz
            'gps_lat': 37.123456 + np.random.normal(0, 0.0001, 200),
            'gps_lon': -122.456789 + np.random.normal(0, 0.0001, 200),
            'gps_alt': 100.0 + np.random.normal(0, 2.0, 200),
            'fix_type': np.random.choice([3, 4], 200, p=[0.7, 0.3]),
            'hdop': np.random.uniform(0.8, 2.0, 200),
            'vdop': np.random.uniform(1.5, 3.0, 200),
            'imu_ax': np.random.normal(0, 0.2, 200),
            'imu_ay': np.random.normal(0, 0.2, 200),
            'imu_az': np.random.normal(9.8, 0.3, 200),
            'imu_gx': np.random.normal(0, 0.02, 200),
            'imu_gy': np.random.normal(0, 0.02, 200),
            'imu_gz': np.random.normal(0, 0.02, 200)
        })
        
        metrics = self.validator.validate_dataset(df, original_count=220)
        
        self.assertEqual(metrics.total_samples, 200)
        self.assertAlmostEqual(metrics.data_loss_ratio, 20/220, places=2)
        self.assertAlmostEqual(metrics.sampling_rate_hz, 10.0, places=0)
        self.assertGreater(metrics.overall_quality_score, 70)  # Should be good quality


class TestDataQualityReporter(unittest.TestCase):
    """Test data quality reporting functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.reporter = DataQualityReporter()
    
    def test_generate_report_complete(self):
        """Test complete report generation."""
        metrics = DataQualityMetrics(
            total_samples=1000,
            valid_samples=950,
            data_loss_ratio=0.05,
            duration_seconds=100.0,
            sampling_rate_hz=10.0,
            time_gaps_count=2,
            max_gap_seconds=3.0,
            gps_fix_ratio=0.9,
            avg_hdop=1.5,
            avg_vdop=2.5,
            position_std_m=5.0,
            imu_data_ratio=0.95,
            accel_std=0.2,
            gyro_std=0.01,
            overall_quality_score=85.0,
            processing_warnings=['Minor warning'],
            processing_errors=[]
        )
        
        report = self.reporter.generate_report(metrics)
        
        # Check report structure
        required_sections = [
            'report_metadata', 'summary', 'temporal_quality',
            'gps_quality', 'imu_quality', 'issues', 'recommendations'
        ]
        for section in required_sections:
            self.assertIn(section, report)
        
        # Check summary section
        self.assertEqual(report['summary']['overall_quality_score'], 85.0)
        self.assertEqual(report['summary']['quality_grade'], 'B')
        self.assertEqual(report['summary']['total_samples'], 1000)
        
        # Check temporal quality section
        self.assertEqual(report['temporal_quality']['duration_seconds'], 100.0)
        self.assertEqual(report['temporal_quality']['sampling_rate_hz'], 10.0)
        
        # Check GPS quality section
        self.assertEqual(report['gps_quality']['gps_fix_ratio'], 0.9)
        self.assertEqual(report['gps_quality']['avg_hdop'], 1.5)
        
        # Check IMU quality section
        self.assertEqual(report['imu_quality']['imu_data_ratio'], 0.95)
        self.assertEqual(report['imu_quality']['accel_std'], 0.2)
        
        # Check issues section
        self.assertEqual(report['issues']['warnings'], ['Minor warning'])
        self.assertEqual(report['issues']['errors'], [])
        
        # Check recommendations
        self.assertIsInstance(report['recommendations'], list)
        self.assertGreater(len(report['recommendations']), 0)
    
    def test_generate_report_with_file_output(self):
        """Test report generation with file output."""
        metrics = DataQualityMetrics(overall_quality_score=75.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            report = self.reporter.generate_report(metrics, output_path=temp_path)
            
            # Check that file was created
            self.assertTrue(os.path.exists(temp_path))
            
            # Check file contents
            with open(temp_path, 'r') as f:
                saved_report = json.load(f)
            
            self.assertEqual(saved_report['summary']['overall_quality_score'], 75.0)
            
        finally:
            os.unlink(temp_path)
    
    def test_get_quality_grade_all_grades(self):
        """Test quality grade assignment for all grade levels."""
        test_cases = [
            (95, 'A'),
            (85, 'B'),
            (75, 'C'),
            (65, 'D'),
            (45, 'F'),
            (100, 'A'),
            (0, 'F')
        ]
        
        for score, expected_grade in test_cases:
            grade = self.reporter._get_quality_grade(score)
            self.assertEqual(grade, expected_grade, f"Score {score} should be grade {expected_grade}")
    
    def test_generate_recommendations_good_quality(self):
        """Test recommendation generation for good quality data."""
        metrics = DataQualityMetrics(
            data_loss_ratio=0.02,
            sampling_rate_hz=15.0,
            gps_fix_ratio=0.95,
            avg_hdop=1.5,
            imu_data_ratio=0.98,
            time_gaps_count=1,
            overall_quality_score=90.0
        )
        
        recommendations = self.reporter._generate_recommendations(metrics)
        
        # Should have minimal recommendations for good quality data
        self.assertIn('no major issues', ' '.join(recommendations).lower())
    
    def test_generate_recommendations_poor_quality(self):
        """Test recommendation generation for poor quality data."""
        metrics = DataQualityMetrics(
            data_loss_ratio=0.15,  # High data loss
            sampling_rate_hz=3.0,  # Low sampling rate
            gps_fix_ratio=0.6,  # Poor GPS
            avg_hdop=8.0,  # High HDOP
            imu_data_ratio=0.7,  # Low IMU availability
            time_gaps_count=10,  # Many gaps
            overall_quality_score=45.0  # Poor overall
        )
        
        recommendations = self.reporter._generate_recommendations(metrics)
        
        # Should have multiple recommendations
        self.assertGreater(len(recommendations), 3)
        
        # Check for specific recommendations
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('data loss', rec_text)
        self.assertIn('sampling rate', rec_text)
        self.assertIn('gps', rec_text)
        self.assertIn('hdop', rec_text)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for data quality."""
    
    def test_validate_processing_pipeline(self):
        """Test processing pipeline validation."""
        # Create input data
        input_data = {
            'source1': pd.DataFrame({'col1': range(100)}),
            'source2': pd.DataFrame({'col1': range(50)})
        }
        
        # Create output data
        output_data = pd.DataFrame({
            'timestamp': np.arange(0, 10, 0.1),
            'gps_lat': [37.123] * 100,
            'gps_lon': [-122.456] * 100
        })
        
        metrics = validate_processing_pipeline(input_data, output_data)
        
        self.assertIsInstance(metrics, DataQualityMetrics)
        self.assertEqual(metrics.total_samples, 100)
        # Should calculate data loss: (150 - 100) / 150 = 0.333
        self.assertAlmostEqual(metrics.data_loss_ratio, 0.333, places=2)
    
    def test_check_data_consistency_clean_data(self):
        """Test data consistency checking with clean data."""
        df = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0],
            'gps_lat': [37.123, 37.124, 37.125, 37.126, 37.127],
            'gps_lon': [-122.456, -122.457, -122.458, -122.459, -122.460],
            'gps_alt': [100.0, 101.0, 102.0, 103.0, 104.0],
            'variable_col': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        
        issues = check_data_consistency(df)
        
        self.assertEqual(issues, [])
    
    def test_check_data_consistency_problematic_data(self):
        """Test data consistency checking with problematic data."""
        df = pd.DataFrame({
            'timestamp': [1.0, 1.0, 2.0, 3.0],  # Duplicate timestamp
            'gps_lat': [37.123, 91.0, 37.125, 37.126],  # Invalid latitude
            'gps_lon': [-122.456, -122.457, 181.0, -122.459],  # Invalid longitude
            'gps_alt': [100.0, 60000.0, 102.0, 103.0],  # Extreme altitude
            'constant_col': [5.0, 5.0, 5.0, 5.0],  # Constant values
            'variable_col': [1.0, 2.0, 3.0, 4.0]
        })
        
        issues = check_data_consistency(df)
        
        self.assertGreater(len(issues), 0)
        
        # Check for specific issues
        issues_text = ' '.join(issues).lower()
        self.assertIn('duplicate', issues_text)
        self.assertIn('latitude', issues_text)
        self.assertIn('longitude', issues_text)
        self.assertIn('altitude', issues_text)
        self.assertIn('constant', issues_text)
    
    def test_check_data_consistency_empty_data(self):
        """Test data consistency checking with empty data."""
        df = pd.DataFrame()
        
        issues = check_data_consistency(df)
        
        self.assertEqual(issues, ['DataFrame is empty'])


if __name__ == '__main__':
    unittest.main()