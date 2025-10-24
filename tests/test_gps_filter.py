"""
Unit tests for GPS reliability filtering.

Tests GPS filtering, source selection, and quality assessment.
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from uav_log_processor.utils.gps_filter import (
    GPSReliabilityFilter, prioritize_gps_units, detect_gps_outages
)


class TestGPSReliabilityFilter(unittest.TestCase):
    """Test GPS reliability filtering functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.filter = GPSReliabilityFilter(min_fix_type=3, max_hdop=5.0, max_vdop=10.0)
    
    def test_filter_gps_data_all_criteria(self):
        """Test filtering with all quality criteria."""
        df = pd.DataFrame({
            'fix_type': [1, 2, 3, 4, 5, 2],
            'hdop': [1.0, 2.0, 3.0, 4.0, 6.0, 2.0],  # One exceeds max_hdop
            'vdop': [2.0, 3.0, 4.0, 5.0, 8.0, 12.0],  # One exceeds max_vdop
            'gps_lat': [37.123, 37.124, 37.125, 37.126, 37.127, 37.128],
            'gps_lon': [-122.456, -122.457, -122.458, -122.459, -122.460, -122.461]
        })
        
        result = self.filter.filter_gps_data(df)
        
        # Should keep only rows that pass all filters:
        # fix_type >= 3, hdop <= 5.0, vdop <= 10.0
        # Rows 2 and 3 should pass (indices 2, 3)
        self.assertEqual(len(result), 2)
        self.assertTrue((result['fix_type'] >= 3).all())
        self.assertTrue((result['hdop'] <= 5.0).all())
        self.assertTrue((result['vdop'] <= 10.0).all())
    
    def test_filter_gps_data_missing_columns(self):
        """Test filtering with missing quality columns."""
        df = pd.DataFrame({
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        result = self.filter.filter_gps_data(df)
        
        # Should return all data when quality columns are missing
        self.assertEqual(len(result), 3)
    
    def test_calculate_gps_quality_score_high_quality(self):
        """Test quality score calculation for high-quality GPS data."""
        np.random.seed(42)  # For reproducible results
        n_samples = 1000
        df = pd.DataFrame({
            'fix_type': [4, 4, 4, 4, 4] * (n_samples // 5),  # RTK fix
            'hdop': [1.0, 1.1, 0.9, 1.2, 1.0] * (n_samples // 5),  # Low HDOP
            'vdop': [1.5, 1.6, 1.4, 1.7, 1.5] * (n_samples // 5),  # Low VDOP
            'gps_lat': 37.123456 + np.random.normal(0, 0.00001, n_samples),  # Stable position
            'gps_lon': -122.456789 + np.random.normal(0, 0.00001, n_samples)
        })
        
        score = self.filter._calculate_gps_quality_score(df)
        
        # High-quality data should get a high score
        self.assertGreater(score, 80)
    
    def test_calculate_gps_quality_score_low_quality(self):
        """Test quality score calculation for low-quality GPS data."""
        np.random.seed(42)  # For reproducible results
        n_samples = 100
        df = pd.DataFrame({
            'fix_type': [2, 1, 2, 1, 2] * (n_samples // 5),  # Poor fix quality
            'hdop': [8.0, 10.0, 9.0, 12.0, 7.0] * (n_samples // 5),  # High HDOP
            'vdop': [15.0, 20.0, 18.0, 25.0, 12.0] * (n_samples // 5),  # High VDOP
            'gps_lat': 37.123 + np.random.normal(0, 0.001, n_samples),  # Unstable position
            'gps_lon': -122.456 + np.random.normal(0, 0.001, n_samples)
        })
        
        score = self.filter._calculate_gps_quality_score(df)
        
        # Low-quality data should get a low score
        self.assertLess(score, 40)
    
    def test_select_best_gps_source_quality_comparison(self):
        """Test GPS source selection based on quality metrics."""
        # Create GPS source with poor quality
        gps_poor = pd.DataFrame({
            'fix_type': [2, 2, 2, 2, 2],
            'hdop': [10.0, 12.0, 8.0, 15.0, 9.0],
            'vdop': [20.0, 25.0, 18.0, 30.0, 22.0],
            'gps_lat': [37.123, 37.124, 37.125, 37.126, 37.127],
            'gps_lon': [-122.456, -122.457, -122.458, -122.459, -122.460]
        })
        
        # Create GPS source with good quality
        gps_good = pd.DataFrame({
            'fix_type': [4, 4, 4, 4, 4],
            'hdop': [1.0, 1.2, 0.8, 1.1, 1.0],
            'vdop': [2.0, 2.2, 1.8, 2.1, 2.0],
            'gps_lat': [37.123, 37.124, 37.125, 37.126, 37.127],
            'gps_lon': [-122.456, -122.457, -122.458, -122.459, -122.460]
        })
        
        sources = {'gps_poor': gps_poor, 'gps_good': gps_good}
        
        best_name, best_data = self.filter.select_best_gps_source(sources)
        
        self.assertEqual(best_name, 'gps_good')
        self.assertEqual(len(best_data), 5)
    
    def test_select_best_gps_source_no_valid_data(self):
        """Test GPS source selection when no sources have valid data."""
        gps_invalid = pd.DataFrame({
            'fix_type': [1, 1, 1],  # All below minimum
            'hdop': [20.0, 25.0, 30.0],  # All above maximum
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        sources = {'gps1': gps_invalid}
        
        with self.assertRaises(ValueError):
            self.filter.select_best_gps_source(sources)
    
    def test_get_gps_quality_report_comprehensive(self):
        """Test comprehensive GPS quality report generation."""
        df = pd.DataFrame({
            'fix_type': [3, 4, 2, 3, 4, 1],
            'hdop': [1.5, 1.0, 8.0, 2.0, 1.2, 15.0],
            'vdop': [3.0, 2.0, 12.0, 4.0, 2.5, 20.0],
            'gps_lat': [37.123456, 37.123457, 37.123458, 37.123459, 37.123460, 37.123461],
            'gps_lon': [-122.456789, -122.456790, -122.456791, -122.456792, -122.456793, -122.456794]
        })
        
        report = self.filter.get_gps_quality_report(df)
        
        self.assertEqual(report['total_samples'], 6)
        self.assertEqual(report['reliable_samples'], 4)  # 4 pass all filters (fix_type>=3, hdop<=5, vdop<=10)
        self.assertAlmostEqual(report['reliability_ratio'], 4/6)  # 4 out of 6 samples
        self.assertAlmostEqual(report['avg_hdop'], 4.783333, places=2)
        self.assertAlmostEqual(report['avg_vdop'], 7.25)
        self.assertAlmostEqual(report['avg_fix_type'], 2.833333, places=2)
        self.assertIsInstance(report['position_std_m'], float)


class TestGPSUtilityFunctions(unittest.TestCase):
    """Test GPS utility functions."""
    
    def test_prioritize_gps_units_empty(self):
        """Test GPS unit prioritization with empty input."""
        result = prioritize_gps_units({})
        self.assertEqual(result, [])
    
    def test_prioritize_gps_units_multiple(self):
        """Test GPS unit prioritization with multiple units."""
        gps1 = pd.DataFrame({
            'fix_type': [3, 3, 3],  # Valid fix type but poor HDOP
            'hdop': [4.5, 4.8, 4.2],  # High but still acceptable HDOP
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        gps2 = pd.DataFrame({
            'fix_type': [4, 4, 4],
            'hdop': [1.0, 1.2, 0.8],
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        gps3 = pd.DataFrame({
            'fix_type': [3, 3, 3],
            'hdop': [2.0, 2.2, 1.8],
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        gps_dict = {'gps1': gps1, 'gps2': gps2, 'gps3': gps3}
        
        result = prioritize_gps_units(gps_dict)
        
        # Should be ordered by quality: gps2 (best), gps3 (medium), gps1 (worst)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], 'gps2')  # Best quality first
        self.assertEqual(result[1][0], 'gps3')  # Medium quality second
        self.assertEqual(result[2][0], 'gps1')  # Worst quality last
    
    def test_prioritize_gps_units_with_config(self):
        """Test GPS unit prioritization with custom configuration."""
        gps_data = pd.DataFrame({
            'fix_type': [3, 3, 3],
            'hdop': [2.0, 2.2, 1.8],
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        config = {
            'min_gps_fix_type': 4,  # Stricter requirement
            'max_hdop': 1.5,  # Stricter requirement
            'max_vdop': 5.0
        }
        
        result = prioritize_gps_units({'gps1': gps_data}, config)
        
        # Should return empty list due to strict requirements
        self.assertEqual(result, [])
    
    def test_detect_gps_outages_no_gaps(self):
        """Test GPS outage detection with continuous data."""
        df = pd.DataFrame({
            'timestamp': np.arange(0, 10, 0.1),  # Continuous 0.1s intervals
            'gps_lat': [37.123] * 100,
            'gps_lon': [-122.456] * 100
        })
        
        outages = detect_gps_outages(df, max_gap_seconds=1.0)
        
        self.assertEqual(outages, [])
    
    def test_detect_gps_outages_with_gaps(self):
        """Test GPS outage detection with data gaps."""
        timestamps = [0.0, 0.1, 0.2, 5.0, 5.1, 10.0, 10.1]  # Gaps at 0.2->5.0 and 5.1->10.0
        df = pd.DataFrame({
            'timestamp': timestamps,
            'gps_lat': [37.123] * len(timestamps),
            'gps_lon': [-122.456] * len(timestamps)
        })
        
        outages = detect_gps_outages(df, max_gap_seconds=1.0)
        
        self.assertEqual(len(outages), 2)
        # First outage: 0.2 to 5.0
        self.assertAlmostEqual(outages[0][0], 0.2)
        self.assertAlmostEqual(outages[0][1], 5.0)
        # Second outage: 5.1 to 10.0
        self.assertAlmostEqual(outages[1][0], 5.1)
        self.assertAlmostEqual(outages[1][1], 10.0)
    
    def test_detect_gps_outages_empty_data(self):
        """Test GPS outage detection with empty data."""
        df = pd.DataFrame()
        
        outages = detect_gps_outages(df)
        
        self.assertEqual(outages, [])
    
    def test_detect_gps_outages_no_timestamp(self):
        """Test GPS outage detection without timestamp column."""
        df = pd.DataFrame({
            'gps_lat': [37.123, 37.124, 37.125],
            'gps_lon': [-122.456, -122.457, -122.458]
        })
        
        outages = detect_gps_outages(df)
        
        self.assertEqual(outages, [])


if __name__ == '__main__':
    unittest.main()