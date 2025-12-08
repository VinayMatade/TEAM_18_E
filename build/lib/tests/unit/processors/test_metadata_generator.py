"""
Tests for MetadataGenerator processor.

Tests metadata generation, feature descriptions, normalization statistics,
and processing parameter documentation functionality.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from uav_log_processor.processors.metadata_generator import MetadataGenerator


class TestMetadataGenerator:
    """Test cases for MetadataGenerator processor."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'target_frequency': 15.0,
            'coordinate_system': 'ENU',
            'accel_threshold': 0.5,
            'gyro_threshold': 0.1,
            'normalization_method': 'zscore'
        }
        self.generator = MetadataGenerator(self.config)
        
        # Create sample data with all standard features
        self.sample_data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'gps_x': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'gps_y': [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
            'gps_z': [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0],
            'imu_ax': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'imu_ay': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            'imu_az': [9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
            'hdop': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
            'fix_type': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'gps_error_norm': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        })
        
        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test MetadataGenerator initialization."""
        generator = MetadataGenerator()
        assert generator.config == {}
        
        generator_with_config = MetadataGenerator(self.config)
        assert generator_with_config.config == self.config
    
    def test_process_generates_complete_metadata(self):
        """Test that process generates complete metadata structure."""
        metadata = self.generator.process(self.sample_data)
        
        # Check main sections exist
        expected_sections = [
            'dataset_info', 'features', 'processing_info', 
            'data_quality', 'temporal_info', 'statistics', 'generation_info'
        ]
        for section in expected_sections:
            assert section in metadata
        
        # Check dataset info
        assert metadata['dataset_info']['total_samples'] == 10
        assert metadata['dataset_info']['total_features'] == 10
        assert metadata['dataset_info']['coordinate_system'] == 'ENU'
        assert metadata['dataset_info']['target_frequency_hz'] == 15.0
    
    def test_feature_metadata_generation(self):
        """Test feature metadata generation with descriptions and statistics."""
        metadata = self.generator.process(self.sample_data)
        features = metadata['features']
        
        # Check all features are documented
        for column in self.sample_data.columns:
            assert column in features
            
            feature_info = features[column]
            assert 'description' in feature_info
            assert 'unit' in feature_info
            assert 'type' in feature_info
            assert 'source' in feature_info
            assert 'count' in feature_info
            assert 'missing_count' in feature_info
            assert 'missing_percentage' in feature_info
        
        # Check continuous feature statistics
        gps_x_info = features['gps_x']
        assert gps_x_info['type'] == 'continuous'
        assert gps_x_info['source'] == 'gps'
        assert 'mean' in gps_x_info
        assert 'std' in gps_x_info
        assert 'min' in gps_x_info
        assert 'max' in gps_x_info
        
        # Check categorical feature statistics
        fix_type_info = features['fix_type']
        assert fix_type_info['type'] == 'categorical'
        assert 'unique_count' in fix_type_info
        assert 'unique_values' in fix_type_info
    
    def test_processing_info_generation(self):
        """Test processing information generation."""
        metadata = self.generator.process(self.sample_data)
        processing_info = metadata['processing_info']
        
        # Check synchronization info
        assert 'synchronization' in processing_info
        sync_info = processing_info['synchronization']
        assert sync_info['target_frequency_hz'] == 15.0
        
        # Check motion classification info
        assert 'motion_classification' in processing_info
        motion_info = processing_info['motion_classification']
        assert motion_info['accel_threshold_ms2'] == 0.5
        assert motion_info['gyro_threshold_rads'] == 0.1
        
        # Check normalization info
        assert 'normalization' in processing_info
        norm_info = processing_info['normalization']
        assert norm_info['method'] == 'zscore'
    
    def test_data_quality_analysis(self):
        """Test data quality metrics generation."""
        metadata = self.generator.process(self.sample_data)
        quality = metadata['data_quality']
        
        # Check completeness metrics
        assert 'completeness' in quality
        completeness = quality['completeness']
        assert completeness['total_missing_values'] == 0
        assert completeness['missing_percentage'] == 0.0
        assert completeness['complete_rows'] == 10
        assert completeness['complete_rows_percentage'] == 100.0
        
        # Check GPS quality metrics
        assert 'gps_quality' in quality
        gps_quality = quality['gps_quality']
        assert gps_quality['high_quality_fixes'] == 10
        assert gps_quality['high_quality_percentage'] == 100.0
        
        # Check HDOP quality metrics
        assert 'hdop_quality' in quality
        hdop_quality = quality['hdop_quality']
        assert 'mean_hdop' in hdop_quality
        assert 'good_hdop_count' in hdop_quality
    
    def test_temporal_info_generation(self):
        """Test temporal information generation."""
        metadata = self.generator.process(self.sample_data)
        temporal = metadata['temporal_info']
        
        assert temporal['start_time'] == 1.0
        assert temporal['end_time'] == 10.0
        assert temporal['duration_seconds'] == 9.0
        assert temporal['duration_minutes'] == 0.15
        assert 'actual_frequency_hz' in temporal
        assert 'time_gaps' in temporal
        
        # Check time gap analysis
        gaps = temporal['time_gaps']
        assert 'gap_count' in gaps
        assert 'max_gap_seconds' in gaps
        assert 'mean_gap_seconds' in gaps
    
    def test_statistics_generation(self):
        """Test overall statistics generation."""
        metadata = self.generator.process(self.sample_data)
        stats = metadata['statistics']
        
        # Check shape information
        assert stats['shape']['rows'] == 10
        assert stats['shape']['columns'] == 10
        
        # Check data types
        assert 'data_types' in stats
        assert stats['data_types']['numeric_features'] > 0
        
        # Check error statistics
        assert 'error_statistics' in stats
        error_stats = stats['error_statistics']
        assert 'mean_error_m' in error_stats
        assert 'std_error_m' in error_stats
        assert 'max_error_m' in error_stats
        assert error_stats['mean_error_m'] == 0.55  # Mean of 0.1 to 1.0
    
    def test_generation_info(self):
        """Test generation information."""
        metadata = self.generator.process(self.sample_data)
        gen_info = metadata['generation_info']
        
        assert 'generated_at' in gen_info
        assert 'generator_version' in gen_info
        assert 'pandas_version' in gen_info
        assert gen_info['generator_version'] == '1.0'
    
    def test_generate_with_normalization_stats(self):
        """Test metadata generation with normalization statistics."""
        normalization_stats = {
            'gps_x': {'mean': 104.5, 'std': 3.02, 'normalized': True},
            'gps_y': {'mean': 204.5, 'std': 3.02, 'normalized': True},
            'fix_type': {'type': 'ordinal', 'unique_values': [3], 'encoded': True}
        }
        
        metadata = self.generator.generate_with_normalization_stats(
            self.sample_data, normalization_stats
        )
        
        # Check normalization info is added to features
        assert 'normalization' in metadata['features']['gps_x']
        assert metadata['features']['gps_x']['normalization']['normalized'] == True
        assert metadata['features']['gps_x']['normalization']['mean'] == 104.5
        
        # Check normalization summary
        assert 'normalization_summary' in metadata
        norm_summary = metadata['normalization_summary']
        assert len(norm_summary['normalized_features']) == 2
        assert 'gps_x' in norm_summary['normalized_features']
        assert 'gps_y' in norm_summary['normalized_features']
        assert norm_summary['total_normalized'] == 2
    
    def test_save_metadata(self):
        """Test metadata saving to JSON file."""
        metadata = self.generator.process(self.sample_data)
        output_path = Path(self.temp_dir) / "test_metadata.json"
        
        self.generator.save_metadata(metadata, str(output_path))
        
        # Check file was created
        assert output_path.exists()
        
        # Check file content
        with open(output_path, 'r') as f:
            saved_metadata = json.load(f)
        
        assert saved_metadata['dataset_info']['total_samples'] == 10
        assert 'features' in saved_metadata
        assert 'processing_info' in saved_metadata
    
    def test_generate_feature_summary(self):
        """Test feature summary generation."""
        summary = self.generator.generate_feature_summary(self.sample_data)
        
        assert "Dataset Summary: 10 samples, 10 features" in summary
        assert "GPS Features" in summary
        assert "IMU Features" in summary
        assert "gps_x" in summary
        assert "imu_ax" in summary
    
    def test_missing_data_handling(self):
        """Test handling of missing data in metadata generation."""
        # Create data with missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[0:2, 'gps_x'] = np.nan
        data_with_missing.loc[5:7, 'hdop'] = np.nan
        
        metadata = self.generator.process(data_with_missing)
        
        # Check missing data is properly reported
        gps_x_info = metadata['features']['gps_x']
        assert gps_x_info['missing_count'] == 3
        assert gps_x_info['missing_percentage'] == 30.0
        
        hdop_info = metadata['features']['hdop']
        assert hdop_info['missing_count'] == 3
        assert hdop_info['missing_percentage'] == 30.0
        
        # Check overall completeness metrics
        completeness = metadata['data_quality']['completeness']
        assert completeness['total_missing_values'] == 6
        assert completeness['complete_rows'] < 10
    
    def test_feature_statistics_calculation(self):
        """Test detailed feature statistics calculation."""
        # Test continuous feature statistics
        gps_x_stats = self.generator._calculate_feature_statistics(self.sample_data['gps_x'])
        
        assert gps_x_stats['count'] == 10
        assert gps_x_stats['missing_count'] == 0
        assert gps_x_stats['mean'] == 104.5
        assert gps_x_stats['min'] == 100.0
        assert gps_x_stats['max'] == 109.0
        
        # Test categorical feature statistics
        fix_type_stats = self.generator._calculate_feature_statistics(self.sample_data['fix_type'])
        
        assert fix_type_stats['unique_count'] == 1
        assert fix_type_stats['most_frequent'] == '3'
        assert fix_type_stats['most_frequent_count'] == 10
        assert fix_type_stats['unique_values'] == ['3']
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        empty_data = pd.DataFrame()
        
        # Should not raise an exception
        metadata = self.generator.process(empty_data)
        
        assert metadata['dataset_info']['total_samples'] == 0
        assert metadata['dataset_info']['total_features'] == 0
        assert metadata['features'] == {}
    
    def test_unknown_features_handling(self):
        """Test handling of features not in predefined descriptions."""
        # Add unknown feature
        data_with_unknown = self.sample_data.copy()
        data_with_unknown['unknown_feature'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        metadata = self.generator.process(data_with_unknown)
        
        # Check unknown feature is handled gracefully
        unknown_info = metadata['features']['unknown_feature']
        assert unknown_info['description'] == 'Feature: unknown_feature'
        assert unknown_info['unit'] == 'unknown'
        assert unknown_info['source'] == 'unknown'
        assert unknown_info['type'] == 'continuous'  # Should detect as numeric