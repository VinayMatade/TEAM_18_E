"""
Tests for DatasetFormatter processor.

Tests feature standardization, normalization, dataset splitting,
and CSV output generation functionality.
"""

import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from uav_log_processor.processors.dataset_formatter import DatasetFormatter


class TestDatasetFormatter:
    """Test cases for DatasetFormatter processor."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.formatter = DatasetFormatter()

        # Create sample synchronized data
        self.sample_data: pd.DataFrame = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            'gps_x': [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            'gps_y': [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
            'gps_z': [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0, 307.0, 308.0, 309.0],
            'imu_ax': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'imu_ay': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            'imu_az': [9.8, 9.9, 10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
            'imu_gx': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
            'imu_gy': [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11],
            'imu_gz': [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12],
            'velocity_x': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            'velocity_y': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
            'velocity_z': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'hdop': [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4],
            'vdop': [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
            'fix_type': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            'ground_truth_x': [100.1, 101.1, 102.1, 103.1, 104.1, 105.1, 106.1, 107.1, 108.1, 109.1],
            'ground_truth_y': [200.1, 201.1, 202.1, 203.1, 204.1, 205.1, 206.1, 207.1, 208.1, 209.1],
            'ground_truth_z': [300.1, 301.1, 302.1, 303.1, 304.1, 305.1, 306.1, 307.1, 308.1, 309.1],
            'gps_error_x': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'gps_error_y': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'gps_error_z': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'gps_error_norm': [0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173, 0.173]
        })

    def test_initialization(self):
        """Test DatasetFormatter initialization."""
        formatter = DatasetFormatter()
        assert formatter.config == {}
        assert formatter.normalization_stats == {}

        config = {'test_param': 'value'}
        formatter_with_config = DatasetFormatter(config)
        assert formatter_with_config.config == config

    def test_standardize_features_basic(self):
        """Test basic feature standardization with standard column names."""
        result = self.formatter.standardize_features(self.sample_data)

        # Should maintain all standard columns
        expected_columns = [
            col for col in self.formatter.STANDARD_COLUMNS if col in self.sample_data.columns]
        assert list(result.columns) == expected_columns

        # Data should be unchanged for standard columns
        pd.testing.assert_frame_equal(
            result, self.sample_data[expected_columns])

    def test_standardize_features_column_mapping(self):
        """Test feature standardization with non-standard column names."""
        # Create data with alternative column names
        alt_data = pd.DataFrame({
            'timestamp': [1.0, 2.0, 3.0],
            'lat': [200.0, 201.0, 202.0],  # Should map to gps_y
            'lon': [100.0, 101.0, 102.0],  # Should map to gps_x
            'alt': [300.0, 301.0, 302.0],  # Should map to gps_z
            'accel_x': [0.1, 0.2, 0.3],    # Should map to imu_ax
            'gyro_y': [0.02, 0.03, 0.04],  # Should map to imu_gy
            'fix': [3, 3, 3]               # Should map to fix_type
        })

        result = self.formatter.standardize_features(alt_data)

        # Check that columns were mapped correctly
        assert 'gps_x' in result.columns
        assert 'gps_y' in result.columns
        assert 'gps_z' in result.columns
        assert 'imu_ax' in result.columns
        assert 'imu_gy' in result.columns
        assert 'fix_type' in result.columns

        # Check data values were preserved
        assert result['gps_x'].iloc[0] == 100.0  # From 'lon'
        assert result['gps_y'].iloc[0] == 200.0  # From 'lat'
        assert result['gps_z'].iloc[0] == 300.0  # From 'alt'

    def test_handle_missing_features(self):
        """Test graceful handling of missing features."""
        # Create data missing some features
        partial_data = self.sample_data[[
            'timestamp', 'gps_x', 'gps_y', 'gps_z']].copy()

        result = self.formatter.standardize_features(partial_data)

        # Should only include available columns
        expected_columns = ['timestamp', 'gps_x', 'gps_y', 'gps_z']
        assert list(result.columns) == expected_columns

    def test_normalize_features_basic(self):
        """Test Z-score normalization of continuous features."""
        normalized_data, stats = self.formatter.normalize_features(
            self.sample_data)

        # Check that normalization stats were calculated
        assert len(stats) > 0

        # Check specific feature normalization
        for feature in self.formatter.NORMALIZATION_FEATURES:
            if feature in self.sample_data.columns:
                assert feature in stats
                assert 'mean' in stats[feature]
                assert 'std' in stats[feature]
                assert 'normalized' in stats[feature]

                if stats[feature]['normalized']:
                    # Normalized feature should have mean ≈ 0 and std ≈ 1
                    normalized_mean = normalized_data[feature].mean()
                    normalized_std = normalized_data[feature].std()
                    # Should be very close to 0
                    assert abs(normalized_mean) < 1e-10
                    # Should be very close to 1
                    assert abs(normalized_std - 1.0) < 1e-10

    def test_normalize_features_constant_values(self):
        """Test normalization handling of constant features."""
        # Create data with constant feature
        constant_data = self.sample_data.copy()
        constant_data['gps_x'] = 100.0  # All same value

        normalized_data, stats = self.formatter.normalize_features(
            constant_data)

        # Constant feature should not be normalized
        assert stats['gps_x']['normalized'] is False
        # Default std for constant features
        assert stats['gps_x']['std'] == 1.0

        # Original values should be preserved
        assert (normalized_data['gps_x'] == 100.0).all()

    def test_categorical_feature_encoding(self):
        """Test encoding of categorical features."""
        normalized_data, stats = self.formatter.normalize_features(
            self.sample_data)

        # fix_type should be encoded as integer
        assert normalized_data['fix_type'].dtype == int
        assert 'fix_type' in stats
        assert stats['fix_type']['type'] == 'ordinal'
        assert stats['fix_type']['encoded'] is True

    def test_split_dataset_single_flight(self):
        """Test dataset splitting for single flight."""
        train_df, val_df, test_df = self.formatter.split_dataset(
            self.sample_data)

        # Check split sizes (approximately 70/15/15)
        total_samples = len(self.sample_data)
        assert len(train_df) == int(total_samples * 0.7)
        assert len(val_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(val_df) + len(test_df) == total_samples

        # Check sequential ordering (no overlap)
        assert train_df['timestamp'].max() <= val_df['timestamp'].min()
        assert val_df['timestamp'].max() <= test_df['timestamp'].min()

    def test_split_dataset_custom_ratios(self):
        """Test dataset splitting with custom ratios."""
        train_df, val_df, test_df = self.formatter.split_dataset(
            self.sample_data, train_ratio=0.6, val_ratio=0.2
        )

        total_samples = len(self.sample_data)

        # Check approximate ratios (allowing for rounding)
        train_ratio = len(train_df) / total_samples
        val_ratio = len(val_df) / total_samples
        test_ratio = len(test_df) / total_samples

        assert abs(train_ratio - 0.6) < 0.1
        assert abs(val_ratio - 0.2) < 0.1
        assert abs(test_ratio - 0.2) < 0.1

    def test_split_dataset_multiple_flights(self):
        """Test dataset splitting with multiple flight segments."""
        # Create data with time gap (simulating multiple flights)
        multi_flight_data = self.sample_data.copy()

        # Add second flight with large time gap
        second_flight = self.sample_data.copy()
        second_flight['timestamp'] += 1000  # Large gap

        combined_data = pd.concat(
            [multi_flight_data, second_flight], ignore_index=True)

        train_df, val_df, test_df = self.formatter.split_dataset(combined_data)

        # Should have data from both flights in each split
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        assert len(train_df) + len(val_df) + len(test_df) == len(combined_data)

    def test_validate_split_integrity(self):
        """Test validation of dataset split temporal integrity."""
        train_df, val_df, test_df = self.formatter.split_dataset(
            self.sample_data)

        # Should pass validation (no temporal overlap)
        is_valid = self.formatter.validate_split_integrity(
            train_df, val_df, test_df)
        assert is_valid is True

    def test_validate_split_integrity_no_timestamp(self):
        """Test split validation without timestamp column."""
        # Remove timestamp column
        no_timestamp_data = self.sample_data.drop('timestamp', axis=1)
        train_df, val_df, test_df = self.formatter.split_dataset(
            no_timestamp_data)

        # Should still pass validation (warning logged)
        is_valid = self.formatter.validate_split_integrity(
            train_df, val_df, test_df)
        assert is_valid is True

    def test_save_datasets(self):
        """Test CSV output generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            train_df, val_df, test_df = self.formatter.split_dataset(
                self.sample_data)

            saved_paths = self.formatter.save_datasets(
                train_df, val_df, test_df, output_dir=temp_dir
            )

            # Check that all files were created
            assert 'train' in saved_paths
            assert 'valid' in saved_paths
            assert 'test' in saved_paths

            # Check files exist and are readable
            for split_name, file_path in saved_paths.items():
                assert Path(file_path).exists()

                # Try reading the file
                loaded_df = pd.read_csv(file_path)
                assert len(loaded_df) > 0
                assert list(loaded_df.columns) == list(train_df.columns)

    def test_save_large_dataset_chunked(self):
        """Test chunked saving for large datasets."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create larger dataset
            large_data = pd.concat([self.sample_data] * 100, ignore_index=True)
            train_df, val_df, test_df = self.formatter.split_dataset(
                large_data)

            # Save with small chunk size to test chunking
            saved_paths = self.formatter.save_datasets(
                train_df, val_df, test_df,
                output_dir=temp_dir,
                chunk_size=50
            )

            # Verify files were saved correctly
            for split_name, file_path in saved_paths.items():
                loaded_df = pd.read_csv(file_path)

                if split_name == 'train':
                    assert len(loaded_df) == len(train_df)
                elif split_name == 'valid':
                    assert len(loaded_df) == len(val_df)
                elif split_name == 'test':
                    assert len(loaded_df) == len(test_df)

    def test_generate_dataset_summary(self):
        """Test dataset summary generation."""
        train_df, val_df, test_df = self.formatter.split_dataset(
            self.sample_data)

        summary = self.formatter.generate_dataset_summary(
            train_df, val_df, test_df)

        # Check summary structure
        assert 'total_samples' in summary
        assert 'splits' in summary
        assert 'features' in summary
        assert 'feature_count' in summary
        assert 'temporal_ranges' in summary

        # Check split information
        assert 'train' in summary['splits']
        assert 'validation' in summary['splits']
        assert 'test' in summary['splits']

        # Check that percentages sum to 100
        total_percentage = (
            summary['splits']['train']['percentage'] +
            summary['splits']['validation']['percentage'] +
            summary['splits']['test']['percentage']
        )
        assert abs(total_percentage - 100.0) < 0.1

    def test_format_dataset_complete_pipeline(self):
        """Test complete dataset formatting pipeline."""
        formatted_data, metadata = self.formatter.format_dataset(
            self.sample_data)

        # Check that data was processed
        assert len(formatted_data) == len(self.sample_data)

        # Check metadata structure
        assert 'total_samples' in metadata
        assert 'features' in metadata
        assert 'normalization_stats' in metadata
        assert 'categorical_features' in metadata
        assert 'continuous_features' in metadata

        # Check that normalization was applied
        assert len(metadata['normalization_stats']) > 0

    def test_denormalize_features(self):
        """Test feature denormalization."""
        # Normalize first
        normalized_data, stats = self.formatter.normalize_features(
            self.sample_data)

        # Then denormalize
        denormalized_data = self.formatter.denormalize_features(
            normalized_data, stats)

        # Should be close to original data for normalized features
        for feature in self.formatter.NORMALIZATION_FEATURES:
            if feature in self.sample_data.columns and stats[feature]['normalized']:
                original_values = self.sample_data[feature]
                denormalized_values = denormalized_data[feature]

                # Should be very close (within floating point precision)
                np.testing.assert_allclose(
                    original_values, denormalized_values, rtol=1e-10
                )
