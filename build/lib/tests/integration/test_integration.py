"""
End-to-end integration tests for UAV Log Processor.

Tests the complete pipeline with real flight data scenarios.
"""

import unittest
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from uav_log_processor import UAVLogProcessor, ProcessingConfig
from uav_log_processor.cli import main as cli_main


class TestEndToEndProcessing(unittest.TestCase):
    """Test complete pipeline processing with synthetic data."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create test configuration
        self.config = ProcessingConfig(
            target_frequency=10.0,  # Lower frequency for faster tests
            output_dir=str(self.output_dir),
            verbose=False,
            save_intermediate=True,
            create_visualizations=False,  # Skip visualizations for speed
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        # Create synthetic log files
        self._create_synthetic_logs()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_synthetic_logs(self):
        """Create synthetic log files for testing."""
        # Create synthetic TLOG data (simplified)
        tlog_path = Path(self.temp_dir) / "test_flight.tlog"
        with open(tlog_path, 'wb') as f:
            # Write minimal TLOG header (simplified for testing)
            f.write(b'\xfe\x1c\x00\x00\x01\x00\x00GLOBAL_POSITION_INT')
        
        # Create synthetic CSV log data
        csv_path = Path(self.temp_dir) / "test_flight.txt"
        csv_data = self._generate_synthetic_flight_data()
        csv_data.to_csv(csv_path, index=False)
        
        self.log_files = [str(tlog_path), str(csv_path)]
    
    def _generate_synthetic_flight_data(self, duration=60, frequency=15):
        """Generate synthetic flight data for testing."""
        n_samples = int(duration * frequency)
        timestamps = np.linspace(0, duration, n_samples)
        
        # Generate GPS trajectory (circular pattern with noise)
        t = timestamps / 10.0  # Scale time for circular motion
        base_lat = 37.123456
        base_lon = -122.123456
        base_alt = 100.0
        
        # Circular trajectory
        radius = 0.001  # ~100m radius in degrees
        gps_lat = base_lat + radius * np.cos(t) + np.random.normal(0, 0.00001, n_samples)
        gps_lon = base_lon + radius * np.sin(t) + np.random.normal(0, 0.00001, n_samples)
        gps_alt = base_alt + 10 * np.sin(t/2) + np.random.normal(0, 0.5, n_samples)
        
        # Generate IMU data
        # Acceleration (with gravity and motion)
        imu_ax = 0.5 * np.sin(t) + np.random.normal(0, 0.1, n_samples)
        imu_ay = 0.5 * np.cos(t) + np.random.normal(0, 0.1, n_samples)
        imu_az = 9.8 + np.random.normal(0, 0.2, n_samples)  # Gravity + noise
        
        # Gyroscope (angular velocity)
        imu_gx = 0.1 * np.cos(t) + np.random.normal(0, 0.02, n_samples)
        imu_gy = 0.1 * np.sin(t) + np.random.normal(0, 0.02, n_samples)
        imu_gz = 0.2 + np.random.normal(0, 0.01, n_samples)  # Yaw rate
        
        # Velocity (derived from position)
        velocity_x = np.gradient(gps_lon) * frequency * 111320  # Approximate m/s
        velocity_y = np.gradient(gps_lat) * frequency * 111320
        velocity_z = np.gradient(gps_alt) * frequency
        
        # GPS quality indicators
        hdop = 1.0 + 0.5 * np.random.random(n_samples)
        vdop = 1.2 + 0.3 * np.random.random(n_samples)
        fix_type = np.full(n_samples, 3.0)  # 3D fix
        
        # Add some stationary periods (first 10 seconds and last 10 seconds)
        stationary_mask = (timestamps < 10) | (timestamps > 50)
        imu_ax[stationary_mask] = np.random.normal(0, 0.05, np.sum(stationary_mask))
        imu_ay[stationary_mask] = np.random.normal(0, 0.05, np.sum(stationary_mask))
        imu_gx[stationary_mask] = np.random.normal(0, 0.01, np.sum(stationary_mask))
        imu_gy[stationary_mask] = np.random.normal(0, 0.01, np.sum(stationary_mask))
        imu_gz[stationary_mask] = np.random.normal(0, 0.01, np.sum(stationary_mask))
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'lat': gps_lat,
            'lon': gps_lon,
            'alt': gps_alt,
            'acc_x': imu_ax,
            'acc_y': imu_ay,
            'acc_z': imu_az,
            'gyro_x': imu_gx,
            'gyro_y': imu_gy,
            'gyro_z': imu_gz,
            'vel_x': velocity_x,
            'vel_y': velocity_y,
            'vel_z': velocity_z,
            'hdop': hdop,
            'vdop': vdop,
            'fix_type': fix_type
        })
    
    @patch('uav_log_processor.parsers.tlog_parser.TLogParser.parse')
    def test_complete_pipeline_processing(self, mock_tlog_parse):
        """Test complete pipeline from log files to output datasets."""
        # Mock TLOG parser to return synthetic data
        synthetic_data = self._generate_synthetic_flight_data()
        # Rename columns to match parser output format
        synthetic_data = synthetic_data.rename(columns={
            'lat': 'gps_lat',
            'lon': 'gps_lon', 
            'alt': 'gps_alt',
            'acc_x': 'imu_ax',
            'acc_y': 'imu_ay',
            'acc_z': 'imu_az',
            'gyro_x': 'imu_gx',
            'gyro_y': 'imu_gy',
            'gyro_z': 'imu_gz',
            'vel_x': 'velocity_x',
            'vel_y': 'velocity_y',
            'vel_z': 'velocity_z'
        })
        mock_tlog_parse.return_value = synthetic_data
        
        # Initialize processor
        processor = UAVLogProcessor(self.config)
        
        # Process logs
        results = processor.process_logs([self.log_files[1]])  # Use only CSV file
        
        # Verify results structure
        self.assertIn('output_files', results)
        self.assertIn('statistics', results)
        self.assertIn('metadata', results)
        
        # Check output files exist
        expected_files = ['train.csv', 'validation.csv', 'test.csv', 'metadata.json']
        if self.config.save_intermediate:
            expected_files.append('aligned_full.csv')
        
        for filename in expected_files:
            file_path = self.output_dir / filename
            self.assertTrue(file_path.exists(), f"Output file {filename} not found")
            self.assertIn(str(file_path), results['output_files'])
        
        # Verify dataset files
        train_df = pd.read_csv(self.output_dir / 'train.csv')
        val_df = pd.read_csv(self.output_dir / 'validation.csv')
        test_df = pd.read_csv(self.output_dir / 'test.csv')
        
        # Check dataset sizes match expected ratios
        total_samples = len(train_df) + len(val_df) + len(test_df)
        self.assertGreater(total_samples, 0)
        
        train_ratio = len(train_df) / total_samples
        val_ratio = len(val_df) / total_samples
        test_ratio = len(test_df) / total_samples
        
        self.assertAlmostEqual(train_ratio, self.config.train_ratio, delta=0.05)
        self.assertAlmostEqual(val_ratio, self.config.val_ratio, delta=0.05)
        self.assertAlmostEqual(test_ratio, self.config.test_ratio, delta=0.05)
        
        # Check required columns exist
        required_columns = [
            'timestamp', 'gps_x', 'gps_y', 'gps_z',
            'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz',
            'velocity_x', 'velocity_y', 'velocity_z', 'hdop', 'vdop', 'fix_type',
            'ground_truth_x', 'ground_truth_y', 'ground_truth_z',
            'gps_error_x', 'gps_error_y', 'gps_error_z', 'gps_error_norm'
        ]
        
        for col in required_columns:
            self.assertIn(col, train_df.columns, f"Column {col} missing from train dataset")
            self.assertIn(col, val_df.columns, f"Column {col} missing from validation dataset")
            self.assertIn(col, test_df.columns, f"Column {col} missing from test dataset")
        
        # Verify metadata
        with open(self.output_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.assertIn('features', metadata)
        self.assertIn('normalization_stats', metadata)
        self.assertIn('processing_config', metadata)
        
        # Check statistics
        stats = results['statistics']
        self.assertIn('total_samples', stats)
        self.assertIn('dataset_sizes', stats)
        self.assertIn('error_statistics', stats)
        
        self.assertEqual(stats['total_samples'], total_samples)
        self.assertGreater(stats['error_statistics']['mean_error_m'], 0)
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline handling of missing data."""
        # Create data with significant gaps
        synthetic_data = self._generate_synthetic_flight_data(duration=30, frequency=5)
        
        # Introduce missing data
        missing_indices = np.random.choice(len(synthetic_data), size=len(synthetic_data)//3, replace=False)
        synthetic_data.loc[missing_indices, ['lat', 'lon', 'alt']] = np.nan
        
        # Save to file
        csv_path = Path(self.temp_dir) / "missing_data.txt"
        synthetic_data.to_csv(csv_path, index=False)
        
        # Process with relaxed configuration
        config = self.config.copy()
        config.min_data_coverage = 0.3  # Allow more missing data
        
        processor = UAVLogProcessor(config)
        results = processor.process_logs([str(csv_path)])
        
        # Should still produce output despite missing data
        self.assertIn('output_files', results)
        self.assertTrue(len(results['output_files']) > 0)
    
    def test_pipeline_performance_large_dataset(self):
        """Test pipeline performance with larger dataset."""
        # Create larger synthetic dataset
        large_data = self._generate_synthetic_flight_data(duration=300, frequency=15)  # 5 minutes at 15Hz
        
        csv_path = Path(self.temp_dir) / "large_flight.txt"
        large_data.to_csv(csv_path, index=False)
        
        # Process with chunking enabled
        config = self.config.copy()
        config.chunk_size = 1000
        
        processor = UAVLogProcessor(config)
        
        import time
        start_time = time.time()
        results = processor.process_logs([str(csv_path)])
        processing_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 30.0, "Processing took too long")
        
        # Verify output quality
        self.assertIn('statistics', results)
        stats = results['statistics']
        self.assertGreater(stats['total_samples'], 1000)
    
    def test_multiple_file_processing(self):
        """Test processing multiple log files simultaneously."""
        # Create multiple synthetic files
        file_paths = []
        for i in range(3):
            data = self._generate_synthetic_flight_data(duration=30, frequency=10)
            # Add slight offset to each flight
            data['lat'] += i * 0.001
            data['lon'] += i * 0.001
            
            file_path = Path(self.temp_dir) / f"flight_{i}.txt"
            data.to_csv(file_path, index=False)
            file_paths.append(str(file_path))
        
        processor = UAVLogProcessor(self.config)
        results = processor.process_logs(file_paths)
        
        # Should combine all flights into single dataset
        self.assertIn('statistics', results)
        stats = results['statistics']
        
        # Total samples should be approximately 3x single flight
        expected_samples = 3 * 30 * 10  # 3 flights * 30 seconds * 10 Hz
        self.assertGreater(stats['total_samples'], expected_samples * 0.8)  # Allow for some data loss
    
    def test_configuration_validation(self):
        """Test configuration validation and error handling."""
        # Test invalid configuration
        with self.assertRaises(ValueError):
            ProcessingConfig(target_frequency=-1.0)
        
        with self.assertRaises(ValueError):
            ProcessingConfig(train_ratio=1.5)
        
        with self.assertRaises(ValueError):
            ProcessingConfig(train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)  # Sum > 1
        
        # Test configuration file loading
        config_path = Path(self.temp_dir) / "test_config.json"
        config_data = {
            "target_frequency": 20.0,
            "accel_threshold": 0.3,
            "output_dir": str(self.output_dir)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        loaded_config = ProcessingConfig.from_file(str(config_path))
        self.assertEqual(loaded_config.target_frequency, 20.0)
        self.assertEqual(loaded_config.accel_threshold, 0.3)


class TestCLIIntegration(unittest.TestCase):
    """Test command-line interface integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "cli_output"
        
        # Create test log file
        self.test_file = Path(self.temp_dir) / "test_flight.txt"
        test_data = pd.DataFrame({
            'timestamp': np.linspace(0, 30, 300),
            'lat': 37.123456 + np.random.normal(0, 0.00001, 300),
            'lon': -122.123456 + np.random.normal(0, 0.00001, 300),
            'alt': 100.0 + np.random.normal(0, 0.5, 300),
            'acc_x': np.random.normal(0, 0.1, 300),
            'acc_y': np.random.normal(0, 0.1, 300),
            'acc_z': 9.8 + np.random.normal(0, 0.2, 300),
            'gyro_x': np.random.normal(0, 0.02, 300),
            'gyro_y': np.random.normal(0, 0.02, 300),
            'gyro_z': np.random.normal(0, 0.01, 300)
        })
        test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    @patch('sys.argv')
    def test_cli_basic_processing(self, mock_argv):
        """Test basic CLI processing."""
        mock_argv.__getitem__.side_effect = [
            'uav-log-processor',  # Program name
            str(self.test_file),  # Input file
            '--output', str(self.output_dir),
            '--quiet'
        ]
        mock_argv.__len__.return_value = 5
        
        # Mock sys.argv properly
        with patch('sys.argv', ['uav-log-processor', str(self.test_file), '--output', str(self.output_dir), '--quiet']):
            try:
                result = cli_main()
                self.assertEqual(result, 0)  # Success
                
                # Check output files
                self.assertTrue(self.output_dir.exists())
                self.assertTrue((self.output_dir / 'train.csv').exists())
                self.assertTrue((self.output_dir / 'metadata.json').exists())
                
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Success exit
    
    @patch('sys.argv')
    def test_cli_configuration_file(self, mock_argv):
        """Test CLI with configuration file."""
        # Create config file
        config_path = Path(self.temp_dir) / "config.json"
        config_data = {
            "target_frequency": 5.0,
            "save_intermediate": True,
            "create_visualizations": False
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f)
        
        with patch('sys.argv', [
            'uav-log-processor',
            str(self.test_file),
            '--config', str(config_path),
            '--output', str(self.output_dir),
            '--quiet'
        ]):
            try:
                result = cli_main()
                self.assertEqual(result, 0)
                
                # Check that intermediate file was saved (from config)
                self.assertTrue((self.output_dir / 'aligned_full.csv').exists())
                
            except SystemExit as e:
                self.assertEqual(e.code, 0)
    
    @patch('sys.argv')
    def test_cli_input_directory(self, mock_argv):
        """Test CLI with input directory discovery."""
        # Create multiple log files in directory
        log_dir = Path(self.temp_dir) / "logs"
        log_dir.mkdir()
        
        for i in range(2):
            test_data = pd.DataFrame({
                'timestamp': np.linspace(0, 10, 100),
                'lat': 37.123456 + i * 0.001,
                'lon': -122.123456 + i * 0.001,
                'alt': 100.0,
                'acc_x': np.random.normal(0, 0.1, 100),
                'acc_y': np.random.normal(0, 0.1, 100),
                'acc_z': 9.8,
                'gyro_x': 0.0,
                'gyro_y': 0.0,
                'gyro_z': 0.0
            })
            test_data.to_csv(log_dir / f"flight_{i}.txt", index=False)
        
        with patch('sys.argv', [
            'uav-log-processor',
            '--input-dir', str(log_dir),
            '--output', str(self.output_dir),
            '--quiet'
        ]):
            try:
                result = cli_main()
                self.assertEqual(result, 0)
                
                # Should process both files
                self.assertTrue((self.output_dir / 'train.csv').exists())
                
            except SystemExit as e:
                self.assertEqual(e.code, 0)
    
    @patch('sys.argv')
    def test_cli_validation_only(self, mock_argv):
        """Test CLI validation-only mode."""
        with patch('sys.argv', [
            'uav-log-processor',
            str(self.test_file),
            '--validate-only'
        ]):
            try:
                result = cli_main()
                self.assertEqual(result, 0)
                
                # Should not create output files
                self.assertFalse(self.output_dir.exists())
                
            except SystemExit as e:
                self.assertEqual(e.code, 0)
    
    @patch('sys.argv')
    def test_cli_error_handling(self, mock_argv):
        """Test CLI error handling."""
        # Test with non-existent file
        with patch('sys.argv', [
            'uav-log-processor',
            'nonexistent.tlog',
            '--quiet'
        ]):
            try:
                result = cli_main()
                self.assertNotEqual(result, 0)  # Should fail
            except SystemExit as e:
                self.assertNotEqual(e.code, 0)  # Should fail


class TestOutputFormatCompliance(unittest.TestCase):
    """Test output format compliance and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        
        # Create minimal test data
        self.test_data = pd.DataFrame({
            'timestamp': np.linspace(0, 10, 100),
            'lat': 37.123456 + np.random.normal(0, 0.00001, 100),
            'lon': -122.123456 + np.random.normal(0, 0.00001, 100),
            'alt': 100.0 + np.random.normal(0, 0.5, 100),
            'acc_x': np.random.normal(0, 0.1, 100),
            'acc_y': np.random.normal(0, 0.1, 100),
            'acc_z': 9.8 + np.random.normal(0, 0.2, 100),
            'gyro_x': np.random.normal(0, 0.02, 100),
            'gyro_y': np.random.normal(0, 0.02, 100),
            'gyro_z': np.random.normal(0, 0.01, 100)
        })
        
        self.test_file = Path(self.temp_dir) / "test.txt"
        self.test_data.to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_csv_format_compliance(self):
        """Test that output CSV files comply with expected format."""
        config = ProcessingConfig(
            output_dir=str(self.output_dir),
            verbose=False,
            create_visualizations=False
        )
        
        processor = UAVLogProcessor(config)
        results = processor.process_logs([str(self.test_file)])
        
        # Test each dataset file
        for dataset_name in ['train', 'validation', 'test']:
            csv_path = self.output_dir / f"{dataset_name}.csv"
            self.assertTrue(csv_path.exists())
            
            # Load and validate CSV
            df = pd.read_csv(csv_path)
            
            # Check basic structure
            self.assertGreater(len(df), 0)
            self.assertFalse(df.empty)
            
            # Check for required columns
            required_columns = [
                'timestamp', 'gps_x', 'gps_y', 'gps_z',
                'gps_error_x', 'gps_error_y', 'gps_error_z', 'gps_error_norm'
            ]
            
            for col in required_columns:
                self.assertIn(col, df.columns, f"Missing column {col} in {dataset_name}")
            
            # Check data types
            self.assertTrue(pd.api.types.is_numeric_dtype(df['timestamp']))
            self.assertTrue(pd.api.types.is_numeric_dtype(df['gps_error_norm']))
            
            # Check for NaN values in critical columns
            critical_columns = ['timestamp', 'gps_error_norm']
            for col in critical_columns:
                self.assertFalse(df[col].isna().any(), f"NaN values in {col} of {dataset_name}")
            
            # Check timestamp ordering
            self.assertTrue(df['timestamp'].is_monotonic_increasing, 
                          f"Timestamps not ordered in {dataset_name}")
    
    def test_metadata_format_compliance(self):
        """Test metadata.json format compliance."""
        config = ProcessingConfig(
            output_dir=str(self.output_dir),
            verbose=False,
            create_visualizations=False
        )
        
        processor = UAVLogProcessor(config)
        results = processor.process_logs([str(self.test_file)])
        
        metadata_path = self.output_dir / "metadata.json"
        self.assertTrue(metadata_path.exists())
        
        # Load and validate metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check required sections
        required_sections = ['features', 'normalization_stats', 'processing_config']
        for section in required_sections:
            self.assertIn(section, metadata, f"Missing section {section} in metadata")
        
        # Check features section
        features = metadata['features']
        self.assertIsInstance(features, dict)
        self.assertIn('gps_error_norm', features)
        
        # Check normalization stats
        norm_stats = metadata['normalization_stats']
        self.assertIsInstance(norm_stats, dict)
        
        # Check processing config
        proc_config = metadata['processing_config']
        self.assertIsInstance(proc_config, dict)
        self.assertIn('target_frequency', proc_config)
    
    def test_reproducibility_outputs(self):
        """Test reproducibility output compliance."""
        config = ProcessingConfig(
            output_dir=str(self.output_dir),
            save_intermediate=True,
            verbose=False,
            create_visualizations=False
        )
        
        processor = UAVLogProcessor(config)
        results = processor.process_logs([str(self.test_file)])
        
        # Check intermediate file
        aligned_path = self.output_dir / "aligned_full.csv"
        self.assertTrue(aligned_path.exists())
        
        aligned_df = pd.read_csv(aligned_path)
        self.assertGreater(len(aligned_df), 0)
        
        # Should contain raw synchronized data
        expected_columns = ['timestamp', 'gps_lat', 'gps_lon', 'gps_alt']
        for col in expected_columns:
            self.assertIn(col, aligned_df.columns)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)