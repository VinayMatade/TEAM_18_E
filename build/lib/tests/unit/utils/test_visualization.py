"""
Tests for TrajectoryVisualizer utility.

Tests trajectory plotting, error distribution visualizations,
and motion segment visualization functionality.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

from uav_log_processor.utils.visualization import TrajectoryVisualizer


class TestTrajectoryVisualizer:
    """Test cases for TrajectoryVisualizer utility."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = {
            'accel_threshold': 0.5,
            'target_frequency': 15.0
        }
        self.visualizer = TrajectoryVisualizer(self.config)
        
        # Create sample GPS trajectory data
        self.gps_data = pd.DataFrame({
            'timestamp': np.linspace(0, 60, 100),  # 1 minute of data
            'gps_x': 100 + np.cumsum(np.random.normal(0, 0.1, 100)),
            'gps_y': 200 + np.cumsum(np.random.normal(0, 0.1, 100)),
            'gps_z': 300 + np.random.normal(0, 0.5, 100),
            'imu_ax': np.random.normal(0, 0.5, 100),
            'imu_ay': np.random.normal(0, 0.5, 100),
            'imu_az': np.random.normal(9.8, 0.2, 100),
            'imu_gx': np.random.normal(0, 0.1, 100),
            'imu_gy': np.random.normal(0, 0.1, 100),
            'imu_gz': np.random.normal(0, 0.1, 100),
            'hdop': np.random.uniform(1.0, 3.0, 100),
            'fix_type': np.random.choice([2, 3, 4], 100, p=[0.1, 0.8, 0.1]),
            'gps_error_norm': np.random.exponential(0.5, 100)
        })
        
        # Create ground truth data
        self.ground_truth = pd.DataFrame({
            'timestamp': self.gps_data['timestamp'],
            'ground_truth_x': self.gps_data['gps_x'] + np.random.normal(0, 0.05, 100),
            'ground_truth_y': self.gps_data['gps_y'] + np.random.normal(0, 0.05, 100),
            'ground_truth_z': self.gps_data['gps_z'] + np.random.normal(0, 0.05, 100)
        })
        
        # Add ground truth to GPS data for some tests
        self.gps_with_truth = self.gps_data.copy()
        self.gps_with_truth['ground_truth_x'] = self.ground_truth['ground_truth_x']
        self.gps_with_truth['ground_truth_y'] = self.ground_truth['ground_truth_y']
        self.gps_with_truth['ground_truth_z'] = self.ground_truth['ground_truth_z']
        
        # Create temporary directory for file operations
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up after each test method."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close('all')  # Close all matplotlib figures
    
    def test_initialization(self):
        """Test TrajectoryVisualizer initialization."""
        visualizer = TrajectoryVisualizer()
        assert visualizer.config == {}
        
        visualizer_with_config = TrajectoryVisualizer(self.config)
        assert visualizer_with_config.config == self.config
    
    def test_plot_trajectory_basic(self):
        """Test basic trajectory plotting without ground truth."""
        output_path = Path(self.temp_dir) / "test_trajectory.png"
        
        result_path = self.visualizer.plot_trajectory(
            self.gps_data, 
            output_path=str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
        assert Path(result_path).suffix == '.png'
        
        # Check file size is reasonable (not empty)
        assert Path(result_path).stat().st_size > 1000  # At least 1KB
    
    def test_plot_trajectory_with_ground_truth(self):
        """Test trajectory plotting with ground truth comparison."""
        output_path = Path(self.temp_dir) / "test_trajectory_with_truth.png"
        
        result_path = self.visualizer.plot_trajectory(
            self.gps_with_truth,
            self.ground_truth,
            str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
        assert Path(result_path).suffix == '.png'
        
        # Check file size is reasonable
        assert Path(result_path).stat().st_size > 1000
    
    def test_plot_trajectory_default_path(self):
        """Test trajectory plotting with default output path."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            result_path = self.visualizer.plot_trajectory(self.gps_data)
            
            # Check default filename is used
            assert result_path == "trajectory_visualization.png"
            mock_savefig.assert_called_once()
    
    def test_create_error_analysis_plot(self):
        """Test error analysis visualization creation."""
        output_path = Path(self.temp_dir) / "test_error_analysis.png"
        
        result_path = self.visualizer.create_error_analysis_plot(
            self.gps_data,
            str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
        assert Path(result_path).suffix == '.png'
        
        # Check file size is reasonable
        assert Path(result_path).stat().st_size > 1000
    
    def test_create_error_analysis_without_error_data(self):
        """Test error analysis with missing error data."""
        data_without_error = self.gps_data.drop(columns=['gps_error_norm'])
        
        with pytest.raises(ValueError, match="GPS error data not available"):
            self.visualizer.create_error_analysis_plot(data_without_error)
    
    def test_create_motion_visualization(self):
        """Test motion segment visualization creation."""
        output_path = Path(self.temp_dir) / "test_motion.png"
        
        # Create motion labels
        motion_labels = pd.Series(
            np.random.choice(['stationary', 'moving'], len(self.gps_data)),
            index=self.gps_data.index
        )
        
        result_path = self.visualizer.create_motion_visualization(
            self.gps_data,
            motion_labels,
            str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
        assert Path(result_path).suffix == '.png'
        
        # Check file size is reasonable
        assert Path(result_path).stat().st_size > 1000
    
    def test_create_motion_visualization_without_labels(self):
        """Test motion visualization without motion labels."""
        output_path = Path(self.temp_dir) / "test_motion_no_labels.png"
        
        result_path = self.visualizer.create_motion_visualization(
            self.gps_data,
            output_path=str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
        assert Path(result_path).suffix == '.png'
    
    def test_create_motion_visualization_default_path(self):
        """Test motion visualization with default output path."""
        with patch('matplotlib.pyplot.savefig') as mock_savefig:
            result_path = self.visualizer.create_motion_visualization(self.gps_data)
            
            # Check default filename is used
            assert result_path == "motion_analysis.png"
            mock_savefig.assert_called_once()
    
    def test_plot_trajectory_missing_gps_data(self):
        """Test trajectory plotting with missing GPS coordinates."""
        data_without_gps = self.gps_data.drop(columns=['gps_x', 'gps_y'])
        output_path = Path(self.temp_dir) / "test_no_gps.png"
        
        # Should not raise an exception
        result_path = self.visualizer.plot_trajectory(
            data_without_gps,
            output_path=str(output_path)
        )
        
        # Check file was still created
        assert Path(result_path).exists()
    
    def test_plot_trajectory_missing_timestamp(self):
        """Test trajectory plotting without timestamp data."""
        data_without_time = self.gps_data.drop(columns=['timestamp'])
        output_path = Path(self.temp_dir) / "test_no_time.png"
        
        # Should not raise an exception
        result_path = self.visualizer.plot_trajectory(
            data_without_time,
            output_path=str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
    
    def test_plot_trajectory_missing_imu_data(self):
        """Test trajectory plotting without IMU data."""
        data_without_imu = self.gps_data.drop(columns=[
            'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz'
        ])
        output_path = Path(self.temp_dir) / "test_no_imu.png"
        
        # Should not raise an exception
        result_path = self.visualizer.plot_trajectory(
            data_without_imu,
            output_path=str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
    
    def test_error_analysis_statistics_accuracy(self):
        """Test accuracy of error statistics in visualization."""
        # Create data with known error statistics
        known_errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        test_data = pd.DataFrame({
            'gps_error_norm': known_errors,
            'timestamp': [1, 2, 3, 4, 5],
            'hdop': [1.5, 1.6, 1.7, 1.8, 1.9]
        })
        
        output_path = Path(self.temp_dir) / "test_error_stats.png"
        
        # This should not raise an exception and should handle the statistics correctly
        result_path = self.visualizer.create_error_analysis_plot(
            test_data,
            str(output_path)
        )
        
        assert Path(result_path).exists()
    
    def test_motion_visualization_with_boolean_labels(self):
        """Test motion visualization with boolean motion labels."""
        output_path = Path(self.temp_dir) / "test_motion_bool.png"
        
        # Create boolean motion labels
        motion_labels = pd.Series(
            np.random.choice([True, False], len(self.gps_data)),
            index=self.gps_data.index
        )
        
        result_path = self.visualizer.create_motion_visualization(
            self.gps_data,
            motion_labels,
            str(output_path)
        )
        
        # Check file was created
        assert Path(result_path).exists()
    
    def test_visualization_with_minimal_data(self):
        """Test visualizations with minimal data (single point)."""
        minimal_data = pd.DataFrame({
            'timestamp': [1.0],
            'gps_x': [100.0],
            'gps_y': [200.0],
            'gps_z': [300.0],
            'gps_error_norm': [0.5]
        })
        
        output_path = Path(self.temp_dir) / "test_minimal.png"
        
        # Should handle minimal data gracefully
        result_path = self.visualizer.plot_trajectory(
            minimal_data,
            output_path=str(output_path)
        )
        
        assert Path(result_path).exists()
    
    def test_visualization_with_nan_values(self):
        """Test visualizations with NaN values in data."""
        data_with_nan = self.gps_data.copy()
        data_with_nan.loc[10:20, 'gps_x'] = np.nan
        data_with_nan.loc[30:40, 'gps_error_norm'] = np.nan
        
        output_path = Path(self.temp_dir) / "test_with_nan.png"
        
        # Should handle NaN values gracefully
        result_path = self.visualizer.plot_trajectory(
            data_with_nan,
            output_path=str(output_path)
        )
        
        assert Path(result_path).exists()
    
    def test_large_dataset_handling(self):
        """Test visualization with larger datasets."""
        # Create larger dataset
        large_data = pd.DataFrame({
            'timestamp': np.linspace(0, 3600, 10000),  # 1 hour of data
            'gps_x': 100 + np.cumsum(np.random.normal(0, 0.1, 10000)),
            'gps_y': 200 + np.cumsum(np.random.normal(0, 0.1, 10000)),
            'gps_z': 300 + np.random.normal(0, 0.5, 10000),
            'gps_error_norm': np.random.exponential(0.5, 10000)
        })
        
        output_path = Path(self.temp_dir) / "test_large.png"
        
        # Should handle large datasets without issues
        result_path = self.visualizer.plot_trajectory(
            large_data,
            output_path=str(output_path)
        )
        
        assert Path(result_path).exists()
        # File should be reasonably sized (not too large)
        assert Path(result_path).stat().st_size < 10 * 1024 * 1024  # Less than 10MB
    
    @patch('matplotlib.pyplot.savefig')
    def test_plot_parameters(self, mock_savefig):
        """Test that plots are saved with correct parameters."""
        self.visualizer.plot_trajectory(self.gps_data)
        
        # Check savefig was called with expected parameters
        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        assert call_args[1]['dpi'] == 300
        assert call_args[1]['bbox_inches'] == 'tight'
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrames."""
        empty_data = pd.DataFrame()
        output_path = Path(self.temp_dir) / "test_empty.png"
        
        # Should handle empty data gracefully without crashing
        result_path = self.visualizer.plot_trajectory(
            empty_data,
            output_path=str(output_path)
        )
        
        assert Path(result_path).exists()