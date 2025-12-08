"""Unit tests for trajectory data structures."""

import pytest
import numpy as np
from uav_log_processor.data_structures import TrajectoryData, ErrorMetrics


class TestTrajectoryData:
    """Tests for TrajectoryData dataclass."""
    
    def test_valid_trajectory_creation(self):
        """Test creating a valid trajectory."""
        timestamps = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        
        traj = TrajectoryData(
            timestamps=timestamps,
            x=x,
            y=y,
            label="Test Trajectory",
            color="blue",
            style="-"
        )
        
        assert traj.length == 3
        assert traj.label == "Test Trajectory"
        assert traj.color == "blue"
        assert traj.style == "-"
    
    def test_trajectory_with_default_style(self):
        """Test that default style is '-'."""
        timestamps = np.array([0.0])
        x = np.array([0.0])
        y = np.array([0.0])
        
        traj = TrajectoryData(
            timestamps=timestamps,
            x=x,
            y=y,
            label="Test",
            color="red"
        )
        
        assert traj.style == "-"
    
    def test_trajectory_length_mismatch(self):
        """Test that mismatched array lengths raise ValueError."""
        timestamps = np.array([0.0, 1.0])
        x = np.array([0.0, 1.0, 2.0])  # Different length
        y = np.array([0.0, 0.5])
        
        with pytest.raises(ValueError, match="same length"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue"
            )
    
    def test_trajectory_empty_arrays(self):
        """Test that empty arrays raise ValueError."""
        timestamps = np.array([])
        x = np.array([])
        y = np.array([])
        
        with pytest.raises(ValueError, match="at least one data point"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue"
            )
    
    def test_trajectory_nan_values(self):
        """Test that NaN values raise ValueError."""
        timestamps = np.array([0.0, 1.0, np.nan])
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue"
            )
    
    def test_trajectory_inf_values(self):
        """Test that Inf values raise ValueError."""
        timestamps = np.array([0.0, 1.0, 2.0])
        x = np.array([0.0, np.inf, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        
        with pytest.raises(ValueError, match="NaN or Inf"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue"
            )
    
    def test_trajectory_non_monotonic_timestamps(self):
        """Test that non-monotonic timestamps raise ValueError."""
        timestamps = np.array([0.0, 2.0, 1.0])  # Not monotonic
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 0.5, 1.0])
        
        with pytest.raises(ValueError, match="monotonically increasing"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue"
            )
    
    def test_trajectory_empty_label(self):
        """Test that empty label raises ValueError."""
        timestamps = np.array([0.0, 1.0])
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.5])
        
        with pytest.raises(ValueError, match="non-empty string"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="",
                color="blue"
            )
    
    def test_trajectory_invalid_style(self):
        """Test that invalid line style raises ValueError."""
        timestamps = np.array([0.0, 1.0])
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.5])
        
        with pytest.raises(ValueError, match="style must be one of"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue",
                style="invalid"
            )
    
    def test_trajectory_get_bounds(self):
        """Test getting trajectory bounding box."""
        timestamps = np.array([0.0, 1.0, 2.0])
        x = np.array([-1.0, 0.0, 2.0])
        y = np.array([0.5, -0.5, 1.0])
        
        traj = TrajectoryData(
            timestamps=timestamps,
            x=x,
            y=y,
            label="Test",
            color="blue"
        )
        
        min_x, max_x, min_y, max_y = traj.get_bounds()
        assert min_x == -1.0
        assert max_x == 2.0
        assert min_y == -0.5
        assert max_y == 1.0
    
    def test_trajectory_2d_arrays_rejected(self):
        """Test that 2D arrays are rejected."""
        timestamps = np.array([[0.0, 1.0]])  # 2D
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 0.5])
        
        with pytest.raises(ValueError, match="must be 1D"):
            TrajectoryData(
                timestamps=timestamps,
                x=x,
                y=y,
                label="Test",
                color="blue"
            )


class TestErrorMetrics:
    """Tests for ErrorMetrics dataclass."""
    
    def test_valid_error_metrics_creation(self):
        """Test creating valid error metrics."""
        metrics = ErrorMetrics(
            mae=1.5,
            rmse=2.0,
            max_error=5.0
        )
        
        assert metrics.mae == 1.5
        assert metrics.rmse == 2.0
        assert metrics.max_error == 5.0
        assert metrics.median_error is None
        assert metrics.std_error is None
    
    def test_error_metrics_with_optional_fields(self):
        """Test creating error metrics with optional fields."""
        metrics = ErrorMetrics(
            mae=1.5,
            rmse=2.0,
            max_error=5.0,
            median_error=1.2,
            std_error=0.8
        )
        
        assert metrics.median_error == 1.2
        assert metrics.std_error == 0.8
    
    def test_error_metrics_negative_mae(self):
        """Test that negative MAE raises ValueError."""
        with pytest.raises(ValueError, match="mae must be non-negative"):
            ErrorMetrics(
                mae=-1.0,
                rmse=2.0,
                max_error=5.0
            )
    
    def test_error_metrics_negative_rmse(self):
        """Test that negative RMSE raises ValueError."""
        with pytest.raises(ValueError, match="rmse must be non-negative"):
            ErrorMetrics(
                mae=1.0,
                rmse=-2.0,
                max_error=5.0
            )
    
    def test_error_metrics_negative_max_error(self):
        """Test that negative max_error raises ValueError."""
        with pytest.raises(ValueError, match="max_error must be non-negative"):
            ErrorMetrics(
                mae=1.0,
                rmse=2.0,
                max_error=-5.0
            )
    
    def test_error_metrics_nan_values(self):
        """Test that NaN values raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN or Inf"):
            ErrorMetrics(
                mae=np.nan,
                rmse=2.0,
                max_error=5.0
            )
    
    def test_error_metrics_inf_values(self):
        """Test that Inf values raise ValueError."""
        with pytest.raises(ValueError, match="cannot be NaN or Inf"):
            ErrorMetrics(
                mae=1.0,
                rmse=np.inf,
                max_error=5.0
            )
    
    def test_error_metrics_rmse_less_than_mae(self):
        """Test that RMSE < MAE raises ValueError."""
        with pytest.raises(ValueError, match="RMSE.*should be >= MAE"):
            ErrorMetrics(
                mae=3.0,
                rmse=2.0,  # RMSE should be >= MAE
                max_error=5.0
            )
    
    def test_error_metrics_max_less_than_mae(self):
        """Test that max_error < MAE raises ValueError."""
        with pytest.raises(ValueError, match="max_error.*should be >= MAE"):
            ErrorMetrics(
                mae=3.0,
                rmse=3.5,
                max_error=2.0  # max_error should be >= MAE
            )
    
    def test_error_metrics_str_representation(self):
        """Test string representation of error metrics."""
        metrics = ErrorMetrics(
            mae=1.5,
            rmse=2.0,
            max_error=5.0
        )
        
        str_repr = str(metrics)
        assert "MAE: 1.50 m" in str_repr
        assert "RMSE: 2.00 m" in str_repr
        assert "Max Error: 5.00 m" in str_repr
    
    def test_error_metrics_str_with_optional_fields(self):
        """Test string representation with optional fields."""
        metrics = ErrorMetrics(
            mae=1.5,
            rmse=2.0,
            max_error=5.0,
            median_error=1.2,
            std_error=0.8
        )
        
        str_repr = str(metrics)
        assert "Median Error: 1.20 m" in str_repr
        assert "Std Dev: 0.80 m" in str_repr
