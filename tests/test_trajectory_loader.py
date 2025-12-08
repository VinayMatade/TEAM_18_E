"""Unit tests for trajectory loading and preprocessing."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from uav_log_processor.trajectory_loader import (
    load_trajectory_csv,
    validate_trajectory_data,
    has_rtk_data,
    extract_gps_trajectory,
    extract_rtk_trajectory,
    extract_imu_data,
    REQUIRED_COLUMNS,
    OPTIONAL_COLUMNS
)


@pytest.fixture
def valid_trajectory_data():
    """Create a valid trajectory DataFrame for testing."""
    n_samples = 100
    data = {
        'GPS_Lat': np.linspace(15.366, 15.367, n_samples),
        'GPS_Lng': np.linspace(75.125, 75.126, n_samples),
        'IMU_AccX': np.random.randn(n_samples) * 0.5,
        'IMU_AccY': np.random.randn(n_samples) * 0.5,
        'IMU_AccZ': np.random.randn(n_samples) * 0.5 - 9.8,
        'IMU_GyrX': np.random.randn(n_samples) * 0.1,
        'IMU_GyrY': np.random.randn(n_samples) * 0.1,
        'IMU_GyrZ': np.random.randn(n_samples) * 0.1,
    }
    return pd.DataFrame(data)


@pytest.fixture
def trajectory_with_rtk():
    """Create a trajectory DataFrame with RTK data."""
    n_samples = 100
    data = {
        'GPS_Lat': np.linspace(15.366, 15.367, n_samples),
        'GPS_Lng': np.linspace(75.125, 75.126, n_samples),
        'IMU_AccX': np.random.randn(n_samples) * 0.5,
        'IMU_AccY': np.random.randn(n_samples) * 0.5,
        'IMU_AccZ': np.random.randn(n_samples) * 0.5 - 9.8,
        'IMU_GyrX': np.random.randn(n_samples) * 0.1,
        'IMU_GyrY': np.random.randn(n_samples) * 0.1,
        'IMU_GyrZ': np.random.randn(n_samples) * 0.1,
        'RTK_Lat': np.linspace(15.366, 15.367, n_samples) + np.random.randn(n_samples) * 0.00001,
        'RTK_Lng': np.linspace(75.125, 75.126, n_samples) + np.random.randn(n_samples) * 0.00001,
        'GPA_HAcc': np.random.rand(n_samples) * 2.0 + 0.5,
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(valid_trajectory_data):
    """Create a temporary CSV file with valid trajectory data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        valid_trajectory_data.to_csv(f.name, index=False)
        yield f.name
    # Cleanup
    os.unlink(f.name)


class TestLoadTrajectoryCSV:
    """Tests for load_trajectory_csv function."""
    
    def test_load_valid_csv(self, temp_csv_file):
        """Test loading a valid CSV file."""
        df = load_trajectory_csv(temp_csv_file)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check all required columns are present
        for col in REQUIRED_COLUMNS:
            assert col in df.columns
    
    def test_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_trajectory_csv('nonexistent_file.csv')
    
    def test_missing_required_columns(self):
        """Test error handling for missing required columns."""
        # Create CSV with missing columns
        df = pd.DataFrame({
            'GPS_Lat': [15.366],
            'GPS_Lng': [75.125],
            # Missing IMU columns
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            
            with pytest.raises(ValueError, match="Missing required columns"):
                load_trajectory_csv(f.name)
            
            os.unlink(f.name)
    
    def test_filter_nan_values(self, valid_trajectory_data):
        """Test that NaN values are filtered out."""
        # Add some NaN values
        df = valid_trajectory_data.copy()
        df.loc[5:10, 'GPS_Lat'] = np.nan
        df.loc[15:20, 'IMU_AccX'] = np.nan
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            
            loaded_df = load_trajectory_csv(f.name)
            
            # Should have fewer rows after filtering
            assert len(loaded_df) < len(df)
            
            # No NaN values should remain in required columns
            for col in REQUIRED_COLUMNS:
                assert not loaded_df[col].isna().any()
            
            os.unlink(f.name)
    
    def test_filter_zero_coordinates(self, valid_trajectory_data):
        """Test that zero GPS coordinates are filtered out."""
        df = valid_trajectory_data.copy()
        # Set some coordinates to zero (Null Island problem)
        df.loc[5:10, 'GPS_Lat'] = 0
        df.loc[15:20, 'GPS_Lng'] = 0
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            
            loaded_df = load_trajectory_csv(f.name)
            
            # Should have fewer rows after filtering
            assert len(loaded_df) < len(df)
            
            # No zero coordinates should remain
            assert not (loaded_df['GPS_Lat'] == 0).any()
            assert not (loaded_df['GPS_Lng'] == 0).any()
            
            os.unlink(f.name)
    
    def test_empty_after_filtering(self):
        """Test error handling when no valid data remains."""
        # Create CSV with all invalid data
        df = pd.DataFrame({
            'GPS_Lat': [0, 0, 0],
            'GPS_Lng': [0, 0, 0],
            'IMU_AccX': [0, 0, 0],
            'IMU_AccY': [0, 0, 0],
            'IMU_AccZ': [0, 0, 0],
            'IMU_GyrX': [0, 0, 0],
            'IMU_GyrY': [0, 0, 0],
            'IMU_GyrZ': [0, 0, 0],
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            
            with pytest.raises(ValueError, match="No valid data remaining"):
                load_trajectory_csv(f.name)
            
            os.unlink(f.name)
    
    def test_optional_columns_warning(self, valid_trajectory_data):
        """Test that missing optional columns generate warnings."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            valid_trajectory_data.to_csv(f.name, index=False)
            
            with pytest.warns(UserWarning, match="Optional column"):
                load_trajectory_csv(f.name)
            
            os.unlink(f.name)
    
    def test_load_with_optional_columns(self, trajectory_with_rtk):
        """Test loading CSV with optional columns."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            trajectory_with_rtk.to_csv(f.name, index=False)
            
            df = load_trajectory_csv(f.name)
            
            # Check optional columns are included
            assert 'RTK_Lat' in df.columns
            assert 'RTK_Lng' in df.columns
            assert 'GPA_HAcc' in df.columns
            
            os.unlink(f.name)


class TestValidateTrajectoryData:
    """Tests for validate_trajectory_data function."""
    
    def test_valid_data(self, valid_trajectory_data):
        """Test validation of valid trajectory data."""
        is_valid, error_msg = validate_trajectory_data(valid_trajectory_data)
        
        assert is_valid
        assert error_msg == ""
    
    def test_insufficient_data_points(self):
        """Test validation fails with too few data points."""
        df = pd.DataFrame({
            'GPS_Lat': [15.366],
            'GPS_Lng': [75.125],
            'IMU_AccX': [0.1],
            'IMU_AccY': [0.1],
            'IMU_AccZ': [-9.8],
            'IMU_GyrX': [0.01],
            'IMU_GyrY': [0.01],
            'IMU_GyrZ': [0.01],
        })
        
        is_valid, error_msg = validate_trajectory_data(df)
        
        assert not is_valid
        assert "Insufficient data points" in error_msg
    
    def test_invalid_latitude_range(self, valid_trajectory_data):
        """Test validation fails with invalid latitude."""
        df = valid_trajectory_data.copy()
        df.loc[0, 'GPS_Lat'] = 100.0  # Invalid latitude
        
        is_valid, error_msg = validate_trajectory_data(df)
        
        assert not is_valid
        assert "Invalid latitude range" in error_msg
    
    def test_invalid_longitude_range(self, valid_trajectory_data):
        """Test validation fails with invalid longitude."""
        df = valid_trajectory_data.copy()
        df.loc[0, 'GPS_Lng'] = 200.0  # Invalid longitude
        
        is_valid, error_msg = validate_trajectory_data(df)
        
        assert not is_valid
        assert "Invalid longitude range" in error_msg
    
    def test_no_imu_variation(self, valid_trajectory_data):
        """Test validation fails when IMU data has no variation."""
        df = valid_trajectory_data.copy()
        df['IMU_AccX'] = 0.0  # All same value
        
        is_valid, error_msg = validate_trajectory_data(df)
        
        assert not is_valid
        assert "no variation" in error_msg


class TestHasRTKData:
    """Tests for has_rtk_data function."""
    
    def test_has_rtk_data(self, trajectory_with_rtk):
        """Test detection of RTK data."""
        assert has_rtk_data(trajectory_with_rtk)
    
    def test_no_rtk_columns(self, valid_trajectory_data):
        """Test detection when RTK columns are missing."""
        assert not has_rtk_data(valid_trajectory_data)
    
    def test_rtk_all_nan(self, valid_trajectory_data):
        """Test detection when RTK columns are all NaN."""
        df = valid_trajectory_data.copy()
        df['RTK_Lat'] = np.nan
        df['RTK_Lng'] = np.nan
        
        assert not has_rtk_data(df)
    
    def test_rtk_all_zero(self, valid_trajectory_data):
        """Test detection when RTK columns are all zero."""
        df = valid_trajectory_data.copy()
        df['RTK_Lat'] = 0.0
        df['RTK_Lng'] = 0.0
        
        assert not has_rtk_data(df)


class TestExtractGPSTrajectory:
    """Tests for extract_gps_trajectory function."""
    
    def test_extract_gps(self, valid_trajectory_data):
        """Test extraction of GPS trajectory."""
        lat, lon = extract_gps_trajectory(valid_trajectory_data)
        
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert len(lat) == len(valid_trajectory_data)
        assert len(lon) == len(valid_trajectory_data)
        assert lat.dtype == np.float64
        assert lon.dtype == np.float64


class TestExtractRTKTrajectory:
    """Tests for extract_rtk_trajectory function."""
    
    def test_extract_rtk(self, trajectory_with_rtk):
        """Test extraction of RTK trajectory."""
        result = extract_rtk_trajectory(trajectory_with_rtk)
        
        assert result is not None
        lat, lon = result
        assert isinstance(lat, np.ndarray)
        assert isinstance(lon, np.ndarray)
        assert len(lat) > 0
        assert len(lon) > 0
    
    def test_no_rtk_data(self, valid_trajectory_data):
        """Test extraction when no RTK data is available."""
        result = extract_rtk_trajectory(valid_trajectory_data)
        
        assert result is None
    
    def test_rtk_with_nan_values(self, trajectory_with_rtk):
        """Test extraction filters out NaN values."""
        df = trajectory_with_rtk.copy()
        df.loc[5:10, 'RTK_Lat'] = np.nan
        
        result = extract_rtk_trajectory(df)
        
        assert result is not None
        lat, lon = result
        # Should have fewer points after filtering
        assert len(lat) < len(df)
        assert not np.isnan(lat).any()
        assert not np.isnan(lon).any()


class TestExtractIMUData:
    """Tests for extract_imu_data function."""
    
    def test_extract_imu(self, valid_trajectory_data):
        """Test extraction of IMU data."""
        acc, gyr = extract_imu_data(valid_trajectory_data)
        
        assert isinstance(acc, np.ndarray)
        assert isinstance(gyr, np.ndarray)
        assert acc.shape == (len(valid_trajectory_data), 3)
        assert gyr.shape == (len(valid_trajectory_data), 3)
        assert acc.dtype == np.float32
        assert gyr.dtype == np.float32
