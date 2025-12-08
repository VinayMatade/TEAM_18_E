"""Trajectory loading and preprocessing for visualization.

This module provides functions to load and preprocess trajectory data from CSV files
for visualization and analysis. It handles:
- Loading CSV files with GPS and IMU data
- Validating required columns
- Handling optional columns (RTK ground truth, horizontal accuracy)
- Filtering invalid data (NaN values, zero coordinates)
- Extracting specific data types (GPS, RTK, IMU)

Example usage:
    >>> from uav_log_processor.trajectory_loader import load_trajectory_csv
    >>> df = load_trajectory_csv('path/to/trajectory.csv')
    >>> lat, lon = extract_gps_trajectory(df)
    >>> if has_rtk_data(df):
    ...     rtk_lat, rtk_lon = extract_rtk_trajectory(df)
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import warnings


# Required columns for trajectory processing
REQUIRED_COLUMNS = [
    'GPS_Lat',
    'GPS_Lng',
    'IMU_AccX',
    'IMU_AccY',
    'IMU_AccZ',
    'IMU_GyrX',
    'IMU_GyrY',
    'IMU_GyrZ'
]

# Optional columns that may be present
OPTIONAL_COLUMNS = [
    'GPA_HAcc',  # Horizontal accuracy
    'RTK_Lat',   # RTK ground truth latitude
    'RTK_Lng'    # RTK ground truth longitude
]


def load_trajectory_csv(csv_path: str) -> pd.DataFrame:
    """Load and preprocess trajectory data from a CSV file.
    
    This function:
    1. Loads the CSV file
    2. Validates that required columns exist
    3. Handles missing optional columns
    4. Filters invalid GPS coordinates (NaN, zeros)
    5. Returns a cleaned DataFrame
    
    Args:
        csv_path: Path to the CSV file containing trajectory data
        
    Returns:
        DataFrame with cleaned trajectory data
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing or no valid data remains
    """
    # Load the CSV file
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}")
    
    # Validate required columns exist
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing_cols}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )
    
    # Check for optional columns and warn if missing
    available_optional = []
    for col in OPTIONAL_COLUMNS:
        if col in df.columns:
            available_optional.append(col)
        else:
            warnings.warn(
                f"Optional column '{col}' not found in {csv_path}",
                UserWarning
            )
    
    # Select columns to keep (required + available optional)
    columns_to_keep = REQUIRED_COLUMNS + available_optional
    df = df[columns_to_keep].copy()
    
    # Store original length for reporting
    original_length = len(df)
    
    # Filter out rows with NaN values in required columns
    df = df.dropna(subset=REQUIRED_COLUMNS)
    nan_removed = original_length - len(df)
    
    if nan_removed > 0:
        warnings.warn(
            f"Removed {nan_removed} rows with NaN values in required columns",
            UserWarning
        )
    
    # Filter out invalid GPS coordinates (zeros - "Null Island" problem)
    # GPS coordinates of (0, 0) are invalid for most real-world applications
    valid_gps = (df['GPS_Lat'] != 0) & (df['GPS_Lng'] != 0)
    df = df[valid_gps].copy()
    zero_removed = len(df[~valid_gps])
    
    if zero_removed > 0:
        warnings.warn(
            f"Removed {zero_removed} rows with zero GPS coordinates",
            UserWarning
        )
    
    # Check if we have any data left
    if len(df) == 0:
        raise ValueError(
            f"No valid data remaining after filtering in {csv_path}. "
            f"Original rows: {original_length}, "
            f"NaN removed: {nan_removed}, "
            f"Zero coordinates removed: {zero_removed}"
        )
    
    # Reset index after filtering
    df = df.reset_index(drop=True)
    
    # Report final data statistics
    final_length = len(df)
    if original_length > final_length:
        retention_pct = (final_length / original_length) * 100
        print(
            f"Loaded {csv_path}: "
            f"{final_length}/{original_length} rows retained ({retention_pct:.1f}%)"
        )
    else:
        print(f"Loaded {csv_path}: {final_length} rows")
    
    return df


def validate_trajectory_data(df: pd.DataFrame) -> Tuple[bool, str]:
    """Validate that trajectory data meets minimum requirements.
    
    Args:
        df: DataFrame containing trajectory data
        
    Returns:
        Tuple of (is_valid, error_message)
        If is_valid is True, error_message will be empty
    """
    # Check for minimum number of data points
    min_points = 10
    if len(df) < min_points:
        return False, f"Insufficient data points: {len(df)} < {min_points}"
    
    # Check GPS coordinate ranges (rough sanity check)
    lat_min, lat_max = df['GPS_Lat'].min(), df['GPS_Lat'].max()
    lon_min, lon_max = df['GPS_Lng'].min(), df['GPS_Lng'].max()
    
    if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
        return False, f"Invalid latitude range: [{lat_min}, {lat_max}]"
    
    if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
        return False, f"Invalid longitude range: [{lon_min}, {lon_max}]"
    
    # Check for reasonable IMU values (not all zeros)
    imu_cols = ['IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']
    for col in imu_cols:
        if df[col].std() < 1e-6:
            return False, f"IMU column '{col}' has no variation (all same value)"
    
    return True, ""


def has_rtk_data(df: pd.DataFrame) -> bool:
    """Check if the DataFrame contains RTK ground truth data.
    
    Args:
        df: DataFrame containing trajectory data
        
    Returns:
        True if RTK data is available and valid, False otherwise
    """
    if 'RTK_Lat' not in df.columns or 'RTK_Lng' not in df.columns:
        return False
    
    # Check if RTK columns have valid data (not all NaN or zero)
    rtk_lat_valid = df['RTK_Lat'].notna().sum() > 0
    rtk_lng_valid = df['RTK_Lng'].notna().sum() > 0
    
    if not (rtk_lat_valid and rtk_lng_valid):
        return False
    
    # Check if RTK data is not all zeros
    rtk_lat_nonzero = (df['RTK_Lat'] != 0).sum() > 0
    rtk_lng_nonzero = (df['RTK_Lng'] != 0).sum() > 0
    
    return rtk_lat_nonzero and rtk_lng_nonzero


def extract_gps_trajectory(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract GPS latitude and longitude as numpy arrays.
    
    Args:
        df: DataFrame containing trajectory data
        
    Returns:
        Tuple of (latitude_array, longitude_array)
    """
    lat = df['GPS_Lat'].values.astype(np.float64)
    lon = df['GPS_Lng'].values.astype(np.float64)
    return lat, lon


def extract_rtk_trajectory(df: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract RTK ground truth trajectory if available.
    
    Args:
        df: DataFrame containing trajectory data
        
    Returns:
        Tuple of (latitude_array, longitude_array) if RTK data is available,
        None otherwise
    """
    if not has_rtk_data(df):
        return None
    
    # Extract RTK data and filter out NaN and zero values
    rtk_df = df[['RTK_Lat', 'RTK_Lng']].copy()
    rtk_df = rtk_df.dropna()
    rtk_df = rtk_df[(rtk_df['RTK_Lat'] != 0) & (rtk_df['RTK_Lng'] != 0)]
    
    if len(rtk_df) == 0:
        return None
    
    lat = rtk_df['RTK_Lat'].values.astype(np.float64)
    lon = rtk_df['RTK_Lng'].values.astype(np.float64)
    return lat, lon


def extract_imu_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Extract IMU accelerometer and gyroscope data.
    
    Args:
        df: DataFrame containing trajectory data
        
    Returns:
        Tuple of (accelerometer_array, gyroscope_array)
        Each array has shape (n_samples, 3) for X, Y, Z axes
    """
    acc_cols = ['IMU_AccX', 'IMU_AccY', 'IMU_AccZ']
    gyr_cols = ['IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']
    
    acc = df[acc_cols].values.astype(np.float32)
    gyr = df[gyr_cols].values.astype(np.float32)
    
    return acc, gyr
