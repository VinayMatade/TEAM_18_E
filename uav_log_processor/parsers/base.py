"""
Base class for UAV log parsers.

Defines the common interface that all log format parsers must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
from pathlib import Path


class BaseLogParser(ABC):
    """Abstract base class for UAV log parsers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parser with optional configuration.
        
        Args:
            config: Optional configuration dictionary for parser settings
        """
        self.config = config or {}
        self._supported_extensions = set()
    
    @property
    def supported_extensions(self) -> set:
        """Return set of supported file extensions."""
        return self._supported_extensions
    
    @abstractmethod
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parse a log file and return structured data.
        
        Args:
            file_path: Path to the log file to parse
            
        Returns:
            DataFrame with columns: timestamp, gps_lat, gps_lon, gps_alt,
            imu_ax, imu_ay, imu_az, imu_gx, imu_gy, imu_gz, velocity_x,
            velocity_y, velocity_z, hdop, vdop, fix_type
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is invalid or unsupported
        """
        pass
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate that the file exists and has a supported extension.
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            True if file is valid, False otherwise
        """
        path = Path(file_path)
        return (path.exists() and 
                path.is_file() and 
                path.suffix.lower() in self._supported_extensions)
    
    def _extract_timestamp(self, raw_timestamp: Any) -> float:
        """
        Convert raw timestamp to standardized format (seconds since epoch).
        
        Args:
            raw_timestamp: Raw timestamp from log file
            
        Returns:
            Timestamp in seconds since epoch
        """
        # Default implementation - subclasses should override if needed
        if isinstance(raw_timestamp, (int, float)):
            return float(raw_timestamp)
        raise ValueError(f"Unsupported timestamp format: {type(raw_timestamp)}")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure DataFrame has standardized column names and types.
        
        Args:
            df: Raw parsed DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        required_columns = [
            'timestamp', 'gps_lat', 'gps_lon', 'gps_alt',
            'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz',
            'velocity_x', 'velocity_y', 'velocity_z', 'hdop', 'vdop', 'fix_type'
        ]
        
        # Add missing columns with NaN values
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Ensure correct column order
        df = df[required_columns]
        
        # Convert timestamp to float
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df