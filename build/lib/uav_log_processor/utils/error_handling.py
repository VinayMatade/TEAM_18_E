"""
Robust error handling utilities for UAV log processing.

Provides error handling, recovery mechanisms, and memory-efficient processing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union, Iterator
from pathlib import Path
import logging
import traceback
import psutil
import gc
from contextlib import contextmanager
import warnings

logger = logging.getLogger(__name__)


class ProcessingError(Exception):
    """Base exception for processing errors."""
    pass


class CorruptedFileError(ProcessingError):
    """Exception raised when a file is corrupted or unreadable."""
    pass


class MemoryError(ProcessingError):
    """Exception raised when memory usage exceeds limits."""
    pass


class CoordinateConversionError(ProcessingError):
    """Exception raised during coordinate conversion failures."""
    pass


class RobustErrorHandler:
    """Handles errors gracefully during UAV log processing."""
    
    def __init__(self, max_memory_gb: float = 8.0, enable_recovery: bool = True):
        """
        Initialize error handler.
        
        Args:
            max_memory_gb: Maximum memory usage in GB before triggering memory management
            enable_recovery: Whether to attempt recovery from errors
        """
        self.max_memory_gb = max_memory_gb
        self.enable_recovery = enable_recovery
        self.error_log = []
    
    @contextmanager
    def handle_processing_errors(self, operation_name: str, critical: bool = False):
        """
        Context manager for handling processing errors.
        
        Args:
            operation_name: Name of the operation being performed
            critical: Whether the operation is critical (raises on failure)
        """
        try:
            logger.debug(f"Starting operation: {operation_name}")
            yield
            logger.debug(f"Completed operation: {operation_name}")
        except Exception as e:
            error_info = {
                'operation': operation_name,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            self.error_log.append(error_info)
            
            logger.error(f"Error in {operation_name}: {e}")
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            if critical:
                raise ProcessingError(f"Critical error in {operation_name}: {e}") from e
            else:
                logger.warning(f"Non-critical error in {operation_name}, continuing...")
    
    def check_memory_usage(self) -> float:
        """
        Check current memory usage.
        
        Returns:
            Current memory usage in GB
        """
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        
        if memory_gb > self.max_memory_gb:
            logger.warning(f"High memory usage detected: {memory_gb:.2f} GB")
            gc.collect()  # Force garbage collection
            
            # Check again after garbage collection
            memory_gb = process.memory_info().rss / (1024**3)
            if memory_gb > self.max_memory_gb:
                raise MemoryError(f"Memory usage ({memory_gb:.2f} GB) exceeds limit ({self.max_memory_gb} GB)")
        
        return memory_gb
    
    def safe_file_read(self, file_path: str, parser_func: Callable, **kwargs) -> Optional[pd.DataFrame]:
        """
        Safely read a file with error handling and recovery.
        
        Args:
            file_path: Path to the file to read
            parser_func: Function to parse the file
            **kwargs: Additional arguments for the parser function
            
        Returns:
            Parsed DataFrame or None if reading failed
        """
        path = Path(file_path)
        
        # Check if file exists and is readable
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if not path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return None
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024**2)
        logger.debug(f"Reading file {file_path} ({file_size_mb:.1f} MB)")
        
        try:
            # Attempt to read the file
            with self.handle_processing_errors(f"reading {file_path}", critical=False):
                df = parser_func(file_path, **kwargs)
                
                if df is None or df.empty:
                    logger.warning(f"File {file_path} produced empty data")
                    return None
                
                logger.info(f"Successfully read {len(df)} records from {file_path}")
                return df
                
        except Exception as e:
            if self.enable_recovery:
                logger.warning(f"Attempting recovery for corrupted file: {file_path}")
                return self._attempt_file_recovery(file_path, parser_func, **kwargs)
            else:
                logger.error(f"Failed to read file {file_path}: {e}")
                return None
    
    def _attempt_file_recovery(self, file_path: str, parser_func: Callable, **kwargs) -> Optional[pd.DataFrame]:
        """
        Attempt to recover data from a corrupted file.
        
        Args:
            file_path: Path to the corrupted file
            parser_func: Function to parse the file
            **kwargs: Additional arguments for the parser function
            
        Returns:
            Partially recovered DataFrame or None
        """
        logger.info(f"Attempting file recovery for {file_path}")
        
        # Try reading with different error handling strategies
        recovery_strategies = [
            {'on_bad_lines': 'skip'},
            {'on_bad_lines': 'warn'},
            {'encoding': 'latin-1'},
            {'encoding': 'utf-8', 'errors': 'ignore'}
        ]
        
        for strategy in recovery_strategies:
            try:
                logger.debug(f"Trying recovery strategy: {strategy}")
                merged_kwargs = {**kwargs, **strategy}
                df = parser_func(file_path, **merged_kwargs)
                
                if df is not None and not df.empty:
                    logger.info(f"Partial recovery successful: {len(df)} records recovered")
                    return df
                    
            except Exception as e:
                logger.debug(f"Recovery strategy failed: {e}")
                continue
        
        logger.error(f"All recovery strategies failed for {file_path}")
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors encountered.
        
        Returns:
            Dictionary with error statistics and details
        """
        if not self.error_log:
            return {'total_errors': 0, 'error_types': {}, 'operations': {}}
        
        error_types = {}
        operations = {}
        
        for error in self.error_log:
            error_type = error['error_type']
            operation = error['operation']
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            operations[operation] = operations.get(operation, 0) + 1
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'operations': operations,
            'recent_errors': self.error_log[-5:]  # Last 5 errors
        }


class ChunkedProcessor:
    """Process large datasets in memory-efficient chunks."""
    
    def __init__(self, chunk_size: Optional[int] = None, max_memory_gb: float = 4.0):
        """
        Initialize chunked processor.
        
        Args:
            chunk_size: Number of rows per chunk (None for auto-sizing)
            max_memory_gb: Maximum memory to use for processing
        """
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.error_handler = RobustErrorHandler(max_memory_gb)
    
    def process_in_chunks(self, data: pd.DataFrame, 
                         process_func: Callable[[pd.DataFrame], pd.DataFrame],
                         **kwargs) -> pd.DataFrame:
        """
        Process a large DataFrame in chunks.
        
        Args:
            data: Input DataFrame to process
            process_func: Function to apply to each chunk
            **kwargs: Additional arguments for process_func
            
        Returns:
            Processed DataFrame
        """
        if data.empty:
            return data
        
        # Determine chunk size if not specified
        if self.chunk_size is None:
            chunk_size = self._estimate_chunk_size(data)
        else:
            chunk_size = self.chunk_size
        
        logger.info(f"Processing {len(data)} rows in chunks of {chunk_size}")
        
        processed_chunks = []
        
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i:i + chunk_size]
            
            with self.error_handler.handle_processing_errors(f"chunk {i//chunk_size + 1}", critical=False):
                # Check memory before processing
                self.error_handler.check_memory_usage()
                
                # Process chunk
                processed_chunk = process_func(chunk, **kwargs)
                
                if processed_chunk is not None and not processed_chunk.empty:
                    processed_chunks.append(processed_chunk)
                
                # Force garbage collection after each chunk
                gc.collect()
        
        if not processed_chunks:
            logger.warning("No chunks were successfully processed")
            return pd.DataFrame()
        
        # Combine processed chunks
        logger.info(f"Combining {len(processed_chunks)} processed chunks")
        result = pd.concat(processed_chunks, ignore_index=True)
        
        return result
    
    def _estimate_chunk_size(self, data: pd.DataFrame) -> int:
        """
        Estimate optimal chunk size based on available memory and data size.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Estimated chunk size
        """
        # Estimate memory usage per row
        memory_per_row = data.memory_usage(deep=True).sum() / len(data)
        
        # Target memory usage per chunk (use 1/4 of max memory)
        target_memory_bytes = self.max_memory_gb * (1024**3) / 4
        
        # Calculate chunk size
        chunk_size = int(target_memory_bytes / memory_per_row)
        
        # Ensure reasonable bounds
        chunk_size = max(1000, min(chunk_size, 100000))
        
        logger.debug(f"Estimated chunk size: {chunk_size} rows "
                    f"(~{memory_per_row * chunk_size / (1024**2):.1f} MB per chunk)")
        
        return chunk_size


class CoordinateConverter:
    """Robust coordinate conversion with error handling."""
    
    def __init__(self):
        """Initialize coordinate converter."""
        self.error_handler = RobustErrorHandler()
    
    def safe_coordinate_conversion(self, df: pd.DataFrame, 
                                 home_point: tuple,
                                 source_crs: str = "EPSG:4326",
                                 target_crs: str = "ENU") -> pd.DataFrame:
        """
        Safely convert coordinates with error handling.
        
        Args:
            df: DataFrame with lat/lon coordinates
            home_point: (lat, lon, alt) tuple for ENU origin
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
            
        Returns:
            DataFrame with converted coordinates
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        try:
            # Import pyproj here to handle import errors gracefully
            from pyproj import Transformer, CRS
            
            # Validate home point
            if not self._validate_home_point(home_point):
                raise CoordinateConversionError(f"Invalid home point: {home_point}")
            
            # Set up coordinate transformation
            if target_crs == "ENU":
                # Create local ENU coordinate system
                enu_crs = CRS.from_proj4(
                    f"+proj=tmerc +lat_0={home_point[0]} +lon_0={home_point[1]} "
                    f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
                )
                transformer = Transformer.from_crs(source_crs, enu_crs, always_xy=True)
            else:
                transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
            
            # Convert coordinates in chunks to handle large datasets
            chunk_size = 10000
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size]
                
                with self.error_handler.handle_processing_errors(f"coordinate conversion chunk {i//chunk_size + 1}", critical=False):
                    # Extract coordinates
                    lons = chunk['gps_lon'].values
                    lats = chunk['gps_lat'].values
                    
                    # Handle NaN values
                    valid_mask = ~(np.isnan(lons) | np.isnan(lats))
                    
                    if not valid_mask.any():
                        logger.warning(f"No valid coordinates in chunk {i//chunk_size + 1}")
                        continue
                    
                    # Convert valid coordinates
                    x_coords = np.full_like(lons, np.nan)
                    y_coords = np.full_like(lats, np.nan)
                    
                    if valid_mask.any():
                        x_valid, y_valid = transformer.transform(
                            lons[valid_mask], lats[valid_mask]
                        )
                        x_coords[valid_mask] = x_valid
                        y_coords[valid_mask] = y_valid
                    
                    # Update result DataFrame
                    result_df.loc[chunk.index, 'gps_x'] = x_coords
                    result_df.loc[chunk.index, 'gps_y'] = y_coords
                    
                    # Handle altitude conversion
                    if 'gps_alt' in chunk.columns:
                        z_coords = chunk['gps_alt'].values - home_point[2]  # Relative to home altitude
                        result_df.loc[chunk.index, 'gps_z'] = z_coords
            
            logger.info(f"Successfully converted coordinates for {len(result_df)} points")
            
        except ImportError:
            logger.error("pyproj not available for coordinate conversion")
            raise CoordinateConversionError("pyproj library not available")
        
        except Exception as e:
            logger.error(f"Coordinate conversion failed: {e}")
            raise CoordinateConversionError(f"Coordinate conversion failed: {e}") from e
        
        return result_df
    
    def _validate_home_point(self, home_point: tuple) -> bool:
        """
        Validate home point coordinates.
        
        Args:
            home_point: (lat, lon, alt) tuple
            
        Returns:
            True if valid, False otherwise
        """
        if len(home_point) != 3:
            return False
        
        lat, lon, alt = home_point
        
        # Check for valid latitude (-90 to 90)
        if not (-90 <= lat <= 90):
            return False
        
        # Check for valid longitude (-180 to 180)
        if not (-180 <= lon <= 180):
            return False
        
        # Check for reasonable altitude (-1000 to 10000 meters)
        if not (-1000 <= alt <= 10000):
            return False
        
        return True


def safe_operation(func: Callable, *args, default_return=None, **kwargs):
    """
    Execute an operation safely with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if operation fails
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if operation fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"Safe operation failed: {e}")
        return default_return


def validate_dataframe_integrity(df: pd.DataFrame, required_columns: List[str] = None) -> Dict[str, Any]:
    """
    Validate DataFrame integrity and return quality metrics.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Dictionary with validation results
    """
    if df is None:
        return {
            'valid': False,
            'error': 'DataFrame is None',
            'row_count': 0,
            'column_count': 0,
            'missing_columns': required_columns or [],
            'null_percentages': {}
        }
    
    if df.empty:
        return {
            'valid': False,
            'error': 'DataFrame is empty',
            'row_count': 0,
            'column_count': len(df.columns),
            'missing_columns': required_columns or [],
            'null_percentages': {}
        }
    
    # Check for required columns
    missing_columns = []
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
    
    # Calculate null percentages
    null_percentages = {}
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percentages[col] = null_count / len(df) * 100
    
    # Determine if DataFrame is valid
    is_valid = len(missing_columns) == 0
    
    return {
        'valid': is_valid,
        'error': f'Missing columns: {missing_columns}' if missing_columns else None,
        'row_count': len(df),
        'column_count': len(df.columns),
        'missing_columns': missing_columns,
        'null_percentages': null_percentages,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2)
    }