"""
Data synchronization processor.

Synchronizes multiple data streams to uniform timestamps with interpolation and resampling.
"""

from .base import BaseSynchronizer
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
from datetime import datetime, timezone
import pytz
from pathlib import Path
from itertools import combinations
from ..utils.coordinates import CoordinateConverter


class DataSynchronizer(BaseSynchronizer):
    """Synchronizes multiple data streams to uniform timestamps."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data synchronizer.
        
        Args:
            config: Configuration dictionary with synchronization parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.target_frequency = self.config.get('target_frequency', 15.0)  # Hz
        self.interpolation_method = self.config.get('interpolation_method', 'linear')
        self.max_gap_seconds = self.config.get('max_gap_seconds', 1.0)
        self.min_data_threshold = self.config.get('min_data_threshold', 0.5)  # 50% minimum data
        
        # Initialize coordinate converter
        self.coordinate_converter = CoordinateConverter(config)
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process single stream by normalizing timestamps and resampling.
        
        Args:
            data: Input DataFrame with timestamp column
            
        Returns:
            Processed DataFrame with uniform timestamps
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        if 'timestamp' not in data.columns:
            raise ValueError("Input data must contain 'timestamp' column")
        
        # Normalize timestamps
        data = self._normalize_timestamps(data)
        
        # Resample to target frequency
        data = self._resample_data(data)
        
        return data
    
    def synchronize_streams(self, data_streams: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Synchronize multiple data streams to uniform timestamps.
        
        Args:
            data_streams: Dictionary of stream name to DataFrame
            
        Returns:
            Synchronized DataFrame with aligned timestamps
        """
        if not data_streams:
            raise ValueError("No data streams provided")
        
        self.logger.info(f"Synchronizing {len(data_streams)} data streams")
        
        # Step 1: Normalize timestamps in all streams
        normalized_streams = {}
        for name, stream in data_streams.items():
            if 'timestamp' not in stream.columns:
                self.logger.warning(f"Stream '{name}' missing timestamp column, skipping")
                continue
            
            try:
                normalized_streams[name] = self._normalize_timestamps(stream.copy())
                self.logger.debug(f"Normalized timestamps for stream '{name}'")
            except Exception as e:
                self.logger.warning(f"Failed to normalize stream '{name}': {str(e)}")
                continue
        
        if not normalized_streams:
            raise ValueError("No valid streams after timestamp normalization")

        if self._should_concatenate_streams(normalized_streams):
            self.logger.info("Detected multiple independent flights - concatenating streams")
            synchronized_df = self._concatenate_streams(normalized_streams)
            return synchronized_df
        
        # Step 2: Find common time range
        time_range = self._find_common_time_range(normalized_streams)
        if time_range is None:
            raise ValueError("No overlapping time range found between streams")
        
        start_time, end_time = time_range
        self.logger.info(f"Common time range: {start_time:.3f} to {end_time:.3f} seconds")
        
        # Step 3: Create uniform time axis
        time_axis = self._create_time_axis(start_time, end_time)
        
        # Step 4: Interpolate and align all streams
        aligned_data = {'timestamp': time_axis}
        
        for name, stream in normalized_streams.items():
            try:
                interpolated = self._interpolate_to_time_axis(stream, time_axis)
                
                # Add stream data with prefixed column names (only if not timestamp)
                for col in interpolated.columns:
                    if col != 'timestamp':
                        # Use stream prefix to avoid conflicts
                        prefixed_col = f"{name}_{col}"
                        aligned_data[prefixed_col] = interpolated[col]
                
                self.logger.debug(f"Interpolated stream '{name}' to common time axis")
                
            except Exception as e:
                self.logger.warning(f"Failed to interpolate stream '{name}': {str(e)}")
                continue
        
        # Step 5: Create synchronized DataFrame
        synchronized_df = pd.DataFrame(aligned_data)

        # Simplify column names when only a single stream is present
        if normalized_streams:
            stream_aliases = {
                name: Path(name).stem if Path(name).stem else str(name)
                for name in normalized_streams
            }
            multiple_streams = len(stream_aliases) > 1
            rename_map = {}

            for col in synchronized_df.columns:
                if col == 'timestamp':
                    continue

                renamed = False
                for name, alias in stream_aliases.items():
                    prefix = f"{name}_"
                    if col.startswith(prefix):
                        suffix = col[len(prefix):]
                        if multiple_streams:
                            rename_map[col] = f"{alias}_{suffix}"
                        else:
                            rename_map[col] = suffix
                        renamed = True
                        break

                if not renamed and '_' in col:
                    rename_map[col] = col.rsplit('_', 1)[-1]

            if rename_map:
                synchronized_df = synchronized_df.rename(columns=rename_map)
        
        # Step 6: Handle missing data and quality checks
        synchronized_df = self._handle_missing_data(synchronized_df)
        
        self.logger.info(f"Synchronized data shape: {synchronized_df.shape}")
        return synchronized_df

    def _should_concatenate_streams(self, streams: Dict[str, pd.DataFrame]) -> bool:
        """Heuristically determine if streams are separate flights that should be concatenated."""
        if len(streams) <= 1:
            return False

        extensions = {Path(name).suffix.lower() for name in streams}
        same_extension = len(extensions) == 1 and '' not in extensions

        if not same_extension:
            return False

        total_pairs = 0
        high_overlap_pairs = 0

        for (name_a, df_a), (name_b, df_b) in combinations(streams.items(), 2):
            cols_a = {col for col in df_a.columns if col != 'timestamp'}
            cols_b = {col for col in df_b.columns if col != 'timestamp'}

            if not cols_a or not cols_b:
                continue

            total_pairs += 1
            overlap = len(cols_a & cols_b)
            min_cols = min(len(cols_a), len(cols_b))

            if min_cols == 0:
                continue

            overlap_ratio = overlap / min_cols

            if overlap_ratio >= 0.8:
                high_overlap_pairs += 1

        return total_pairs > 0 and high_overlap_pairs == total_pairs

    def _concatenate_streams(self, streams: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Concatenate multiple flight streams sequentially."""
        concatenated_frames: List[pd.DataFrame] = []
        dt = 1.0 / self.target_frequency if self.target_frequency > 0 else 0.0

        # Sort streams by their initial timestamp to maintain flight order
        sorted_streams = sorted(
            streams.items(),
            key=lambda item: item[1]['timestamp'].iloc[0] if not item[1].empty else float('inf')
        )

        for index, (name, stream) in enumerate(sorted_streams):
            if stream.empty:
                self.logger.warning(f"Stream '{name}' is empty after normalization, skipping")
                continue

            resampled = self._resample_data(stream)
            if resampled.empty:
                self.logger.warning(f"Stream '{name}' produced no samples after resampling, skipping")
                continue

            resampled = resampled.sort_values('timestamp').reset_index(drop=True)

            if concatenated_frames:
                previous_end = concatenated_frames[-1]['timestamp'].iloc[-1]
                current_start = resampled['timestamp'].iloc[0]

                if previous_end is not None and current_start is not None:
                    if current_start <= previous_end:
                        offset = previous_end + dt - current_start
                        resampled['timestamp'] = resampled['timestamp'] + offset
                    else:
                        # Ensure at least a single dt gap for clarity
                        gap = current_start - previous_end
                        if gap < dt:
                            resampled['timestamp'] = resampled['timestamp'] + (dt - gap)

            concatenated_frames.append(resampled)

        if not concatenated_frames:
            raise ValueError("No data available to concatenate after processing streams")

        result_df = pd.concat(concatenated_frames, ignore_index=True)
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)

        result_df = self._handle_missing_data(result_df)

        self.logger.info(f"Concatenated data shape: {result_df.shape}")
        return result_df
    
    def _normalize_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert timestamps to uniform format (seconds since epoch).
        
        Args:
            data: DataFrame with timestamp column
            
        Returns:
            DataFrame with normalized timestamps
        """
        data = data.copy()
        timestamps = data['timestamp']
        
        # Handle different timestamp formats
        if timestamps.dtype == 'object':
            # Try to parse string timestamps
            try:
                # Try common formats
                parsed_timestamps = pd.to_datetime(timestamps, errors='coerce')
                if parsed_timestamps.isna().all():
                    # Try Unix timestamp strings
                    parsed_timestamps = pd.to_numeric(timestamps, errors='coerce')
                    if not parsed_timestamps.isna().all():
                        # Convert to datetime if they're Unix timestamps
                        parsed_timestamps = pd.to_datetime(parsed_timestamps, unit='s', errors='coerce')
                
                if not parsed_timestamps.isna().all():
                    # Convert to seconds since epoch
                    data['timestamp'] = parsed_timestamps.astype('int64') / 1e9
                else:
                    raise ValueError("Could not parse timestamp strings")
                    
            except Exception as e:
                raise ValueError(f"Failed to parse timestamp column: {str(e)}")
                
        elif pd.api.types.is_datetime64_any_dtype(timestamps):
            # Convert datetime to seconds since epoch
            data['timestamp'] = timestamps.astype('int64') / 1e9
            
        elif pd.api.types.is_numeric_dtype(timestamps):
            # Infer appropriate scaling based on timestamp differences
            sorted_ts = timestamps.sort_values()
            diffs = sorted_ts.diff().dropna().abs()
            median_diff = diffs.median() if not diffs.empty else None
            range_val = sorted_ts.iloc[-1] - sorted_ts.iloc[0] if len(sorted_ts) > 1 else 0.0

            scale_divisor = 1.0

            if median_diff is not None and median_diff > 0:
                if median_diff >= 1e9:
                    scale_divisor = 1e9  # nanoseconds → seconds
                elif median_diff >= 1e6:
                    scale_divisor = 1e6  # microseconds → seconds
                elif median_diff >= 1e3:
                    if range_val >= 1e3:
                        scale_divisor = 1e3  # milliseconds → seconds
            else:
                # Fallback to overall range when consecutive diffs are zero
                if range_val >= 1e9:
                    scale_divisor = 1e9
                elif range_val >= 1e6:
                    scale_divisor = 1e6
                elif range_val >= 1e3:
                    scale_divisor = 1e3
                elif range_val >= 1:
                    scale_divisor = 1e3

            if scale_divisor != 1.0:
                data['timestamp'] = timestamps / scale_divisor
            else:
                data['timestamp'] = timestamps.astype('float64')
            
        else:
            raise ValueError(f"Unsupported timestamp format: {timestamps.dtype}")
        
        # Remove invalid timestamps
        data = data.dropna(subset=['timestamp'])
        
        # Sort by timestamp
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        return data
    
    def _find_common_time_range(self, streams: Dict[str, pd.DataFrame]) -> Optional[Tuple[float, float]]:
        """
        Find the time range for processing - either overlapping or concatenated.
        
        Args:
            streams: Dictionary of normalized data streams
            
        Returns:
            Tuple of (start_time, end_time) or None if no valid range
        """
        if not streams:
            return None
        
        # Collect time ranges for each stream
        stream_ranges = []
        for name, stream in streams.items():
            if stream.empty or 'timestamp' not in stream.columns:
                continue
                
            timestamps = stream['timestamp'].dropna()
            if timestamps.empty:
                continue
                
            stream_ranges.append((timestamps.min(), timestamps.max(), name))
        
        if not stream_ranges:
            return None
        
        # Check if streams have significant overlap or are separate flights
        if len(stream_ranges) > 1:
            # Sort by start time
            stream_ranges.sort(key=lambda x: x[0])
            
            # Check for overlap between consecutive streams
            has_overlap = False
            for i in range(len(stream_ranges) - 1):
                current_end = stream_ranges[i][1]
                next_start = stream_ranges[i + 1][0]
                
                # If there's significant overlap (more than 10% of either stream)
                overlap_duration = max(0, current_end - next_start)
                current_duration = stream_ranges[i][1] - stream_ranges[i][0]
                next_duration = stream_ranges[i + 1][1] - stream_ranges[i + 1][0]
                
                if (overlap_duration > 0.1 * current_duration or 
                    overlap_duration > 0.1 * next_duration):
                    has_overlap = True
                    break
            
            if not has_overlap:
                # Separate flights - use full time range for concatenation
                self.logger.info("Detected separate flights - will concatenate rather than find overlap")
                overall_start = min(r[0] for r in stream_ranges)
                overall_end = max(r[1] for r in stream_ranges)
                return overall_start, overall_end
        
        # Default behavior: find overlapping time range
        start_times = [r[0] for r in stream_ranges]
        end_times = [r[1] for r in stream_ranges]
        
        common_start = max(start_times)
        common_end = min(end_times)
        
        # Ensure we have a valid time range
        if common_end <= common_start:
            # No overlap - fall back to concatenation approach
            self.logger.warning("No overlapping time range found - using concatenation approach")
            overall_start = min(start_times)
            overall_end = max(end_times)
            return overall_start, overall_end
        
        return common_start, common_end
    
    def _create_time_axis(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Create uniform time axis at target frequency.
        
        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            Array of timestamps at target frequency
        """
        dt = 1.0 / self.target_frequency
        num_samples = int((end_time - start_time) / dt) + 1
        
        return np.linspace(start_time, start_time + (num_samples - 1) * dt, num_samples)
    
    def _interpolate_to_time_axis(self, data: pd.DataFrame, time_axis: np.ndarray) -> pd.DataFrame:
        """
        Interpolate data to match the target time axis.
        
        Args:
            data: Input DataFrame with timestamp column
            time_axis: Target time axis
            
        Returns:
            DataFrame interpolated to target time axis
        """
        # Collapse duplicate timestamps to avoid reindex failures
        if data['timestamp'].duplicated().any():
            aggregation = {
                col: ('mean' if pd.api.types.is_numeric_dtype(data[col]) else 'last')
                for col in data.columns
                if col != 'timestamp'
            }
            data = data.groupby('timestamp', as_index=False).agg(aggregation)

        # Ensure numeric data is in floating point to avoid nullable integer issues
        numeric_cols = [
            col for col in data.columns
            if col != 'timestamp' and pd.api.types.is_numeric_dtype(data[col])
        ]
        if numeric_cols:
            data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce').astype('float64')

        # Set timestamp as index for interpolation
        data_indexed = data.set_index('timestamp')
        target_index = pd.Index(time_axis, name='timestamp')
        combined_index = data_indexed.index.union(target_index)

        interpolated_frames = {}

        for col in data_indexed.columns:
            series = data_indexed[col].reindex(combined_index)

            if series.dtype in ['object', 'category']:
                filled = series.ffill().bfill()
                interpolated_frames[col] = filled.reindex(target_index)
            else:
                interpolated = series.interpolate(
                    method=self.interpolation_method,
                    limit_area=None,
                    limit_direction='both'
                )
                interpolated_frames[col] = interpolated.reindex(target_index)

        result_df = pd.DataFrame(interpolated_frames, index=target_index).reset_index()
        
        return result_df
    
    def _resample_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Resample data to target frequency using pandas resampling.
        
        Args:
            data: Input DataFrame with timestamp column
            
        Returns:
            Resampled DataFrame
        """
        if data.empty:
            return data
        
        # Convert timestamp to datetime for resampling
        data = data.copy()
        data['datetime'] = pd.to_datetime(data['timestamp'], unit='s')
        data = data.set_index('datetime')
        
        # Calculate resampling frequency
        freq_str = f"{1000/self.target_frequency:.0f}ms"
        
        # Resample numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns
        
        resampled_data = {}
        
        # Interpolate numeric columns
        if len(numeric_cols) > 0:
            numeric_resampled = data[numeric_cols].resample(freq_str).mean()
            numeric_resampled = numeric_resampled.interpolate(method=self.interpolation_method)
            resampled_data.update(numeric_resampled.to_dict('series'))
        
        # Forward fill categorical columns
        if len(categorical_cols) > 0:
            categorical_resampled = data[categorical_cols].resample(freq_str).ffill()
            resampled_data.update(categorical_resampled.to_dict('series'))
        
        # Create result DataFrame
        result_df = pd.DataFrame(resampled_data)
        result_df['timestamp'] = result_df.index.astype('int64') / 1e9
        result_df = result_df.reset_index(drop=True)
        
        return result_df
    
    def _handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data and perform quality checks.
        
        Args:
            data: Synchronized DataFrame
            
        Returns:
            DataFrame with missing data handled
        """
        if data.empty:
            return data
        
        # Calculate missing data percentage for each column
        missing_percentages = data.isnull().sum() / len(data)
        
        # Log warnings for columns with high missing data
        for col, missing_pct in missing_percentages.items():
            if missing_pct > (1 - self.min_data_threshold):
                self.logger.warning(
                    f"Column '{col}' has {missing_pct:.1%} missing data"
                )
        
        # Remove rows where critical columns are all missing
        critical_cols = [col for col in data.columns if 'gps' in col.lower() or 'imu' in col.lower()]
        if critical_cols:
            # Keep rows that have at least some critical data
            data = data.dropna(subset=critical_cols, how='all')
        
        return data
    
    def detect_data_gaps(self, data: pd.DataFrame, max_gap_seconds: Optional[float] = None) -> List[Tuple[float, float]]:
        """
        Detect gaps in timestamp data that exceed the maximum allowed gap.
        
        Args:
            data: DataFrame with timestamp column
            max_gap_seconds: Maximum allowed gap in seconds (uses config default if None)
            
        Returns:
            List of (start_time, end_time) tuples representing gaps
        """
        if max_gap_seconds is None:
            max_gap_seconds = self.max_gap_seconds
        
        if data.empty or 'timestamp' not in data.columns:
            return []
        
        timestamps = data['timestamp'].dropna().sort_values()
        if len(timestamps) < 2:
            return []
        
        # Calculate time differences
        time_diffs = timestamps.diff().dropna()
        
        # Find gaps larger than threshold
        gap_indices = time_diffs[time_diffs > max_gap_seconds].index
        
        gaps = []
        for idx in gap_indices:
            gap_start = timestamps.loc[idx - 1]
            gap_end = timestamps.loc[idx]
            gaps.append((gap_start, gap_end))
        
        return gaps
    
    def interpolate_gaps(self, data: pd.DataFrame, max_gap_seconds: Optional[float] = None) -> pd.DataFrame:
        """
        Interpolate small gaps in data while preserving larger gaps as NaN.
        
        Args:
            data: DataFrame with timestamp column
            max_gap_seconds: Maximum gap size to interpolate (larger gaps left as NaN)
            
        Returns:
            DataFrame with small gaps interpolated
        """
        if max_gap_seconds is None:
            max_gap_seconds = self.max_gap_seconds
        
        if data.empty:
            return data
        
        data = data.copy()
        
        # Detect gaps
        gaps = self.detect_data_gaps(data, max_gap_seconds)
        
        if gaps:
            self.logger.info(f"Found {len(gaps)} data gaps larger than {max_gap_seconds}s")
        
        # Set timestamp as index for interpolation
        data_indexed = data.set_index('timestamp')
        
        # Interpolate numeric columns
        numeric_cols = data_indexed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Interpolate the entire series
            interpolated = data_indexed[col].interpolate(
                method=self.interpolation_method, 
                limit_area='inside'
            )
            
            # Set large gaps back to NaN
            for gap_start, gap_end in gaps:
                gap_mask = (data_indexed.index > gap_start) & (data_indexed.index < gap_end)
                interpolated.loc[gap_mask] = np.nan
            
            data_indexed[col] = interpolated
        
        return data_indexed.reset_index()
    
    def resample_with_quality_check(self, data: pd.DataFrame, 
                                   target_freq: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Resample data with quality metrics.
        
        Args:
            data: Input DataFrame
            target_freq: Target frequency in Hz (uses config default if None)
            
        Returns:
            Tuple of (resampled DataFrame, quality metrics dict)
        """
        if target_freq is None:
            target_freq = self.target_frequency
        
        if data.empty:
            return data, {}
        
        original_length = len(data)
        original_duration = data['timestamp'].max() - data['timestamp'].min()
        
        # Resample the data
        resampled_data = self._resample_data(data)
        
        # Calculate quality metrics
        quality_metrics = {
            'original_samples': original_length,
            'resampled_samples': len(resampled_data),
            'original_duration_seconds': original_duration,
            'target_frequency_hz': target_freq,
            'actual_frequency_hz': len(resampled_data) / original_duration if original_duration > 0 else 0,
            'data_retention_ratio': len(resampled_data.dropna()) / len(resampled_data) if len(resampled_data) > 0 else 0
        }
        
        return resampled_data, quality_metrics
    
    def convert_coordinates_to_enu(self, data: pd.DataFrame, 
                                  home_point: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """
        Convert GPS coordinates to ENU coordinate system.
        
        Args:
            data: DataFrame with GPS coordinates
            home_point: Optional home point (lat, lon, alt). If None, calculated from data.
            
        Returns:
            DataFrame with ENU coordinates added
        """
        if data.empty:
            return data
        
        # Check for GPS columns (including prefixed ones from synchronization)
        gps_cols = ['gps_lat', 'gps_lon', 'gps_alt']
        available_gps_cols = []
        
        # Look for direct GPS columns
        for col in gps_cols:
            if col in data.columns:
                available_gps_cols.append(col)
        
        # Look for prefixed GPS columns (from stream synchronization)
        if not available_gps_cols:
            for col in data.columns:
                if any(gps_col in col for gps_col in gps_cols):
                    available_gps_cols.append(col)
        
        if not available_gps_cols:
            self.logger.warning("No GPS columns found for coordinate conversion")
            return data
        
        # If we have prefixed columns, we need to create a temporary DataFrame with standard names
        if not any(col in data.columns for col in gps_cols):
            # Find the GPS columns and create standard mapping
            temp_data = data.copy()
            gps_mapping = {}
            
            for standard_col in gps_cols:
                matching_cols = [col for col in data.columns if standard_col in col]
                if matching_cols:
                    # Use the first matching column
                    temp_data[standard_col] = data[matching_cols[0]]
                    gps_mapping[standard_col] = matching_cols[0]
            
            # Only proceed if we have all required GPS columns
            if len(gps_mapping) == len(gps_cols):
                try:
                    # Convert using temporary data
                    converted_temp = self.coordinate_converter.convert_dataframe_to_enu(temp_data, home_point)
                    
                    # Add ENU columns back to original data
                    if 'enu_x' in converted_temp.columns:
                        data['enu_x'] = converted_temp['enu_x']
                        data['enu_y'] = converted_temp['enu_y'] 
                        data['enu_z'] = converted_temp['enu_z']
                        if hasattr(converted_temp, 'attrs') and 'home_point' in converted_temp.attrs:
                            data.attrs['home_point'] = converted_temp.attrs['home_point']
                        
                        self.logger.info("Successfully converted GPS coordinates to ENU")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert coordinates: {str(e)}")
            else:
                self.logger.warning(f"Incomplete GPS data for conversion. Found: {list(gps_mapping.keys())}")
            
            return data
        
        try:
            # Convert coordinates using the coordinate converter
            converted_data = self.coordinate_converter.convert_dataframe_to_enu(
                data, home_point
            )
            
            self.logger.info("Successfully converted GPS coordinates to ENU")
            return converted_data
            
        except Exception as e:
            self.logger.error(f"Failed to convert coordinates to ENU: {str(e)}")
            return data
    
    def synchronize_with_coordinate_conversion(self, data_streams: Dict[str, pd.DataFrame],
                                             home_point: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """
        Synchronize data streams and convert coordinates to ENU in one step.
        
        Args:
            data_streams: Dictionary of stream name to DataFrame
            home_point: Optional home point for coordinate conversion
            
        Returns:
            Synchronized DataFrame with ENU coordinates
        """
        # First synchronize the streams
        synchronized_data = self.synchronize_streams(data_streams)
        
        # Then convert coordinates to ENU
        if not synchronized_data.empty:
            synchronized_data = self.convert_coordinates_to_enu(synchronized_data, home_point)
        
        return synchronized_data