"""
Motion classification processor.

Classifies motion segments as stationary or moving based on IMU data.
"""

from .base import BaseClassifier
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging


class MotionClassifier(BaseClassifier):
    """Classifies motion segments as stationary or moving."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the motion classifier.
        
        Args:
            config: Configuration dictionary with classification parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.accel_threshold = self.config.get('accel_threshold', 0.5)  # m/s²
        self.gyro_threshold = self.config.get('gyro_threshold', 0.1)    # rad/s
        self.window_size_seconds = self.config.get('window_size_seconds', 5.0)  # seconds
        self.min_segment_duration = self.config.get('min_segment_duration', 3.0)  # seconds
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data by adding motion classification labels.
        
        Args:
            data: Input DataFrame with IMU data
            
        Returns:
            DataFrame with motion classification added
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data")
        
        # Classify motion
        motion_labels = self.classify(data)
        
        # Add labels to data
        result = data.copy()
        result['motion_label'] = motion_labels
        
        return result
    
    def classify(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify motion segments as stationary or moving.
        
        Args:
            data: Input DataFrame with IMU data
            
        Returns:
            Series with motion labels ('stationary' or 'moving')
        """
        if data.empty:
            return pd.Series(dtype=str)
        
        # Calculate motion magnitudes
        accel_magnitude = self.calculate_acceleration_magnitude(data)
        gyro_magnitude = self.calculate_gyroscope_magnitude(data)
        
        # Apply sliding window smoothing
        accel_smoothed = self.apply_sliding_window_smoothing(accel_magnitude, data)
        gyro_smoothed = self.apply_sliding_window_smoothing(gyro_magnitude, data)
        
        # Classify based on thresholds
        motion_labels = self._classify_motion_segments(accel_smoothed, gyro_smoothed)
        
        # Filter short segments
        motion_labels = self._filter_short_segments(motion_labels, data)
        
        return motion_labels
    
    def calculate_acceleration_magnitude(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate acceleration magnitude from IMU data.
        
        Args:
            data: DataFrame with IMU acceleration columns
            
        Returns:
            Series with acceleration magnitudes
        """
        # Find acceleration columns (handle prefixed columns from synchronization)
        accel_cols = self._find_imu_columns(data, ['ax', 'ay', 'az'])
        
        if len(accel_cols) < 3:
            self.logger.warning(f"Found only {len(accel_cols)} acceleration columns: {accel_cols}")
            # Return zeros if insufficient data
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        # Extract acceleration components
        ax = data[accel_cols[0]]
        ay = data[accel_cols[1]] 
        az = data[accel_cols[2]]
        
        # Calculate magnitude of horizontal acceleration components first
        # This better represents motion-induced acceleration
        horizontal_magnitude = np.sqrt(ax**2 + ay**2)
        
        # For vertical component, remove gravity
        vertical_motion = np.abs(az - 9.81)
        
        # Combine horizontal and vertical motion
        magnitude = np.sqrt(horizontal_magnitude**2 + vertical_motion**2)
        
        return magnitude
    
    def calculate_gyroscope_magnitude(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate gyroscope magnitude from IMU data.
        
        Args:
            data: DataFrame with IMU gyroscope columns
            
        Returns:
            Series with gyroscope magnitudes
        """
        # Find gyroscope columns (handle prefixed columns from synchronization)
        gyro_cols = self._find_imu_columns(data, ['gx', 'gy', 'gz'])
        
        if len(gyro_cols) < 3:
            self.logger.warning(f"Found only {len(gyro_cols)} gyroscope columns: {gyro_cols}")
            # Return zeros if insufficient data
            return pd.Series(np.zeros(len(data)), index=data.index)
        
        # Extract gyroscope components
        gx = data[gyro_cols[0]]
        gy = data[gyro_cols[1]]
        gz = data[gyro_cols[2]]
        
        # Calculate magnitude: sqrt(gx² + gy² + gz²)
        magnitude = np.sqrt(gx**2 + gy**2 + gz**2)
        
        return magnitude
    
    def apply_sliding_window_smoothing(self, signal: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Apply sliding window smoothing for noise reduction.
        
        Args:
            signal: Input signal to smooth
            data: DataFrame with timestamp information
            
        Returns:
            Smoothed signal
        """
        if signal.empty or data.empty:
            return signal
        
        # Calculate window size in samples
        if 'timestamp' in data.columns:
            # Calculate sampling rate
            time_diffs = data['timestamp'].diff().dropna()
            if not time_diffs.empty:
                median_dt = time_diffs.median()
                sampling_rate = 1.0 / median_dt if median_dt > 0 else 15.0  # Default 15 Hz
            else:
                sampling_rate = 15.0
        else:
            sampling_rate = 15.0  # Default assumption
        
        window_samples = int(self.window_size_seconds * sampling_rate)
        window_samples = max(1, min(window_samples, len(signal)))  # Ensure valid window size
        
        # Apply rolling mean for smoothing
        smoothed = signal.rolling(window=window_samples, center=True, min_periods=1).mean()
        
        return smoothed
    
    def get_stationary_segments(self, motion_labels: pd.Series, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Get list of stationary segment indices.
        
        Args:
            motion_labels: Series with motion classification labels
            data: DataFrame with timestamp information
            
        Returns:
            List of (start_index, end_index) tuples for stationary segments
        """
        if motion_labels.empty:
            return []
        
        segments = []
        current_start = None
        
        for i, label in enumerate(motion_labels):
            if label == 'stationary':
                if current_start is None:
                    current_start = i
            else:  # moving or other
                if current_start is not None:
                    segments.append((current_start, i - 1))
                    current_start = None
        
        # Handle case where data ends with stationary segment
        if current_start is not None:
            segments.append((current_start, len(motion_labels) - 1))
        
        # Filter segments by minimum duration
        if 'timestamp' in data.columns:
            diffs = data['timestamp'].diff().dropna()
            sample_interval = float(diffs.median()) if not diffs.empty else None
            tolerance = 0.0
            if self.min_segment_duration > 0:
                tolerance = 0.1 * self.min_segment_duration
            if sample_interval and sample_interval > 0:
                tolerance = max(tolerance, 2 * sample_interval)

            filtered_segments = []
            for start_idx, end_idx in segments:
                if start_idx < len(data) and end_idx < len(data):
                    duration = data.iloc[end_idx]['timestamp'] - data.iloc[start_idx]['timestamp']
                    if sample_interval and sample_interval > 0:
                        duration += sample_interval
                    if duration + tolerance + 1e-9 >= self.min_segment_duration:
                        filtered_segments.append((start_idx, end_idx))
            return filtered_segments
        
        return segments
    
    def _find_imu_columns(self, data: pd.DataFrame, suffixes: List[str]) -> List[str]:
        """
        Find IMU columns with given suffixes, handling prefixed column names.
        
        Args:
            data: DataFrame to search
            suffixes: List of column suffixes to find (e.g., ['ax', 'ay', 'az'])
            
        Returns:
            List of matching column names
        """
        found_cols = []
        
        for suffix in suffixes:
            # Look for exact match first
            if f'imu_{suffix}' in data.columns:
                found_cols.append(f'imu_{suffix}')
                continue
            
            # Look for prefixed columns (from stream synchronization)
            matching_cols = [col for col in data.columns if col.endswith(f'imu_{suffix}')]
            if matching_cols:
                found_cols.append(matching_cols[0])  # Use first match
                continue
            
            # Look for any column containing the suffix
            matching_cols = [col for col in data.columns if suffix in col and 'imu' in col]
            if matching_cols:
                found_cols.append(matching_cols[0])
                continue
        
        return found_cols
    
    def _classify_motion_segments(self, accel_magnitude: pd.Series, gyro_magnitude: pd.Series) -> pd.Series:
        """
        Classify motion based on acceleration and gyroscope thresholds.
        
        Args:
            accel_magnitude: Smoothed acceleration magnitude
            gyro_magnitude: Smoothed gyroscope magnitude
            
        Returns:
            Series with motion labels
        """
        # Initialize all as moving
        labels = pd.Series(['moving'] * len(accel_magnitude), index=accel_magnitude.index)
        
        # Mark as stationary where both thresholds are met
        stationary_mask = (accel_magnitude <= self.accel_threshold) & (gyro_magnitude <= self.gyro_threshold)
        labels[stationary_mask] = 'stationary'
        
        return labels
    
    def _filter_short_segments(self, motion_labels: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Filter out short motion segments to reduce noise.
        
        Args:
            motion_labels: Initial motion labels
            data: DataFrame with timestamp information
            
        Returns:
            Filtered motion labels
        """
        if motion_labels.empty or 'timestamp' not in data.columns:
            return motion_labels
        
        filtered_labels = motion_labels.copy()
        
        # Find segment boundaries
        label_changes = motion_labels != motion_labels.shift(1)
        segment_starts = label_changes[label_changes].index.tolist()
        
        if not segment_starts:
            return filtered_labels
        
        # Add end index
        segment_starts.append(len(motion_labels))
        
        # Check each segment duration
        for i in range(len(segment_starts) - 1):
            start_idx = segment_starts[i]
            end_idx = segment_starts[i + 1] - 1
            
            if start_idx < len(data) and end_idx < len(data):
                duration = data.iloc[end_idx]['timestamp'] - data.iloc[start_idx]['timestamp']
                
                # If segment is too short, change it to match neighboring segments
                if duration < self.min_segment_duration:
                    # Determine what to change it to based on neighbors
                    if i > 0:
                        # Use previous segment's label
                        prev_start = segment_starts[i - 1]
                        new_label = motion_labels.iloc[prev_start]
                    elif i < len(segment_starts) - 2:
                        # Use next segment's label
                        next_start = segment_starts[i + 1]
                        new_label = motion_labels.iloc[next_start] if next_start < len(motion_labels) else 'moving'
                    else:
                        new_label = 'moving'  # Default
                    
                    filtered_labels.iloc[start_idx:end_idx + 1] = new_label
        
        return filtered_labels
    
    def get_motion_segments(self, motion_labels: pd.Series, data: pd.DataFrame) -> List[Dict]:
        """
        Get detailed information about all motion segments.
        
        Args:
            motion_labels: Series with motion classification labels
            data: DataFrame with timestamp information
            
        Returns:
            List of dictionaries with segment information
        """
        if motion_labels.empty:
            return []
        
        segments = []
        current_label = None
        current_start = None
        
        for i, label in enumerate(motion_labels):
            if label != current_label:
                # End previous segment
                if current_start is not None:
                    segment_info = self._create_segment_info(
                        current_label, current_start, i - 1, data
                    )
                    segments.append(segment_info)
                
                # Start new segment
                current_label = label
                current_start = i
        
        # Handle final segment
        if current_start is not None:
            segment_info = self._create_segment_info(
                current_label, current_start, len(motion_labels) - 1, data
            )
            segments.append(segment_info)
        
        return segments
    
    def detect_transitions(self, motion_labels: pd.Series, data: pd.DataFrame) -> List[Dict]:
        """
        Detect transitions between stationary and moving segments.
        
        Args:
            motion_labels: Series with motion classification labels
            data: DataFrame with timestamp information
            
        Returns:
            List of transition information dictionaries
        """
        if motion_labels.empty:
            return []
        
        transitions = []
        prev_label = None
        
        for i, label in enumerate(motion_labels):
            if prev_label is not None and label != prev_label:
                transition_info = {
                    'index': i,
                    'from_state': prev_label,
                    'to_state': label,
                    'timestamp': data.iloc[i]['timestamp'] if 'timestamp' in data.columns and i < len(data) else None
                }
                transitions.append(transition_info)
            
            prev_label = label
        
        return transitions
    
    def apply_transition_filtering(self, motion_labels: pd.Series, data: pd.DataFrame) -> pd.Series:
        """
        Apply additional filtering to smooth transitions between segments.
        
        Args:
            motion_labels: Initial motion labels
            data: DataFrame with timestamp information
            
        Returns:
            Filtered motion labels with smoother transitions
        """
        if motion_labels.empty:
            return motion_labels
        
        filtered_labels = motion_labels.copy()
        
        # Apply median filter to reduce noise
        window_size = min(5, len(motion_labels))  # Small window for transition smoothing
        
        # Convert labels to numeric for filtering
        label_numeric = (motion_labels == 'stationary').astype(int)
        
        # Apply rolling median
        smoothed_numeric = label_numeric.rolling(
            window=window_size, center=True, min_periods=1
        ).median()
        
        # Convert back to labels
        filtered_labels = (smoothed_numeric >= 0.5).map({True: 'stationary', False: 'moving'})
        
        return filtered_labels
    
    def get_classification_statistics(self, motion_labels: pd.Series, data: pd.DataFrame) -> Dict:
        """
        Calculate statistics about the motion classification.
        
        Args:
            motion_labels: Series with motion classification labels
            data: DataFrame with timestamp information
            
        Returns:
            Dictionary with classification statistics
        """
        if motion_labels.empty:
            return {}
        
        stats = {
            'total_samples': len(motion_labels),
            'stationary_samples': (motion_labels == 'stationary').sum(),
            'moving_samples': (motion_labels == 'moving').sum(),
            'stationary_percentage': (motion_labels == 'stationary').mean() * 100,
            'moving_percentage': (motion_labels == 'moving').mean() * 100
        }
        
        if 'timestamp' in data.columns and len(data) > 0:
            total_duration = data['timestamp'].max() - data['timestamp'].min()
            stats['total_duration_seconds'] = total_duration
            
            # Calculate duration for each state
            stationary_mask = motion_labels == 'stationary'
            if stationary_mask.any():
                # Estimate stationary duration (approximate)
                stats['stationary_duration_seconds'] = total_duration * stats['stationary_percentage'] / 100
            else:
                stats['stationary_duration_seconds'] = 0.0
            
            stats['moving_duration_seconds'] = total_duration - stats['stationary_duration_seconds']
        
        # Get segment information
        segments = self.get_motion_segments(motion_labels, data)
        stationary_segments = [s for s in segments if s['label'] == 'stationary']
        moving_segments = [s for s in segments if s['label'] == 'moving']
        
        stats['num_stationary_segments'] = len(stationary_segments)
        stats['num_moving_segments'] = len(moving_segments)
        stats['total_segments'] = len(segments)
        
        if stationary_segments:
            durations = [s['duration_seconds'] for s in stationary_segments if s['duration_seconds'] is not None]
            if durations:
                stats['avg_stationary_segment_duration'] = np.mean(durations)
                stats['max_stationary_segment_duration'] = np.max(durations)
                stats['min_stationary_segment_duration'] = np.min(durations)
        
        if moving_segments:
            durations = [s['duration_seconds'] for s in moving_segments if s['duration_seconds'] is not None]
            if durations:
                stats['avg_moving_segment_duration'] = np.mean(durations)
                stats['max_moving_segment_duration'] = np.max(durations)
                stats['min_moving_segment_duration'] = np.min(durations)
        
        return stats
    
    def _create_segment_info(self, label: str, start_idx: int, end_idx: int, data: pd.DataFrame) -> Dict:
        """
        Create detailed information dictionary for a segment.
        
        Args:
            label: Segment label ('stationary' or 'moving')
            start_idx: Start index of segment
            end_idx: End index of segment
            data: DataFrame with timestamp information
            
        Returns:
            Dictionary with segment information
        """
        segment_info = {
            'label': label,
            'start_index': start_idx,
            'end_index': end_idx,
            'length_samples': end_idx - start_idx + 1
        }
        
        if 'timestamp' in data.columns and start_idx < len(data) and end_idx < len(data):
            start_time = data.iloc[start_idx]['timestamp']
            end_time = data.iloc[end_idx]['timestamp']
            segment_info.update({
                'start_timestamp': start_time,
                'end_timestamp': end_time,
                'duration_seconds': end_time - start_time
            })
        else:
            segment_info.update({
                'start_timestamp': None,
                'end_timestamp': None,
                'duration_seconds': None
            })
        
        return segment_info
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data has required IMU columns.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not super().validate_input(data):
            return False
        
        # Check for IMU columns
        accel_cols = self._find_imu_columns(data, ['ax', 'ay', 'az'])
        gyro_cols = self._find_imu_columns(data, ['gx', 'gy', 'gz'])
        
        if len(accel_cols) < 3 or len(gyro_cols) < 3:
            self.logger.warning("Insufficient IMU columns for motion classification")
            return False
        
        return True