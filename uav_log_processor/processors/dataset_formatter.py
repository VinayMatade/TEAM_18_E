"""
Dataset formatting processor for UAV log data.

Handles feature standardization, normalization, and dataset splitting
for machine learning consumption.
"""

from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from .base import BaseFormatter


logger = logging.getLogger(__name__)


class DatasetFormatter(BaseFormatter):
    """
    Formats synchronized UAV data for machine learning training.
    
    Handles feature standardization, normalization, dataset splitting,
    and CSV output generation.
    """
    
    # Standard column order for consistent output
    STANDARD_COLUMNS = [
        'timestamp',
        'gps_x', 'gps_y', 'gps_z',
        'imu_ax', 'imu_ay', 'imu_az',
        'imu_gx', 'imu_gy', 'imu_gz',
        'velocity_x', 'velocity_y', 'velocity_z',
        'hdop', 'vdop', 'fix_type',
        'motion_label',  # Add motion_label to preserve it during standardization
        'ground_truth_x', 'ground_truth_y', 'ground_truth_z',
        'gps_error_x', 'gps_error_y', 'gps_error_z', 'gps_error_norm'
    ]
    
    # Features that should be normalized (continuous numerical features)
    NORMALIZATION_FEATURES = [
        'gps_x', 'gps_y', 'gps_z',
        'imu_ax', 'imu_ay', 'imu_az',
        'imu_gx', 'imu_gy', 'imu_gz',
        'velocity_x', 'velocity_y', 'velocity_z',
        'hdop', 'vdop',
        'ground_truth_x', 'ground_truth_y', 'ground_truth_z',
        'gps_error_x', 'gps_error_y', 'gps_error_z', 'gps_error_norm'
    ]
    
    # Categorical features that need encoding
    CATEGORICAL_FEATURES = ['fix_type']
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize dataset formatter.
        
        Args:
            config: Configuration dictionary with formatting parameters
        """
        super().__init__(config)
        self.normalization_stats = {}
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data through complete formatting pipeline.
        
        Args:
            data: Synchronized UAV data DataFrame
            
        Returns:
            Formatted DataFrame ready for ML training
        """
        # Standardize features
        formatted_data = self.standardize_features(data)
        
        # Normalize continuous features
        normalized_data, _ = self.normalize_features(formatted_data)
        
        return normalized_data
    
    def standardize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names and ordering, combine features into aligned dataset.
        
        Args:
            data: Input DataFrame with various column names
            
        Returns:
            DataFrame with standardized column names and order
        """
        logger.info("Standardizing feature columns and ordering")
        
        # Create a copy to avoid modifying original data
        standardized_data = data.copy()
        
        # Map common column name variations to standard names
        column_mapping = self._create_column_mapping(data.columns)
        
        # Handle multiple streams by combining duplicate mappings
        standardized_data = self._combine_multiple_streams(standardized_data, column_mapping)
        
        # Handle missing features with graceful degradation
        standardized_data = self._handle_missing_features(standardized_data)
        
        # Reorder columns to standard order (only include columns that exist)
        available_standard_columns = [col for col in self.STANDARD_COLUMNS if col in standardized_data.columns]
        
        # Also preserve any additional columns that aren't in the standard list (for intermediate processing)
        additional_columns = [col for col in standardized_data.columns if col not in self.STANDARD_COLUMNS]
        
        # Combine standard columns first, then additional columns
        final_columns = available_standard_columns + additional_columns
        standardized_data = standardized_data[final_columns]
        
        logger.info(f"Standardized dataset with {len(final_columns)} features")
        return standardized_data
    
    def _create_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """
        Create mapping from input column names to standard names.
        
        Args:
            columns: List of input column names
            
        Returns:
            Dictionary mapping input names to standard names
        """
        mapping = {}
        
        # Common variations for GPS coordinates
        gps_mappings = {
            'lat': 'gps_y', 'latitude': 'gps_y', 'gps_lat': 'gps_y',
            'lon': 'gps_x', 'longitude': 'gps_x', 'gps_lon': 'gps_x',
            'alt': 'gps_z', 'altitude': 'gps_z', 'gps_alt': 'gps_z',
            'gps_latitude': 'gps_y', 'gps_longitude': 'gps_x', 'gps_altitude': 'gps_z'
        }
        
        # Common variations for IMU data
        imu_mappings = {
            'accel_x': 'imu_ax', 'acc_x': 'imu_ax', 'ax': 'imu_ax', 'imu_ax': 'imu_ax',
            'accel_y': 'imu_ay', 'acc_y': 'imu_ay', 'ay': 'imu_ay', 'imu_ay': 'imu_ay',
            'accel_z': 'imu_az', 'acc_z': 'imu_az', 'az': 'imu_az', 'imu_az': 'imu_az',
            'gyro_x': 'imu_gx', 'gyr_x': 'imu_gx', 'gx': 'imu_gx', 'imu_gx': 'imu_gx',
            'gyro_y': 'imu_gy', 'gyr_y': 'imu_gy', 'gy': 'imu_gy', 'imu_gy': 'imu_gy',
            'gyro_z': 'imu_gz', 'gyr_z': 'imu_gz', 'gz': 'imu_gz', 'imu_gz': 'imu_gz'
        }
        
        # Common variations for velocity
        velocity_mappings = {
            'vel_x': 'velocity_x', 'vx': 'velocity_x', 'velocity_x': 'velocity_x',
            'vel_y': 'velocity_y', 'vy': 'velocity_y', 'velocity_y': 'velocity_y',
            'vel_z': 'velocity_z', 'vz': 'velocity_z', 'velocity_z': 'velocity_z'
        }
        
        # Common variations for GPS quality
        quality_mappings = {
            'horizontal_dop': 'hdop', 'h_dop': 'hdop', 'hdop': 'hdop',
            'vertical_dop': 'vdop', 'v_dop': 'vdop', 'vdop': 'vdop',
            'gps_fix_type': 'fix_type', 'fix': 'fix_type', 'fix_type': 'fix_type'
        }
        
        # Combine all mappings
        all_mappings = {**gps_mappings, **imu_mappings, **velocity_mappings, **quality_mappings}
        
        # Find matches in input columns
        for col in columns:
            # Handle prefixed columns from synchronizer (e.g., "/tmp/file.txt_gps_lat")
            if '_' in col:
                # Extract the suffix after the last underscore
                suffix = col.split('_')[-1]
                # Try to match the suffix
                if suffix.lower() in all_mappings:
                    mapping[col] = all_mappings[suffix.lower()]
                    continue
                
                # Try to match the last two parts (e.g., "gps_lat")
                if len(col.split('_')) >= 2:
                    last_two = '_'.join(col.split('_')[-2:])
                    if last_two.lower() in all_mappings:
                        mapping[col] = all_mappings[last_two.lower()]
                        continue
            
            # Try exact match
            col_lower = col.lower()
            if col_lower in all_mappings:
                mapping[col] = all_mappings[col_lower]
        
        return mapping
    
    def _combine_multiple_streams(self, data: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Combine multiple streams that map to the same standard columns.
        
        Args:
            data: DataFrame with potentially duplicate column mappings
            column_mapping: Mapping from original to standard column names
            
        Returns:
            DataFrame with combined streams
        """
        if not column_mapping:
            return data
        
        result_data = data.copy()
        
        # Group columns by their target standard name
        standard_to_original = {}
        for original_col, standard_col in column_mapping.items():
            if standard_col not in standard_to_original:
                standard_to_original[standard_col] = []
            standard_to_original[standard_col].append(original_col)
        
        # For each standard column that has multiple sources, combine them
        for standard_col, original_cols in standard_to_original.items():
            if len(original_cols) > 1:
                logger.info(f"Combining {len(original_cols)} streams for {standard_col}")
                
                # Combine multiple columns by taking the first non-null value
                combined_series = None
                for col in original_cols:
                    if col in result_data.columns:
                        if combined_series is None:
                            combined_series = result_data[col].copy()
                        else:
                            # Fill nulls in combined series with values from current column
                            mask = combined_series.isna() & result_data[col].notna()
                            combined_series[mask] = result_data[col][mask]
                
                # Add the combined column and remove originals
                if combined_series is not None:
                    result_data[standard_col] = combined_series
                    for col in original_cols:
                        if col in result_data.columns:
                            result_data = result_data.drop(columns=[col])
            else:
                # Single column mapping - just rename
                original_col = original_cols[0]
                if original_col in result_data.columns:
                    result_data = result_data.rename(columns={original_col: standard_col})
        
        return result_data
    
    def _handle_missing_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing features with graceful degradation.
        
        Args:
            data: DataFrame potentially missing some standard features
            
        Returns:
            DataFrame with missing features handled appropriately
        """
        result_data = data.copy()
        
        # Check for missing critical features
        missing_features = [col for col in self.STANDARD_COLUMNS if col not in data.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # For this implementation, we'll just work with available features
            # and not add default values unless specifically needed
        
        return result_data
    
    def format_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Format complete dataset with standardization and normalization.
        
        Args:
            data: Input synchronized DataFrame
            
        Returns:
            Tuple of (formatted DataFrame, metadata dictionary)
        """
        # Standardize features
        standardized_data = self.standardize_features(data)
        
        # Normalize features
        normalized_data, norm_stats = self.normalize_features(standardized_data)
        
        # Create metadata
        metadata = {
            'total_samples': len(normalized_data),
            'features': list(normalized_data.columns),
            'normalization_stats': norm_stats,
            'categorical_features': [f for f in self.CATEGORICAL_FEATURES if f in normalized_data.columns],
            'continuous_features': [f for f in self.NORMALIZATION_FEATURES if f in normalized_data.columns]
        }
        
        return normalized_data, metadata
    
    def normalize_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply Z-score normalization to continuous features.
        
        Args:
            data: DataFrame with standardized features
            
        Returns:
            Tuple of (normalized DataFrame, normalization statistics)
        """
        logger.info("Applying Z-score normalization to continuous features")
        
        normalized_data = data.copy()
        normalization_stats = {}
        
        # Normalize continuous features
        for feature in self.NORMALIZATION_FEATURES:
            if feature in data.columns:
                # Calculate mean and standard deviation
                mean_val = data[feature].mean()
                std_val = data[feature].std()
                
                # Avoid division by zero for constant features
                if std_val == 0:
                    logger.warning(f"Feature {feature} has zero standard deviation, skipping normalization")
                    normalization_stats[feature] = {'mean': mean_val, 'std': 1.0, 'normalized': False}
                    continue
                
                # Apply Z-score normalization: (x - mean) / std
                normalized_data[feature] = (data[feature] - mean_val) / std_val
                
                # Store statistics for later denormalization if needed
                normalization_stats[feature] = {
                    'mean': mean_val,
                    'std': std_val,
                    'normalized': True
                }
                
                logger.debug(f"Normalized {feature}: mean={mean_val:.4f}, std={std_val:.4f}")
        
        # Handle categorical features (encode if needed)
        normalized_data = self._encode_categorical_features(normalized_data, normalization_stats)
        
        # Store normalization stats for later use
        self.normalization_stats = normalization_stats
        
        normalized_count = len([f for f in normalization_stats if normalization_stats[f].get('normalized', False)])
        logger.info(f"Normalized {normalized_count} continuous features")
        return normalized_data, normalization_stats
    
    def _encode_categorical_features(self, data: pd.DataFrame, stats: Dict[str, Any]) -> pd.DataFrame:
        """
        Encode categorical features appropriately.
        
        Args:
            data: DataFrame with features to encode
            stats: Dictionary to store encoding statistics
            
        Returns:
            DataFrame with encoded categorical features
        """
        encoded_data = data.copy()
        
        for feature in self.CATEGORICAL_FEATURES:
            if feature in data.columns:
                if feature == 'fix_type':
                    # GPS fix_type is ordinal (0=no fix, 1=dead reckoning, 2=2D, 3=3D, etc.)
                    # Keep as integer but ensure valid range
                    filled_feature = data[feature].fillna(0).infer_objects(copy=False)
                    encoded_data[feature] = filled_feature.astype(int)
                    
                    # Store encoding info
                    unique_values = sorted(pd.unique(filled_feature))
                    stats[feature] = {
                        'type': 'ordinal',
                        'unique_values': unique_values,
                        'encoded': True
                    }
                    
                    logger.debug(f"Encoded categorical feature {feature} with values: {unique_values}")
        
        return encoded_data
    
    def denormalize_features(self, data: pd.DataFrame, 
                           normalization_stats: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Reverse Z-score normalization using stored or provided statistics.
        
        Args:
            data: Normalized DataFrame
            normalization_stats: Optional normalization statistics, uses stored if None
            
        Returns:
            DataFrame with original scale features
        """
        if normalization_stats is None:
            normalization_stats = self.normalization_stats
        
        if not normalization_stats:
            logger.warning("No normalization statistics available for denormalization")
            return data.copy()
        
        denormalized_data = data.copy()
        
        for feature, stats in normalization_stats.items():
            if feature in data.columns and stats.get('normalized', False):
                # Reverse Z-score: x_original = (x_normalized * std) + mean
                denormalized_data[feature] = (data[feature] * stats['std']) + stats['mean']
        
        return denormalized_data
    
    def split_dataset(self, data: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset sequentially into train/validation/test sets.
        
        Ensures no temporal overlap between splits and handles multiple flights.
        
        Args:
            data: Input DataFrame to split (must be sorted by timestamp)
            train_ratio: Fraction of data for training (default 0.7)
            val_ratio: Fraction of data for validation (default 0.15)
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting dataset with ratios: train={train_ratio}, val={val_ratio}, test={1-train_ratio-val_ratio}")
        
        # Validate ratios
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio <= 0:
            raise ValueError(f"Invalid split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # Ensure data is sorted by timestamp for sequential splitting
        if 'timestamp' in data.columns:
            sorted_data = data.sort_values('timestamp').reset_index(drop=True)
        else:
            logger.warning("No timestamp column found, using data order as-is")
            sorted_data = data.reset_index(drop=True)
        
        # Check if we have multiple flights (large time gaps)
        flight_segments = self._identify_flight_segments(sorted_data)
        
        if len(flight_segments) > 1:
            logger.info(f"Detected {len(flight_segments)} flight segments, splitting each proportionally")
            return self._split_multiple_flights(sorted_data, flight_segments, train_ratio, val_ratio)
        else:
            logger.info("Single flight detected, performing sequential split")
            return self._split_single_flight(sorted_data, train_ratio, val_ratio)
    
    def _identify_flight_segments(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        Identify separate flight segments based on timestamp gaps.
        
        Args:
            data: Sorted DataFrame with timestamp column
            
        Returns:
            List of (start_idx, end_idx) tuples for each flight segment
        """
        if 'timestamp' not in data.columns or len(data) < 2:
            return [(0, len(data) - 1)]
        
        segments = []
        segment_start = 0
        
        # Look for gaps larger than 10 minutes (600 seconds) as flight boundaries
        gap_threshold = 600.0
        
        for i in range(1, len(data)):
            time_gap = data.iloc[i]['timestamp'] - data.iloc[i-1]['timestamp']
            
            if time_gap > gap_threshold:
                # End current segment
                segments.append((segment_start, i - 1))
                segment_start = i
        
        # Add final segment
        segments.append((segment_start, len(data) - 1))
        
        # Filter out very small segments (less than 100 samples)
        segments = [(start, end) for start, end in segments if end - start >= 100]
        
        logger.info(f"Identified {len(segments)} flight segments")
        return segments
    
    def _split_single_flight(self, data: pd.DataFrame, 
                           train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split single flight data sequentially.
        
        Args:
            data: Sorted DataFrame
            train_ratio: Training data fraction
            val_ratio: Validation data fraction
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n_samples = len(data)
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Ensure we have at least some samples in each split
        train_end = max(1, min(train_end, n_samples - 2))
        val_end = max(train_end + 1, min(val_end, n_samples - 1))
        
        # Split data
        train_df = data.iloc[:train_end].copy()
        val_df = data.iloc[train_end:val_end].copy()
        test_df = data.iloc[val_end:].copy()
        
        logger.info(f"Single flight split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df
    
    def _split_multiple_flights(self, data: pd.DataFrame, 
                              flight_segments: List[Tuple[int, int]],
                              train_ratio: float, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split multiple flight data ensuring no temporal overlap.
        
        Args:
            data: Sorted DataFrame
            flight_segments: List of (start_idx, end_idx) for each flight
            train_ratio: Training data fraction
            val_ratio: Validation data fraction
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for start_idx, end_idx in flight_segments:
            # Extract flight data
            flight_data = data.iloc[start_idx:end_idx + 1].copy()
            
            # Split this flight proportionally
            flight_train, flight_val, flight_test = self._split_single_flight(
                flight_data, train_ratio, val_ratio
            )
            
            train_dfs.append(flight_train)
            val_dfs.append(flight_val)
            test_dfs.append(flight_test)
        
        # Combine all flights
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        
        logger.info(f"Multiple flights split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df
    
    def validate_split_integrity(self, train_df: pd.DataFrame, 
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> bool:
        """
        Validate that dataset splits have no temporal overlap.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset  
            test_df: Test dataset
            
        Returns:
            True if splits are valid, False otherwise
        """
        if 'timestamp' not in train_df.columns:
            logger.warning("Cannot validate temporal integrity without timestamp column")
            return True
        
        # Get timestamp ranges for each split
        train_times = (train_df['timestamp'].min(), train_df['timestamp'].max())
        val_times = (val_df['timestamp'].min(), val_df['timestamp'].max())
        test_times = (test_df['timestamp'].min(), test_df['timestamp'].max())
        
        # Check for overlaps
        overlaps = []
        
        # Train-Val overlap
        if not (train_times[1] < val_times[0] or val_times[1] < train_times[0]):
            overlaps.append("train-validation")
        
        # Train-Test overlap
        if not (train_times[1] < test_times[0] or test_times[1] < train_times[0]):
            overlaps.append("train-test")
        
        # Val-Test overlap
        if not (val_times[1] < test_times[0] or test_times[1] < val_times[0]):
            overlaps.append("validation-test")
        
        if overlaps:
            logger.error(f"Temporal overlaps detected: {overlaps}")
            return False
        
        logger.info("Dataset splits validated - no temporal overlaps")
        return True
    
    def save_datasets(self, train_df: pd.DataFrame, 
                     val_df: pd.DataFrame, 
                     test_df: pd.DataFrame,
                     output_dir: str = "output",
                     chunk_size: int = 10000) -> Dict[str, str]:
        """
        Save train/validation/test datasets to CSV files.
        
        Handles large datasets with chunked writing for memory efficiency.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            output_dir: Directory to save CSV files
            chunk_size: Number of rows per chunk for large datasets
            
        Returns:
            Dictionary with paths to saved files
        """
        logger.info(f"Saving datasets to {output_dir}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Define output file paths
        file_paths = {
            'train': output_path / 'train.csv',
            'valid': output_path / 'valid.csv',
            'test': output_path / 'test.csv'
        }
        
        # Save each dataset
        datasets = {
            'train': train_df,
            'valid': val_df,
            'test': test_df
        }
        
        saved_paths = {}
        
        for split_name, df in datasets.items():
            file_path = file_paths[split_name]
            
            try:
                if len(df) > chunk_size:
                    # Use chunked writing for large datasets
                    self._save_large_csv(df, file_path, chunk_size)
                else:
                    # Direct save for smaller datasets
                    df.to_csv(file_path, index=False, float_format='%.6f')
                
                saved_paths[split_name] = str(file_path)
                logger.info(f"Saved {split_name} dataset: {len(df)} samples to {file_path}")
                
            except Exception as e:
                logger.error(f"Failed to save {split_name} dataset: {e}")
                raise
        
        # Validate saved files
        self._validate_saved_files(saved_paths)
        
        return saved_paths
    
    def _save_large_csv(self, df: pd.DataFrame, file_path: Path, chunk_size: int):
        """
        Save large DataFrame to CSV using chunked writing.
        
        Args:
            df: DataFrame to save
            file_path: Output file path
            chunk_size: Number of rows per chunk
        """
        logger.info(f"Saving large dataset ({len(df)} rows) in chunks of {chunk_size}")
        
        # Write header first
        header_written = False
        
        for start_idx in range(0, len(df), chunk_size):
            end_idx = min(start_idx + chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]
            
            # Write chunk
            chunk.to_csv(
                file_path, 
                mode='w' if not header_written else 'a',
                header=not header_written,
                index=False,
                float_format='%.6f'
            )
            
            header_written = True
            
            if start_idx % (chunk_size * 10) == 0:  # Log progress every 10 chunks
                logger.debug(f"Saved {end_idx}/{len(df)} rows")
    
    def _validate_saved_files(self, file_paths: Dict[str, str]):
        """
        Validate that saved CSV files are readable and have expected format.
        
        Args:
            file_paths: Dictionary of split names to file paths
        """
        for split_name, file_path in file_paths.items():
            try:
                # Try to read a few rows to validate format
                test_df = pd.read_csv(file_path, nrows=5)
                
                if test_df.empty:
                    raise ValueError(f"Saved file {file_path} is empty")
                
                logger.debug(f"Validated {split_name} file: {file_path}")
                
            except Exception as e:
                logger.error(f"Validation failed for {split_name} file {file_path}: {e}")
                raise
    
    def save_full_dataset(self, data: pd.DataFrame, 
                         output_dir: str = "output",
                         filename: str = "aligned_full.csv") -> str:
        """
        Save complete aligned dataset for reproducibility.
        
        Args:
            data: Complete synchronized dataset
            output_dir: Output directory
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        try:
            # Save with high precision for reproducibility
            data.to_csv(file_path, index=False, float_format='%.8f')
            logger.info(f"Saved full dataset: {len(data)} samples to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save full dataset: {e}")
            raise
    
    def generate_dataset_summary(self, train_df: pd.DataFrame,
                               val_df: pd.DataFrame, 
                               test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate summary statistics for the dataset splits.
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            
        Returns:
            Dictionary with dataset summary statistics
        """
        total_samples = len(train_df) + len(val_df) + len(test_df)
        
        summary = {
            'total_samples': total_samples,
            'splits': {
                'train': {
                    'samples': len(train_df),
                    'percentage': len(train_df) / total_samples * 100
                },
                'validation': {
                    'samples': len(val_df),
                    'percentage': len(val_df) / total_samples * 100
                },
                'test': {
                    'samples': len(test_df),
                    'percentage': len(test_df) / total_samples * 100
                }
            },
            'features': list(train_df.columns),
            'feature_count': len(train_df.columns)
        }
        
        # Add timestamp ranges if available
        if 'timestamp' in train_df.columns:
            summary['temporal_ranges'] = {
                'train': {
                    'start': float(train_df['timestamp'].min()),
                    'end': float(train_df['timestamp'].max()),
                    'duration': float(train_df['timestamp'].max() - train_df['timestamp'].min())
                },
                'validation': {
                    'start': float(val_df['timestamp'].min()),
                    'end': float(val_df['timestamp'].max()),
                    'duration': float(val_df['timestamp'].max() - val_df['timestamp'].min())
                },
                'test': {
                    'start': float(test_df['timestamp'].min()),
                    'end': float(test_df['timestamp'].max()),
                    'duration': float(test_df['timestamp'].max() - test_df['timestamp'].min())
                }
            }
        
        return summary