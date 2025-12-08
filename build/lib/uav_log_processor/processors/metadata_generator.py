"""
Metadata generation processor for UAV log data.

Generates comprehensive metadata.json with feature descriptions,
source information, sampling rates, and processing parameters.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import logging

from .base import BaseProcessor


logger = logging.getLogger(__name__)


class MetadataGenerator(BaseProcessor):
    """
    Generates comprehensive metadata for UAV datasets.
    
    Creates metadata.json with feature descriptions, source information,
    sampling rates, normalization statistics, and processing parameters.
    """
    
    # Categorical features that need special handling
    CATEGORICAL_FEATURES = ['fix_type']
    
    # Feature descriptions for documentation
    FEATURE_DESCRIPTIONS = {
        'timestamp': {
            'description': 'Unix timestamp in seconds',
            'unit': 'seconds',
            'type': 'continuous',
            'source': 'synchronized'
        },
        'gps_x': {
            'description': 'GPS position in East direction (ENU coordinates)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'gps'
        },
        'gps_y': {
            'description': 'GPS position in North direction (ENU coordinates)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'gps'
        },
        'gps_z': {
            'description': 'GPS position in Up direction (ENU coordinates)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'gps'
        },
        'imu_ax': {
            'description': 'IMU acceleration in X axis',
            'unit': 'm/s²',
            'type': 'continuous',
            'source': 'imu'
        },
        'imu_ay': {
            'description': 'IMU acceleration in Y axis',
            'unit': 'm/s²',
            'type': 'continuous',
            'source': 'imu'
        },
        'imu_az': {
            'description': 'IMU acceleration in Z axis',
            'unit': 'm/s²',
            'type': 'continuous',
            'source': 'imu'
        },
        'imu_gx': {
            'description': 'IMU gyroscope angular velocity in X axis',
            'unit': 'rad/s',
            'type': 'continuous',
            'source': 'imu'
        },
        'imu_gy': {
            'description': 'IMU gyroscope angular velocity in Y axis',
            'unit': 'rad/s',
            'type': 'continuous',
            'source': 'imu'
        },
        'imu_gz': {
            'description': 'IMU gyroscope angular velocity in Z axis',
            'unit': 'rad/s',
            'type': 'continuous',
            'source': 'imu'
        },
        'velocity_x': {
            'description': 'Velocity in East direction',
            'unit': 'm/s',
            'type': 'continuous',
            'source': 'derived'
        },
        'velocity_y': {
            'description': 'Velocity in North direction',
            'unit': 'm/s',
            'type': 'continuous',
            'source': 'derived'
        },
        'velocity_z': {
            'description': 'Velocity in Up direction',
            'unit': 'm/s',
            'type': 'continuous',
            'source': 'derived'
        },
        'hdop': {
            'description': 'Horizontal Dilution of Precision',
            'unit': 'dimensionless',
            'type': 'continuous',
            'source': 'gps'
        },
        'vdop': {
            'description': 'Vertical Dilution of Precision',
            'unit': 'dimensionless',
            'type': 'continuous',
            'source': 'gps'
        },
        'fix_type': {
            'description': 'GPS fix type (0=no fix, 1=dead reckoning, 2=2D, 3=3D, etc.)',
            'unit': 'categorical',
            'type': 'categorical',
            'source': 'gps'
        },
        'ground_truth_x': {
            'description': 'Ground truth position in East direction (sensor fusion)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'ground_truth'
        },
        'ground_truth_y': {
            'description': 'Ground truth position in North direction (sensor fusion)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'ground_truth'
        },
        'ground_truth_z': {
            'description': 'Ground truth position in Up direction (sensor fusion)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'ground_truth'
        },
        'gps_error_x': {
            'description': 'GPS error in East direction (GPS - ground truth)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'calculated'
        },
        'gps_error_y': {
            'description': 'GPS error in North direction (GPS - ground truth)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'calculated'
        },
        'gps_error_z': {
            'description': 'GPS error in Up direction (GPS - ground truth)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'calculated'
        },
        'gps_error_norm': {
            'description': 'GPS error magnitude (Euclidean norm of error vector)',
            'unit': 'meters',
            'type': 'continuous',
            'source': 'calculated'
        }
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metadata generator.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        super().__init__(config)
        
    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate metadata for the given dataset.
        
        Args:
            data: Input DataFrame to generate metadata for
            
        Returns:
            Comprehensive metadata dictionary
        """
        logger.info("Generating dataset metadata")
        
        metadata = {
            'dataset_info': self._generate_dataset_info(data),
            'features': self._generate_feature_metadata(data),
            'processing_info': self._generate_processing_info(),
            'data_quality': self._generate_data_quality_info(data),
            'temporal_info': self._generate_temporal_info(data),
            'statistics': self._generate_statistics(data),
            'generation_info': self._generate_generation_info()
        }
        
        logger.info(f"Generated metadata for {len(data)} samples with {len(data.columns)} features")
        return metadata
    
    def _generate_dataset_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate basic dataset information."""
        return {
            'name': 'UAV Log Dataset',
            'description': 'Time-aligned UAV sensor data for GPS error correction training',
            'version': '1.0',
            'total_samples': len(data),
            'total_features': len(data.columns),
            'memory_usage_mb': float(data.memory_usage(deep=True).sum() / 1024 / 1024),
            'coordinate_system': self.config.get('coordinate_system', 'ENU'),
            'target_frequency_hz': self.config.get('target_frequency', 15.0)
        }
    
    def _generate_feature_metadata(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Generate detailed metadata for each feature."""
        feature_metadata = {}
        
        for column in data.columns:
            # Get base description from predefined descriptions
            base_info = self.FEATURE_DESCRIPTIONS.get(column, {
                'description': f'Feature: {column}',
                'unit': 'unknown',
                'type': 'categorical' if column in self.CATEGORICAL_FEATURES or (pd.api.types.is_numeric_dtype(data[column]) and data[column].nunique() <= 10 and column == 'fix_type') else ('continuous' if pd.api.types.is_numeric_dtype(data[column]) else 'categorical'),
                'source': 'unknown'
            })
            
            # Add statistical information
            feature_info = base_info.copy()
            feature_info.update(self._calculate_feature_statistics(data[column]))
            
            feature_metadata[column] = feature_info
        
        return feature_metadata
    
    def _calculate_feature_statistics(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistical information for a feature."""
        # Use safe conversion to handle potential NaN values
        count_val = series.count()
        missing_val = series.isnull().sum()
        
        stats = {
            'count': int(count_val) if pd.notna(count_val) else 0,
            'missing_count': int(missing_val) if pd.notna(missing_val) else 0,
            'missing_percentage': float(missing_val / len(series) * 100) if len(series) > 0 and pd.notna(missing_val) else 0.0
        }
        
        # Check if this is a categorical feature based on name or characteristics
        series_name = series.name if hasattr(series, 'name') else ''
        is_categorical = (
            series_name in self.CATEGORICAL_FEATURES or
            (pd.api.types.is_numeric_dtype(series) and series.nunique() <= 10 and series_name == 'fix_type')
        )
        
        if is_categorical or not pd.api.types.is_numeric_dtype(series):
            # Categorical feature statistics
            value_counts = series.value_counts()
            nunique_val = series.nunique()
            
            stats.update({
                'unique_count': int(nunique_val) if pd.notna(nunique_val) else 0,
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 and pd.notna(value_counts.iloc[0]) else None,
                'unique_values': [str(val) for val in series.unique() if pd.notna(val)]
            })
        else:
            # Continuous feature statistics
            stats.update({
                'mean': float(series.mean()) if not series.isnull().all() else None,
                'std': float(series.std()) if not series.isnull().all() else None,
                'min': float(series.min()) if not series.isnull().all() else None,
                'max': float(series.max()) if not series.isnull().all() else None,
                'median': float(series.median()) if not series.isnull().all() else None,
                'q25': float(series.quantile(0.25)) if not series.isnull().all() else None,
                'q75': float(series.quantile(0.75)) if not series.isnull().all() else None
            })
        
        return stats
    
    def _generate_processing_info(self) -> Dict[str, Any]:
        """Generate information about processing parameters."""
        return {
            'synchronization': {
                'target_frequency_hz': self.config.get('target_frequency', 15.0),
                'interpolation_method': self.config.get('interpolation_method', 'linear'),
                'max_gap_seconds': self.config.get('max_gap_seconds', 1.0)
            },
            'motion_classification': {
                'accel_threshold_ms2': self.config.get('accel_threshold', 0.5),
                'gyro_threshold_rads': self.config.get('gyro_threshold', 0.1),
                'min_stationary_duration_s': self.config.get('min_stationary_duration', 3.0),
                'motion_window_size_s': self.config.get('motion_window_size', 5.0)
            },
            'ground_truth_generation': {
                'fusion_method': self.config.get('fusion_method', 'ekf'),
                'drift_correction': self.config.get('drift_correction', True),
                'smoothing_method': self.config.get('smoothing_method', 'cubic_spline')
            },
            'data_quality': {
                'min_gps_fix_type': self.config.get('min_gps_fix_type', 3),
                'max_hdop': self.config.get('max_hdop', 5.0),
                'max_data_loss_warning': self.config.get('max_data_loss_warning', 0.1)
            },
            'normalization': {
                'method': self.config.get('normalization_method', 'zscore')
            }
        }
    
    def _generate_data_quality_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality metrics."""
        total_cells = len(data) * len(data.columns) if len(data) > 0 and len(data.columns) > 0 else 0
        # Safe calculations with NaN handling
        total_missing = data.isnull().sum().sum()
        complete_rows = (~data.isnull().any(axis=1)).sum()
        
        quality_info = {
            'completeness': {
                'total_missing_values': int(total_missing) if pd.notna(total_missing) else 0,
                'missing_percentage': float(total_missing / total_cells * 100) if total_cells > 0 and pd.notna(total_missing) else 0.0,
                'complete_rows': int(complete_rows) if pd.notna(complete_rows) else 0,
                'complete_rows_percentage': float(complete_rows / len(data) * 100) if len(data) > 0 and pd.notna(complete_rows) else 0.0
            }
        }
        
        # GPS quality metrics if available
        if 'fix_type' in data.columns:
            fix_type_counts = data['fix_type'].value_counts()
            high_quality_count = (data['fix_type'] >= 3).sum()
            
            quality_info['gps_quality'] = {
                'fix_type_distribution': {str(k): int(v) if pd.notna(v) else 0 for k, v in fix_type_counts.items()},
                'high_quality_fixes': int(high_quality_count) if pd.notna(high_quality_count) else 0,
                'high_quality_percentage': float(high_quality_count / len(data) * 100) if len(data) > 0 and pd.notna(high_quality_count) else 0.0
            }
        
        # HDOP quality if available
        if 'hdop' in data.columns:
            good_hdop_count = (data['hdop'] <= 2.0).sum()
            
            quality_info['hdop_quality'] = {
                'mean_hdop': float(data['hdop'].mean()) if not data['hdop'].isnull().all() else None,
                'good_hdop_count': int(good_hdop_count) if pd.notna(good_hdop_count) else 0,
                'good_hdop_percentage': float(good_hdop_count / len(data) * 100) if len(data) > 0 and pd.notna(good_hdop_count) else 0.0
            }
        
        return quality_info
    
    def _generate_temporal_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate temporal information about the dataset."""
        temporal_info = {}
        
        if 'timestamp' in data.columns:
            timestamps = data['timestamp'].dropna()
            if len(timestamps) > 0:
                duration = float(timestamps.max() - timestamps.min())
                
                # Calculate actual sampling rate
                if len(timestamps) > 1:
                    time_diffs = timestamps.diff().dropna()
                    actual_frequency = 1.0 / time_diffs.mean() if time_diffs.mean() > 0 else 0.0
                else:
                    actual_frequency = 0.0
                
                temporal_info = {
                    'start_time': float(timestamps.min()),
                    'end_time': float(timestamps.max()),
                    'duration_seconds': duration,
                    'duration_minutes': duration / 60.0,
                    'actual_frequency_hz': float(actual_frequency),
                    'frequency_deviation': float(abs(actual_frequency - self.config.get('target_frequency', 15.0))),
                    'time_gaps': self._analyze_time_gaps(timestamps)
                }
        
        return temporal_info
    
    def _analyze_time_gaps(self, timestamps: pd.Series) -> Dict[str, Any]:
        """Analyze time gaps in the timestamp series."""
        if len(timestamps) < 2:
            return {'gap_count': 0, 'max_gap_seconds': 0.0, 'mean_gap_seconds': 0.0}
        
        time_diffs = timestamps.diff().dropna()
        expected_interval = 1.0 / self.config.get('target_frequency', 15.0)
        gap_threshold = expected_interval * 2.0  # Gaps larger than 2x expected interval
        
        gaps = time_diffs[time_diffs > gap_threshold]
        
        return {
            'gap_count': len(gaps),
            'max_gap_seconds': float(gaps.max()) if len(gaps) > 0 else 0.0,
            'mean_gap_seconds': float(gaps.mean()) if len(gaps) > 0 else 0.0,
            'total_gap_time': float(gaps.sum()) if len(gaps) > 0 else 0.0
        }
    
    def _generate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate overall dataset statistics."""
        stats = {
            'shape': {
                'rows': len(data),
                'columns': len(data.columns)
            },
            'data_types': {
                'numeric_features': len(data.select_dtypes(include=[np.number]).columns),
                'categorical_features': len(data.select_dtypes(exclude=[np.number]).columns)
            }
        }
        
        # Error statistics if available
        if 'gps_error_norm' in data.columns:
            error_series = data['gps_error_norm'].dropna()
            if len(error_series) > 0:
                stats['error_statistics'] = {
                    'mean_error_m': float(error_series.mean()),
                    'std_error_m': float(error_series.std()),
                    'min_error_m': float(error_series.min()),
                    'max_error_m': float(error_series.max()),
                    'median_error_m': float(error_series.median()),
                    'q95_error_m': float(error_series.quantile(0.95))
                }
        
        return stats
    
    def _generate_generation_info(self) -> Dict[str, Any]:
        """Generate information about metadata generation."""
        return {
            'generated_at': datetime.now().isoformat(),
            'generator_version': '1.0',
            'python_version': f"{pd.__version__}",
            'pandas_version': pd.__version__
        }
    
    def generate_with_normalization_stats(self, data: pd.DataFrame, 
                                        normalization_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate metadata including normalization statistics.
        
        Args:
            data: Input DataFrame
            normalization_stats: Normalization statistics from dataset formatter
            
        Returns:
            Metadata dictionary with normalization information
        """
        metadata = self.process(data)
        
        # Add normalization statistics to feature metadata
        if normalization_stats:
            for feature, stats in normalization_stats.items():
                if feature in metadata['features']:
                    metadata['features'][feature]['normalization'] = stats
        
        # Add normalization summary
        metadata['normalization_summary'] = {
            'normalized_features': [f for f, s in normalization_stats.items() 
                                  if s.get('normalized', False)],
            'normalization_method': self.config.get('normalization_method', 'zscore'),
            'total_normalized': len([f for f, s in normalization_stats.items() 
                                   if s.get('normalized', False)])
        }
        
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], output_path: str):
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary to save
            output_path: Path where to save the metadata.json file
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save with pretty formatting
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved metadata to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def generate_feature_summary(self, data: pd.DataFrame) -> str:
        """
        Generate a human-readable summary of features.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Formatted string with feature summary
        """
        summary_lines = [
            f"Dataset Summary: {len(data)} samples, {len(data.columns)} features",
            "=" * 60
        ]
        
        # Group features by source
        feature_groups = {}
        for column in data.columns:
            source = self.FEATURE_DESCRIPTIONS.get(column, {}).get('source', 'unknown')
            if source not in feature_groups:
                feature_groups[source] = []
            feature_groups[source].append(column)
        
        for source, features in feature_groups.items():
            summary_lines.append(f"\n{source.upper()} Features ({len(features)}):")
            for feature in features:
                desc = self.FEATURE_DESCRIPTIONS.get(feature, {})
                unit = desc.get('unit', 'unknown')
                description = desc.get('description', feature)
                summary_lines.append(f"  - {feature} ({unit}): {description}")
        
        return "\n".join(summary_lines)