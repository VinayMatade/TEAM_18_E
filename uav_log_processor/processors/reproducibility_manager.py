"""
Reproducibility manager for UAV log processing.

Handles saving aligned_full.csv, processing logs, configuration saving,
and data quality report generation for full reproducibility.
"""

from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import sys

from .base import BaseProcessor


logger = logging.getLogger(__name__)


class ReproducibilityManager(BaseProcessor):
    """
    Manages reproducibility outputs for UAV log processing.
    
    Saves all necessary data and metadata to ensure processing
    can be reproduced and validated.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize reproducibility manager.
        
        Args:
            config: Configuration dictionary with processing parameters
        """
        super().__init__(config)
        self.processing_log = []
        self.data_quality_metrics = {}
        
    def process(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Process data for reproducibility outputs.
        
        Args:
            data: Input DataFrame to process
            
        Returns:
            Dictionary with reproducibility information
        """
        logger.info("Processing data for reproducibility outputs")
        
        # Generate data quality report
        quality_report = self.generate_data_quality_report(data)
        
        # Create processing summary
        processing_summary = self.create_processing_summary()
        
        return {
            'quality_report': quality_report,
            'processing_summary': processing_summary,
            'data_hash': self._calculate_data_hash(data)
        }
    
    def save_aligned_full_data(self, data: pd.DataFrame, 
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
        logger.info("Saving aligned full dataset for reproducibility")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        try:
            # Save with high precision for reproducibility
            data.to_csv(file_path, index=False, float_format='%.8f')
            
            # Calculate and log file statistics
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            data_hash = self._calculate_data_hash(data)
            
            self.log_processing_step(
                "save_aligned_full",
                f"Saved {len(data)} samples to {file_path}",
                {
                    'file_size_mb': file_size_mb,
                    'data_hash': data_hash,
                    'columns': list(data.columns),
                    'shape': data.shape
                }
            )
            
            logger.info(f"Saved aligned full dataset: {len(data)} samples to {file_path} ({file_size_mb:.2f} MB)")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save aligned full dataset: {e}")
            raise
    
    def save_processing_log(self, output_dir: str = "output",
                          filename: str = "processing_log.json") -> str:
        """
        Save detailed processing log for reproducibility.
        
        Args:
            output_dir: Output directory
            filename: Log filename
            
        Returns:
            Path to saved log file
        """
        logger.info("Saving processing log")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        # Create comprehensive log
        processing_log = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'python_version': sys.version,
                'platform': platform.platform(),
                'processor': 'UAV Log Processor v1.0'
            },
            'configuration': self.config,
            'processing_steps': self.processing_log,
            'environment': self._get_environment_info(),
            'data_quality': self.data_quality_metrics
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(processing_log, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved processing log to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save processing log: {e}")
            raise
    
    def save_configuration(self, config: Dict[str, Any],
                         output_dir: str = "output",
                         filename: str = "processing_config.json") -> str:
        """
        Save processing configuration for reproducibility.
        
        Args:
            config: Configuration dictionary to save
            output_dir: Output directory
            filename: Configuration filename
            
        Returns:
            Path to saved configuration file
        """
        logger.info("Saving processing configuration")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        # Add metadata to configuration
        config_with_metadata = {
            'metadata': {
                'saved_at': datetime.now().isoformat(),
                'version': '1.0',
                'description': 'UAV Log Processing Configuration'
            },
            'configuration': config
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(config_with_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved configuration to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise
    
    def generate_data_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        logger.info("Generating data quality report")
        
        report = {
            'overview': self._generate_overview_metrics(data),
            'completeness': self._analyze_data_completeness(data),
            'consistency': self._analyze_data_consistency(data),
            'accuracy': self._analyze_data_accuracy(data),
            'temporal_quality': self._analyze_temporal_quality(data),
            'sensor_quality': self._analyze_sensor_quality(data)
        }
        
        # Store for later use
        self.data_quality_metrics = report
        
        return report
    
    def _generate_overview_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate overview metrics for the dataset."""
        return {
            'total_samples': len(data),
            'total_features': len(data.columns),
            'memory_usage_mb': float(data.memory_usage(deep=True).sum() / 1024 / 1024),
            'data_types': {
                'numeric': len(data.select_dtypes(include=[np.number]).columns),
                'categorical': len(data.select_dtypes(exclude=[np.number]).columns)
            },
            'data_hash': self._calculate_data_hash(data)
        }
    
    def _analyze_data_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data completeness metrics."""
        total_cells = len(data) * len(data.columns) if not data.empty else 0
        missing_cells = data.isnull().sum().sum() if not data.empty else 0
        
        completeness = {
            'total_cells': total_cells,
            'missing_cells': int(missing_cells),
            'missing_percentage': float(missing_cells / total_cells * 100) if total_cells > 0 else 0.0,
            'complete_rows': int((~data.isnull().any(axis=1)).sum()) if not data.empty else 0,
            'complete_rows_percentage': float((~data.isnull().any(axis=1)).sum() / len(data) * 100) if len(data) > 0 else 0.0,
            'feature_completeness': {}
        }
        
        # Per-feature completeness
        for column in data.columns:
            missing_count = data[column].isnull().sum() if not data.empty else 0
            completeness['feature_completeness'][column] = {
                'missing_count': int(missing_count),
                'missing_percentage': float(missing_count / len(data) * 100) if len(data) > 0 else 0.0,
                'complete_count': int(len(data) - missing_count)
            }
        
        return completeness
    
    def _analyze_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data consistency metrics."""
        consistency = {
            'duplicate_rows': int(data.duplicated().sum()) if not data.empty else 0,
            'duplicate_percentage': float(data.duplicated().sum() / len(data) * 100) if len(data) > 0 else 0.0
        }
        
        # Check for timestamp consistency
        if 'timestamp' in data.columns:
            timestamps = data['timestamp'].dropna()
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dropna()
                expected_interval = 1.0 / self.config.get('target_frequency', 15.0)
                
                consistency['temporal_consistency'] = {
                    'expected_interval_s': expected_interval,
                    'mean_interval_s': float(time_diffs.mean()),
                    'std_interval_s': float(time_diffs.std()),
                    'irregular_intervals': int((abs(time_diffs - expected_interval) > expected_interval * 0.1).sum()),
                    'irregular_percentage': float((abs(time_diffs - expected_interval) > expected_interval * 0.1).sum() / len(time_diffs) * 100)
                }
        
        return consistency
    
    def _analyze_data_accuracy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data accuracy metrics."""
        accuracy = {}
        
        # GPS accuracy metrics
        if 'fix_type' in data.columns:
            fix_type_counts = data['fix_type'].value_counts()
            accuracy['gps_accuracy'] = {
                'fix_type_distribution': {str(k): int(v) for k, v in fix_type_counts.items()},
                'high_accuracy_fixes': int((data['fix_type'] >= 3).sum()),
                'high_accuracy_percentage': float((data['fix_type'] >= 3).sum() / len(data) * 100)
            }
        
        # HDOP accuracy
        if 'hdop' in data.columns:
            hdop_data = data['hdop'].dropna()
            accuracy['hdop_accuracy'] = {
                'mean_hdop': float(hdop_data.mean()),
                'std_hdop': float(hdop_data.std()),
                'excellent_hdop': int((hdop_data <= 1.0).sum()),  # HDOP <= 1.0
                'good_hdop': int(((hdop_data > 1.0) & (hdop_data <= 2.0)).sum()),  # 1.0 < HDOP <= 2.0
                'moderate_hdop': int(((hdop_data > 2.0) & (hdop_data <= 5.0)).sum()),  # 2.0 < HDOP <= 5.0
                'poor_hdop': int((hdop_data > 5.0).sum())  # HDOP > 5.0
            }
        
        # IMU data range checks
        imu_features = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
        imu_present = [f for f in imu_features if f in data.columns]
        
        if imu_present:
            accuracy['imu_accuracy'] = {}
            for feature in imu_present:
                feature_data = data[feature].dropna()
                if len(feature_data) > 0:
                    # Check for reasonable ranges (basic sanity checks)
                    if 'accel' in feature or '_a' in feature:
                        # Accelerometer: typically -20g to +20g (±196 m/s²)
                        outliers = int((abs(feature_data) > 196).sum())
                    else:
                        # Gyroscope: typically -2000 to +2000 deg/s (±35 rad/s)
                        outliers = int((abs(feature_data) > 35).sum())
                    
                    accuracy['imu_accuracy'][feature] = {
                        'outlier_count': outliers,
                        'outlier_percentage': float(outliers / len(feature_data) * 100),
                        'range': [float(feature_data.min()), float(feature_data.max())],
                        'std': float(feature_data.std())
                    }
        
        return accuracy
    
    def _analyze_temporal_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal data quality."""
        temporal_quality = {}
        
        if 'timestamp' in data.columns:
            timestamps = data['timestamp'].dropna()
            if len(timestamps) >= 1:
                if len(timestamps) == 1:
                    # Single timestamp case
                    temporal_quality = {
                        'duration_seconds': 0.0,
                        'duration_minutes': 0.0,
                        'duration_hours': 0.0,
                        'sampling_rate_hz': 0.0,
                        'time_gaps': self._analyze_time_gaps(timestamps),
                        'monotonic': True
                    }
                else:
                    # Multiple timestamps case
                    duration = float(timestamps.max() - timestamps.min())
                    time_diffs = timestamps.diff().dropna()
                    
                    temporal_quality = {
                        'duration_seconds': duration,
                        'duration_minutes': duration / 60.0,
                        'duration_hours': duration / 3600.0,
                        'sampling_rate_hz': float(len(timestamps) / duration) if duration > 0 else 0.0,
                        'time_gaps': self._analyze_time_gaps(timestamps),
                        'monotonic': bool(timestamps.is_monotonic_increasing)
                    }
        
        return temporal_quality
    
    def _analyze_time_gaps(self, timestamps: pd.Series) -> Dict[str, Any]:
        """Analyze time gaps in the timestamp series."""
        if len(timestamps) < 2:
            return {'gap_count': 0, 'max_gap_seconds': 0.0, 'mean_gap_seconds': 0.0, 'total_gap_time': 0.0}
        
        time_diffs = timestamps.diff().dropna()
        expected_interval = 1.0 / self.config.get('target_frequency', 15.0)
        gap_threshold = expected_interval * 2.0  # Gaps larger than 2x expected interval
        
        gaps = time_diffs[time_diffs > gap_threshold]
        
        return {
            'gap_count': len(gaps),
            'max_gap_seconds': float(gaps.max()) if len(gaps) > 0 else 0.0,
            'mean_gap_seconds': float(gaps.mean()) if len(gaps) > 0 else 0.0,
            'total_gap_time': float(gaps.sum()) if len(gaps) > 0 else 0.0,
            'gap_percentage': float(gaps.sum() / (timestamps.iloc[-1] - timestamps.iloc[0]) * 100) if len(gaps) > 0 and len(timestamps) > 1 else 0.0
        }
    
    def _analyze_sensor_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sensor-specific quality metrics."""
        sensor_quality = {}
        
        # GPS sensor quality
        gps_features = ['gps_x', 'gps_y', 'gps_z']
        if all(f in data.columns for f in gps_features):
            gps_data = data[gps_features].dropna()
            if len(gps_data) > 0:
                # Calculate GPS movement statistics
                if len(gps_data) > 1:
                    distances = np.sqrt(gps_data.diff().pow(2).sum(axis=1)).dropna()
                    sensor_quality['gps'] = {
                        'total_distance_m': float(distances.sum()),
                        'max_step_distance_m': float(distances.max()),
                        'mean_step_distance_m': float(distances.mean()),
                        'stationary_samples': int((distances < 0.1).sum()),  # Less than 10cm movement
                        'stationary_percentage': float((distances < 0.1).sum() / len(distances) * 100)
                    }
        
        # IMU sensor quality
        imu_features = ['imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz']
        imu_present = [f for f in imu_features if f in data.columns]
        
        if imu_present:
            sensor_quality['imu'] = {}
            
            # Accelerometer quality
            accel_features = [f for f in imu_present if 'a' in f.split('_')[1]]
            if accel_features:
                accel_data = data[accel_features].dropna()
                if len(accel_data) > 0:
                    accel_magnitude = np.sqrt(accel_data.pow(2).sum(axis=1))
                    sensor_quality['imu']['accelerometer'] = {
                        'mean_magnitude_ms2': float(accel_magnitude.mean()),
                        'std_magnitude_ms2': float(accel_magnitude.std()),
                        'gravity_bias': float(abs(accel_magnitude.mean() - 9.81)),  # Deviation from gravity
                        'zero_readings': int((accel_magnitude < 0.1).sum())
                    }
            
            # Gyroscope quality
            gyro_features = [f for f in imu_present if 'g' in f.split('_')[1]]
            if gyro_features:
                gyro_data = data[gyro_features].dropna()
                if len(gyro_data) > 0:
                    gyro_magnitude = np.sqrt(gyro_data.pow(2).sum(axis=1))
                    sensor_quality['imu']['gyroscope'] = {
                        'mean_magnitude_rads': float(gyro_magnitude.mean()),
                        'std_magnitude_rads': float(gyro_magnitude.std()),
                        'stationary_samples': int((gyro_magnitude < 0.01).sum()),  # Very low rotation
                        'high_rotation_samples': int((gyro_magnitude > 1.0).sum())  # High rotation
                    }
        
        return sensor_quality
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of the dataset for integrity verification."""
        # Create a string representation of the data
        data_string = data.to_csv(index=False)
        
        # Calculate SHA-256 hash
        hash_object = hashlib.sha256(data_string.encode())
        return hash_object.hexdigest()
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information for reproducibility."""
        try:
            import pandas as pd
            import numpy as np
            
            return {
                'python_version': sys.version,
                'platform': platform.platform(),
                'architecture': platform.architecture(),
                'processor': platform.processor(),
                'pandas_version': pd.__version__,
                'numpy_version': np.__version__,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.warning(f"Could not collect full environment info: {e}")
            return {
                'python_version': sys.version,
                'platform': platform.platform(),
                'timestamp': datetime.now().isoformat()
            }
    
    def log_processing_step(self, step_name: str, description: str, 
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Log a processing step for reproducibility.
        
        Args:
            step_name: Name of the processing step
            description: Description of what was done
            metadata: Optional metadata about the step
        """
        step_log = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'description': description,
            'metadata': metadata or {}
        }
        
        self.processing_log.append(step_log)
        logger.debug(f"Logged processing step: {step_name}")
    
    def create_processing_summary(self) -> Dict[str, Any]:
        """Create a summary of all processing steps."""
        return {
            'total_steps': len(self.processing_log),
            'processing_duration': self._calculate_processing_duration(),
            'steps': self.processing_log
        }
    
    def _calculate_processing_duration(self) -> Optional[float]:
        """Calculate total processing duration from logged steps."""
        if len(self.processing_log) < 2:
            return None
        
        try:
            start_time = datetime.fromisoformat(self.processing_log[0]['timestamp'])
            end_time = datetime.fromisoformat(self.processing_log[-1]['timestamp'])
            duration = (end_time - start_time).total_seconds()
            return duration
        except Exception as e:
            logger.warning(f"Could not calculate processing duration: {e}")
            return None
    
    def save_data_quality_report(self, quality_report: Dict[str, Any],
                               output_dir: str = "output",
                               filename: str = "data_quality_report.json") -> str:
        """
        Save data quality report to file.
        
        Args:
            quality_report: Quality report dictionary
            output_dir: Output directory
            filename: Report filename
            
        Returns:
            Path to saved report file
        """
        logger.info("Saving data quality report")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        
        # Add metadata to report
        report_with_metadata = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0',
                'description': 'UAV Log Data Quality Report'
            },
            'quality_report': quality_report
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(report_with_metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Saved data quality report to {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Failed to save data quality report: {e}")
            raise
    
    def create_reproducibility_package(self, data: pd.DataFrame,
                                     config: Dict[str, Any],
                                     output_dir: str = "output") -> Dict[str, str]:
        """
        Create complete reproducibility package.
        
        Args:
            data: Complete dataset
            config: Processing configuration
            output_dir: Output directory
            
        Returns:
            Dictionary with paths to all saved files
        """
        logger.info("Creating complete reproducibility package")
        
        saved_files = {}
        
        # Save aligned full data
        saved_files['aligned_data'] = self.save_aligned_full_data(data, output_dir)
        
        # Save configuration
        saved_files['configuration'] = self.save_configuration(config, output_dir)
        
        # Save processing log
        saved_files['processing_log'] = self.save_processing_log(output_dir)
        
        # Generate and save data quality report
        quality_report = self.generate_data_quality_report(data)
        saved_files['quality_report'] = self.save_data_quality_report(quality_report, output_dir)
        
        logger.info(f"Created reproducibility package with {len(saved_files)} files")
        return saved_files