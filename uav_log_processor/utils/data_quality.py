"""
Data quality validation and reporting utilities.

Provides comprehensive data quality assessment, validation checks, and reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class DataQualityMetrics:
    """Data quality metrics container."""
    
    # Basic metrics
    total_samples: int = 0
    valid_samples: int = 0
    data_loss_ratio: float = 0.0
    
    # Temporal metrics
    duration_seconds: float = 0.0
    sampling_rate_hz: float = 0.0
    time_gaps_count: int = 0
    max_gap_seconds: float = 0.0
    
    # GPS quality metrics
    gps_fix_ratio: float = 0.0
    avg_hdop: float = np.nan
    avg_vdop: float = np.nan
    position_std_m: float = np.nan
    
    # IMU quality metrics
    imu_data_ratio: float = 0.0
    accel_std: float = np.nan
    gyro_std: float = np.nan
    
    # Processing metrics
    processing_warnings: List[str] = field(default_factory=list)
    processing_errors: List[str] = field(default_factory=list)
    
    # Quality score (0-100)
    overall_quality_score: float = 0.0


class DataQualityValidator:
    """Validates data quality and generates comprehensive reports."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize data quality validator.
        
        Args:
            config: Configuration dictionary with quality thresholds
        """
        self.config = config or {}
        
        # Quality thresholds
        self.max_data_loss_warning = self.config.get('max_data_loss_warning', 0.1)  # 10%
        self.min_sampling_rate = self.config.get('min_sampling_rate', 5.0)  # Hz
        self.max_time_gap = self.config.get('max_time_gap', 5.0)  # seconds
        self.min_gps_fix_ratio = self.config.get('min_gps_fix_ratio', 0.8)  # 80%
        self.max_hdop_threshold = self.config.get('max_hdop_threshold', 5.0)
        self.min_position_stability = self.config.get('min_position_stability', 100.0)  # meters std
        
        self.warnings = []
        self.errors = []
    
    def validate_dataset(self, df: pd.DataFrame, 
                        original_count: Optional[int] = None) -> DataQualityMetrics:
        """
        Perform comprehensive data quality validation.
        
        Args:
            df: DataFrame to validate
            original_count: Original sample count before processing (for loss calculation)
            
        Returns:
            DataQualityMetrics object with validation results
        """
        if df is None or df.empty:
            self._add_error("Dataset is empty or None")
            return DataQualityMetrics(
                processing_errors=self.errors.copy(),
                processing_warnings=self.warnings.copy()
            )
        
        logger.info(f"Validating dataset with {len(df)} samples")
        
        # Initialize metrics
        metrics = DataQualityMetrics()
        metrics.total_samples = len(df)
        
        # Calculate data loss
        if original_count is not None:
            metrics.data_loss_ratio = max(0, (original_count - len(df)) / original_count)
            if metrics.data_loss_ratio > self.max_data_loss_warning:
                self._add_warning(f"High data loss: {metrics.data_loss_ratio:.1%} "
                                f"({original_count - len(df)}/{original_count} samples)")
        
        # Validate temporal characteristics
        self._validate_temporal_quality(df, metrics)
        
        # Validate GPS data quality
        self._validate_gps_quality(df, metrics)
        
        # Validate IMU data quality
        self._validate_imu_quality(df, metrics)
        
        # Calculate overall quality score
        metrics.overall_quality_score = self._calculate_quality_score(metrics)
        
        # Add warnings and errors to metrics
        metrics.processing_warnings = self.warnings.copy()
        metrics.processing_errors = self.errors.copy()
        
        # Log summary
        self._log_quality_summary(metrics)
        
        return metrics
    
    def _validate_temporal_quality(self, df: pd.DataFrame, metrics: DataQualityMetrics):
        """Validate temporal data quality."""
        if 'timestamp' not in df.columns:
            self._add_error("No timestamp column found")
            return
        
        timestamps = df['timestamp'].dropna()
        if timestamps.empty:
            self._add_error("No valid timestamps found")
            return
        
        # Calculate duration and sampling rate
        metrics.duration_seconds = timestamps.max() - timestamps.min()
        if metrics.duration_seconds > 0:
            metrics.sampling_rate_hz = len(timestamps) / metrics.duration_seconds
        
        # Check sampling rate
        if metrics.sampling_rate_hz < self.min_sampling_rate:
            self._add_warning(f"Low sampling rate: {metrics.sampling_rate_hz:.1f} Hz "
                            f"(minimum: {self.min_sampling_rate} Hz)")
        
        # Analyze time gaps
        time_diffs = timestamps.diff().dropna()
        if not time_diffs.empty:
            large_gaps = time_diffs[time_diffs > self.max_time_gap]
            metrics.time_gaps_count = len(large_gaps)
            metrics.max_gap_seconds = time_diffs.max()
            
            if metrics.time_gaps_count > 0:
                self._add_warning(f"Found {metrics.time_gaps_count} large time gaps "
                                f"(max: {metrics.max_gap_seconds:.1f}s)")
    
    def _validate_gps_quality(self, df: pd.DataFrame, metrics: DataQualityMetrics):
        """Validate GPS data quality."""
        gps_columns = ['gps_lat', 'gps_lon', 'gps_alt']
        available_gps_cols = [col for col in gps_columns if col in df.columns]
        
        if not available_gps_cols:
            self._add_error("No GPS coordinate columns found")
            return
        
        # Count valid GPS samples
        gps_data = df[available_gps_cols].dropna()
        metrics.valid_samples = len(gps_data)
        
        if metrics.valid_samples == 0:
            self._add_error("No valid GPS data found")
            return
        
        # GPS fix quality
        if 'fix_type' in df.columns:
            fix_data = df['fix_type'].dropna()
            if not fix_data.empty:
                good_fixes = (fix_data >= 3).sum()
                metrics.gps_fix_ratio = good_fixes / len(fix_data)
                
                if metrics.gps_fix_ratio < self.min_gps_fix_ratio:
                    self._add_warning(f"Low GPS fix quality: {metrics.gps_fix_ratio:.1%} "
                                    f"good fixes (minimum: {self.min_gps_fix_ratio:.1%})")
        
        # HDOP/VDOP analysis
        if 'hdop' in df.columns:
            hdop_data = df['hdop'].dropna()
            if not hdop_data.empty:
                metrics.avg_hdop = hdop_data.mean()
                if metrics.avg_hdop > self.max_hdop_threshold:
                    self._add_warning(f"High average HDOP: {metrics.avg_hdop:.2f} "
                                    f"(threshold: {self.max_hdop_threshold})")
        
        if 'vdop' in df.columns:
            vdop_data = df['vdop'].dropna()
            if not vdop_data.empty:
                metrics.avg_vdop = vdop_data.mean()
        
        # Position stability analysis
        if 'gps_lat' in df.columns and 'gps_lon' in df.columns:
            lat_data = df['gps_lat'].dropna()
            lon_data = df['gps_lon'].dropna()
            
            if not lat_data.empty and not lon_data.empty:
                # Convert to approximate meters
                lat_std_m = lat_data.std() * 111000  # degrees to meters
                lon_std_m = lon_data.std() * 111000 * np.cos(np.radians(lat_data.mean()))
                metrics.position_std_m = np.sqrt(lat_std_m**2 + lon_std_m**2)
                
                if metrics.position_std_m > self.min_position_stability:
                    self._add_warning(f"High position variability: {metrics.position_std_m:.1f}m std "
                                    f"(threshold: {self.min_position_stability}m)")
    
    def _validate_imu_quality(self, df: pd.DataFrame, metrics: DataQualityMetrics):
        """Validate IMU data quality."""
        imu_accel_cols = ['imu_ax', 'imu_ay', 'imu_az']
        imu_gyro_cols = ['imu_gx', 'imu_gy', 'imu_gz']
        
        available_accel_cols = [col for col in imu_accel_cols if col in df.columns]
        available_gyro_cols = [col for col in imu_gyro_cols if col in df.columns]
        
        if not available_accel_cols and not available_gyro_cols:
            self._add_warning("No IMU data columns found")
            return
        
        # Count valid IMU samples
        imu_cols = available_accel_cols + available_gyro_cols
        imu_data = df[imu_cols].dropna()
        imu_sample_count = len(imu_data)
        
        if metrics.total_samples > 0:
            metrics.imu_data_ratio = imu_sample_count / metrics.total_samples
        
        if metrics.imu_data_ratio < 0.8:  # Less than 80% IMU data
            self._add_warning(f"Low IMU data availability: {metrics.imu_data_ratio:.1%}")
        
        # Accelerometer analysis
        if available_accel_cols:
            accel_data = df[available_accel_cols].dropna()
            if not accel_data.empty:
                # Calculate magnitude standard deviation
                accel_mag = np.sqrt((accel_data**2).sum(axis=1))
                metrics.accel_std = accel_mag.std()
                
                # Check for unrealistic values
                if (accel_data.abs() > 50).any().any():  # > 50 m/s²
                    self._add_warning("Detected unrealistic accelerometer values (>50 m/s²)")
        
        # Gyroscope analysis
        if available_gyro_cols:
            gyro_data = df[available_gyro_cols].dropna()
            if not gyro_data.empty:
                # Calculate magnitude standard deviation
                gyro_mag = np.sqrt((gyro_data**2).sum(axis=1))
                metrics.gyro_std = gyro_mag.std()
                
                # Check for unrealistic values
                if (gyro_data.abs() > 50).any().any():  # > 50 rad/s
                    self._add_warning("Detected unrealistic gyroscope values (>50 rad/s)")
    
    def _calculate_quality_score(self, metrics: DataQualityMetrics) -> float:
        """Calculate overall quality score (0-100)."""
        score = 100.0
        
        # Data loss penalty
        if metrics.data_loss_ratio > 0:
            score -= min(metrics.data_loss_ratio * 50, 30)  # Max 30 point penalty
        
        # Sampling rate penalty
        if metrics.sampling_rate_hz < self.min_sampling_rate:
            score -= min((self.min_sampling_rate - metrics.sampling_rate_hz) * 5, 20)
        
        # GPS quality penalty
        if metrics.gps_fix_ratio < self.min_gps_fix_ratio:
            score -= (self.min_gps_fix_ratio - metrics.gps_fix_ratio) * 30
        
        # HDOP penalty
        if not np.isnan(metrics.avg_hdop) and metrics.avg_hdop > self.max_hdop_threshold:
            score -= min((metrics.avg_hdop - self.max_hdop_threshold) * 5, 15)
        
        # IMU data availability penalty
        if metrics.imu_data_ratio < 0.8:
            score -= (0.8 - metrics.imu_data_ratio) * 25
        
        # Time gaps penalty
        if metrics.time_gaps_count > 0:
            score -= min(metrics.time_gaps_count * 2, 10)
        
        # Error penalty
        score -= len(metrics.processing_errors) * 10
        
        # Warning penalty
        score -= len(metrics.processing_warnings) * 2
        
        return max(0.0, score)
    
    def _add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def _add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)
    
    def _log_quality_summary(self, metrics: DataQualityMetrics):
        """Log quality summary."""
        logger.info(f"Data Quality Summary:")
        logger.info(f"  Overall Score: {metrics.overall_quality_score:.1f}/100")
        logger.info(f"  Total Samples: {metrics.total_samples}")
        logger.info(f"  Valid GPS Samples: {metrics.valid_samples}")
        logger.info(f"  Data Loss: {metrics.data_loss_ratio:.1%}")
        logger.info(f"  Sampling Rate: {metrics.sampling_rate_hz:.1f} Hz")
        logger.info(f"  GPS Fix Ratio: {metrics.gps_fix_ratio:.1%}")
        logger.info(f"  Warnings: {len(metrics.processing_warnings)}")
        logger.info(f"  Errors: {len(metrics.processing_errors)}")


class DataQualityReporter:
    """Generate comprehensive data quality reports."""
    
    def __init__(self):
        """Initialize data quality reporter."""
        pass
    
    def generate_report(self, metrics: DataQualityMetrics, 
                       output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            metrics: DataQualityMetrics object
            output_path: Optional path to save the report as JSON
            
        Returns:
            Dictionary with the complete report
        """
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_version': '1.0'
            },
            'summary': {
                'overall_quality_score': metrics.overall_quality_score,
                'quality_grade': self._get_quality_grade(metrics.overall_quality_score),
                'total_samples': metrics.total_samples,
                'valid_samples': metrics.valid_samples,
                'data_loss_ratio': metrics.data_loss_ratio
            },
            'temporal_quality': {
                'duration_seconds': metrics.duration_seconds,
                'sampling_rate_hz': metrics.sampling_rate_hz,
                'time_gaps_count': metrics.time_gaps_count,
                'max_gap_seconds': metrics.max_gap_seconds
            },
            'gps_quality': {
                'gps_fix_ratio': metrics.gps_fix_ratio,
                'avg_hdop': metrics.avg_hdop if not np.isnan(metrics.avg_hdop) else None,
                'avg_vdop': metrics.avg_vdop if not np.isnan(metrics.avg_vdop) else None,
                'position_std_m': metrics.position_std_m if not np.isnan(metrics.position_std_m) else None
            },
            'imu_quality': {
                'imu_data_ratio': metrics.imu_data_ratio,
                'accel_std': metrics.accel_std if not np.isnan(metrics.accel_std) else None,
                'gyro_std': metrics.gyro_std if not np.isnan(metrics.gyro_std) else None
            },
            'issues': {
                'warnings': metrics.processing_warnings,
                'errors': metrics.processing_errors
            },
            'recommendations': self._generate_recommendations(metrics)
        }
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Data quality report saved to {output_path}")
        
        return report
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    def _generate_recommendations(self, metrics: DataQualityMetrics) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        if metrics.data_loss_ratio > 0.1:
            recommendations.append("Consider using more robust parsing methods to reduce data loss")
        
        if metrics.sampling_rate_hz < 10:
            recommendations.append("Low sampling rate may affect model training quality")
        
        if metrics.gps_fix_ratio < 0.8:
            recommendations.append("Poor GPS fix quality - consider filtering or using RTK GPS")
        
        if not np.isnan(metrics.avg_hdop) and metrics.avg_hdop > 3:
            recommendations.append("High HDOP values indicate poor GPS geometry")
        
        if metrics.imu_data_ratio < 0.8:
            recommendations.append("Low IMU data availability may affect ground truth generation")
        
        if metrics.time_gaps_count > 5:
            recommendations.append("Multiple time gaps detected - check for sensor failures")
        
        if metrics.overall_quality_score < 70:
            recommendations.append("Overall data quality is poor - consider data preprocessing")
        
        if not recommendations:
            recommendations.append("Data quality is good - no major issues detected")
        
        return recommendations


def validate_processing_pipeline(input_data: Dict[str, pd.DataFrame],
                               output_data: pd.DataFrame,
                               config: Optional[Dict] = None) -> DataQualityMetrics:
    """
    Validate the entire processing pipeline.
    
    Args:
        input_data: Dictionary of input DataFrames by source
        output_data: Final processed DataFrame
        config: Optional configuration for validation
        
    Returns:
        DataQualityMetrics for the pipeline
    """
    validator = DataQualityValidator(config)
    
    # Calculate original sample count
    original_count = sum(len(df) for df in input_data.values())
    
    # Validate output data
    metrics = validator.validate_dataset(output_data, original_count)
    
    # Add pipeline-specific validations
    if len(input_data) > 1:
        logger.info(f"Multi-source processing: {len(input_data)} input sources")
    
    return metrics


def check_data_consistency(df: pd.DataFrame) -> List[str]:
    """
    Check for data consistency issues.
    
    Args:
        df: DataFrame to check
        
    Returns:
        List of consistency issues found
    """
    issues = []
    
    if df.empty:
        return ["DataFrame is empty"]
    
    # Check for duplicate timestamps
    if 'timestamp' in df.columns:
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            issues.append(f"Found {duplicate_timestamps} duplicate timestamps")
    
    # Check for unrealistic GPS coordinates
    if 'gps_lat' in df.columns:
        invalid_lat = ((df['gps_lat'] < -90) | (df['gps_lat'] > 90)).sum()
        if invalid_lat > 0:
            issues.append(f"Found {invalid_lat} invalid latitude values")
    
    if 'gps_lon' in df.columns:
        invalid_lon = ((df['gps_lon'] < -180) | (df['gps_lon'] > 180)).sum()
        if invalid_lon > 0:
            issues.append(f"Found {invalid_lon} invalid longitude values")
    
    # Check for unrealistic altitude values
    if 'gps_alt' in df.columns:
        extreme_alt = ((df['gps_alt'] < -1000) | (df['gps_alt'] > 50000)).sum()
        if extreme_alt > 0:
            issues.append(f"Found {extreme_alt} extreme altitude values")
    
    # Check for constant values (sensor failures)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].nunique() == 1 and not df[col].isna().all():
            issues.append(f"Column '{col}' has constant values (possible sensor failure)")
    
    return issues