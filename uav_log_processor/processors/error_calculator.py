"""
GPS error calculation processor.

Calculates GPS error vectors by comparing raw GPS positions with ground truth positions.
Provides per-axis errors and error magnitude for TCN training targets.
"""

from .base import BaseCalculator
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ErrorCalculator(BaseCalculator):
    """Calculates GPS error vectors for ML training targets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error calculator with configuration.
        
        Args:
            config: Configuration dictionary with optional parameters:
                - max_error_threshold: Maximum allowed error magnitude (default: 100.0 meters)
                - temporal_consistency_window: Window size for consistency checks (default: 5)
        """
        super().__init__(config)
        self.max_error_threshold = self.config.get('max_error_threshold', 100.0)
        self.temporal_consistency_window = self.config.get('temporal_consistency_window', 5)
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process synchronized data to calculate GPS errors.
        
        Args:
            data: DataFrame with GPS and ground truth position columns
            
        Returns:
            DataFrame with added error columns
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data for error calculation")
        
        # Extract GPS and ground truth positions
        gps_data = data[['gps_x', 'gps_y', 'gps_z']].copy()
        ground_truth = data[['ground_truth_x', 'ground_truth_y', 'ground_truth_z']].copy()
        
        # Calculate errors
        errors_df = self.calculate(gps_data, ground_truth)
        
        # Add errors to original data
        result = data.copy()
        for col in errors_df.columns:
            result[col] = errors_df[col]
        
        return result
    
    def calculate(self, gps_data: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GPS error vectors by comparing GPS positions with ground truth.
        
        Args:
            gps_data: DataFrame with GPS position columns (gps_x, gps_y, gps_z)
            ground_truth: DataFrame with ground truth columns (ground_truth_x, ground_truth_y, ground_truth_z)
            
        Returns:
            DataFrame with error columns (gps_error_x, gps_error_y, gps_error_z, gps_error_norm)
        """
        if len(gps_data) != len(ground_truth):
            raise ValueError("GPS data and ground truth must have same length")
        
        # Calculate per-axis errors: error = gps_position - ground_truth_position
        error_x = gps_data['gps_x'].values - ground_truth['ground_truth_x'].values
        error_y = gps_data['gps_y'].values - ground_truth['ground_truth_y'].values  
        error_z = gps_data['gps_z'].values - ground_truth['ground_truth_z'].values
        
        # Calculate error magnitude using Euclidean norm
        error_norm = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        # Create result DataFrame
        errors_df = pd.DataFrame({
            'gps_error_x': error_x,
            'gps_error_y': error_y,
            'gps_error_z': error_z,
            'gps_error_norm': error_norm
        }, index=gps_data.index)
        
        # Ensure temporal consistency
        errors_df = self._ensure_temporal_consistency(errors_df)
        
        logger.info(f"Calculated GPS errors for {len(errors_df)} samples")
        logger.info(f"Mean error magnitude: {error_norm.mean():.3f}m, "
                   f"Max error magnitude: {error_norm.max():.3f}m")
        
        return errors_df
    
    def _ensure_temporal_consistency(self, errors_df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure temporal consistency by detecting and handling outliers.
        
        Args:
            errors_df: DataFrame with error columns
            
        Returns:
            DataFrame with temporally consistent errors
        """
        result = errors_df.copy()
        
        # Check for extreme outliers that might indicate data issues
        error_norm = result['gps_error_norm'].values
        
        # Identify outliers using rolling median and threshold
        if len(error_norm) >= self.temporal_consistency_window:
            rolling_median = pd.Series(error_norm).rolling(
                window=self.temporal_consistency_window, 
                center=True, 
                min_periods=1
            ).median()
            
            # Flag points that are significantly different from local median
            outlier_threshold = 3.0  # 3x the local median
            outlier_mask = error_norm > (outlier_threshold * rolling_median)
            
            if outlier_mask.any():
                n_outliers = outlier_mask.sum()
                logger.warning(f"Detected {n_outliers} potential error outliers "
                             f"({100*n_outliers/len(error_norm):.1f}% of data)")
                
                # For extreme outliers, cap at threshold to maintain temporal consistency
                extreme_outliers = error_norm > self.max_error_threshold
                if extreme_outliers.any():
                    logger.warning(f"Capping {extreme_outliers.sum()} errors above "
                                 f"{self.max_error_threshold}m threshold")
                    
                    # Cap error magnitude while preserving direction
                    for idx in np.where(extreme_outliers)[0]:
                        scale_factor = self.max_error_threshold / error_norm[idx]
                        result.iloc[idx, result.columns.get_loc('gps_error_x')] *= scale_factor
                        result.iloc[idx, result.columns.get_loc('gps_error_y')] *= scale_factor
                        result.iloc[idx, result.columns.get_loc('gps_error_z')] *= scale_factor
                        result.iloc[idx, result.columns.get_loc('gps_error_norm')] = self.max_error_threshold
        
        return result
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data has required GPS and ground truth columns.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not super().validate_input(data):
            return False
        
        required_gps_cols = ['gps_x', 'gps_y', 'gps_z']
        required_truth_cols = ['ground_truth_x', 'ground_truth_y', 'ground_truth_z']
        required_cols = required_gps_cols + required_truth_cols
        
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing required columns for error calculation: {missing_cols}")
            return False
        
        # Check for NaN values in critical columns
        for col in required_cols:
            if data[col].isna().any():
                logger.warning(f"Found NaN values in column {col}")
        
        return True
    
    def compute_error_statistics(self, errors_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive error statistics and quality metrics.
        
        Args:
            errors_df: DataFrame with error columns
            
        Returns:
            Dictionary with error statistics and quality metrics
        """
        if errors_df.empty:
            return {}
        
        stats = {}
        
        # Basic statistics for each error component
        for component in ['gps_error_x', 'gps_error_y', 'gps_error_z', 'gps_error_norm']:
            if component in errors_df.columns:
                values = errors_df[component].dropna()
                if len(values) > 0:
                    stats[f'{component}_mean'] = float(values.mean())
                    stats[f'{component}_std'] = float(values.std())
                    stats[f'{component}_min'] = float(values.min())
                    stats[f'{component}_max'] = float(values.max())
                    stats[f'{component}_median'] = float(values.median())
                    stats[f'{component}_q25'] = float(values.quantile(0.25))
                    stats[f'{component}_q75'] = float(values.quantile(0.75))
                    stats[f'{component}_rms'] = float(np.sqrt((values**2).mean()))
        
        # Error distribution analysis
        if 'gps_error_norm' in errors_df.columns:
            error_norm = errors_df['gps_error_norm'].dropna()
            if len(error_norm) > 0:
                # Percentile analysis
                stats['error_p50'] = float(error_norm.quantile(0.50))
                stats['error_p90'] = float(error_norm.quantile(0.90))
                stats['error_p95'] = float(error_norm.quantile(0.95))
                stats['error_p99'] = float(error_norm.quantile(0.99))
                
                # Error bounds analysis
                stats['errors_under_1m'] = float((error_norm < 1.0).mean() * 100)
                stats['errors_under_2m'] = float((error_norm < 2.0).mean() * 100)
                stats['errors_under_5m'] = float((error_norm < 5.0).mean() * 100)
                stats['errors_over_10m'] = float((error_norm > 10.0).mean() * 100)
                
                # Quality metrics
                stats['total_samples'] = len(error_norm)
                stats['valid_samples'] = len(error_norm)
                stats['data_completeness'] = float(len(error_norm) / len(errors_df) * 100)
        
        # Temporal consistency metrics
        stats.update(self._compute_temporal_consistency_metrics(errors_df))
        
        logger.info(f"Computed error statistics: "
                   f"Mean error: {stats.get('gps_error_norm_mean', 0):.3f}m, "
                   f"95th percentile: {stats.get('error_p95', 0):.3f}m")
        
        return stats
    
    def _compute_temporal_consistency_metrics(self, errors_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute metrics related to temporal consistency of errors.
        
        Args:
            errors_df: DataFrame with error columns
            
        Returns:
            Dictionary with temporal consistency metrics
        """
        metrics = {}
        
        if 'gps_error_norm' not in errors_df.columns or len(errors_df) < 2:
            return metrics
        
        error_norm = errors_df['gps_error_norm'].dropna()
        if len(error_norm) < 2:
            return metrics
        
        # Calculate error rate of change (temporal derivative)
        error_diff = error_norm.diff().dropna()
        metrics['error_rate_of_change_mean'] = float(error_diff.abs().mean())
        metrics['error_rate_of_change_std'] = float(error_diff.std())
        metrics['error_rate_of_change_max'] = float(error_diff.abs().max())
        
        # Detect sudden error jumps
        jump_threshold = 3.0 * error_diff.std() if error_diff.std() > 0 else 1.0
        # Only consider significant jumps (> 1m change) to avoid noise sensitivity
        significant_jumps = (error_diff.abs() > max(jump_threshold, 1.0)).sum()
        metrics['sudden_error_jumps'] = int(significant_jumps)
        metrics['sudden_jump_rate'] = float(significant_jumps / len(error_diff) * 100)
        
        # Autocorrelation analysis (simplified)
        if len(error_norm) > 10:
            # Calculate lag-1 autocorrelation
            lag1_corr = error_norm.autocorr(lag=1)
            metrics['error_autocorr_lag1'] = float(lag1_corr) if not np.isnan(lag1_corr) else 0.0
        
        return metrics
    
    def validate_error_quality(self, errors_df: pd.DataFrame, 
                              max_mean_error: float = 5.0,
                              max_p95_error: float = 15.0,
                              min_data_completeness: float = 90.0) -> Tuple[bool, Dict[str, str]]:
        """
        Validate error quality against specified thresholds.
        
        Args:
            errors_df: DataFrame with error columns
            max_mean_error: Maximum acceptable mean error (meters)
            max_p95_error: Maximum acceptable 95th percentile error (meters)
            min_data_completeness: Minimum acceptable data completeness (percentage)
            
        Returns:
            Tuple of (is_valid, validation_messages)
        """
        validation_messages = {}
        is_valid = True
        
        if errors_df.empty:
            validation_messages['empty_data'] = "Error data is empty"
            return False, validation_messages
        
        # Compute statistics for validation
        stats = self.compute_error_statistics(errors_df)
        
        # Check mean error
        mean_error = stats.get('gps_error_norm_mean', float('inf'))
        if mean_error > max_mean_error:
            validation_messages['high_mean_error'] = (
                f"Mean error {mean_error:.3f}m exceeds threshold {max_mean_error}m"
            )
            is_valid = False
        
        # Check 95th percentile error
        p95_error = stats.get('error_p95', float('inf'))
        if p95_error > max_p95_error:
            validation_messages['high_p95_error'] = (
                f"95th percentile error {p95_error:.3f}m exceeds threshold {max_p95_error}m"
            )
            is_valid = False
        
        # Check data completeness
        completeness = stats.get('data_completeness', 0.0)
        if completeness < min_data_completeness:
            validation_messages['low_completeness'] = (
                f"Data completeness {completeness:.1f}% below threshold {min_data_completeness}%"
            )
            is_valid = False
        
        # Check for excessive sudden jumps
        jump_rate = stats.get('sudden_jump_rate', 0.0)
        if jump_rate > 10.0:  # More than 10% sudden jumps (more lenient threshold)
            validation_messages['excessive_jumps'] = (
                f"High rate of sudden error jumps: {jump_rate:.1f}%"
            )
            is_valid = False
        
        # Check for extreme outliers
        max_error = stats.get('gps_error_norm_max', 0.0)
        if max_error > self.max_error_threshold:
            validation_messages['extreme_outliers'] = (
                f"Maximum error {max_error:.3f}m exceeds system threshold {self.max_error_threshold}m"
            )
            is_valid = False
        
        if is_valid:
            validation_messages['status'] = "Error quality validation passed"
        else:
            logger.warning(f"Error quality validation failed: {len(validation_messages)} issues found")
        
        return is_valid, validation_messages