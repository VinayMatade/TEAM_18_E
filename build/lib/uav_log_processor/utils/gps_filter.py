"""
GPS reliability filtering utilities.

Provides functions for filtering GPS data based on fix type, HDOP, and other quality metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GPSReliabilityFilter:
    """Filter GPS data based on reliability metrics."""
    
    def __init__(self, min_fix_type: int = 3, max_hdop: float = 5.0, max_vdop: float = 10.0):
        """
        Initialize GPS reliability filter.
        
        Args:
            min_fix_type: Minimum GPS fix type to accept (3 = 3D fix)
            max_hdop: Maximum horizontal dilution of precision to accept
            max_vdop: Maximum vertical dilution of precision to accept
        """
        self.min_fix_type = min_fix_type
        self.max_hdop = max_hdop
        self.max_vdop = max_vdop
    
    def filter_gps_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter GPS data based on reliability criteria.
        
        Args:
            df: DataFrame with GPS data including fix_type, hdop, vdop columns
            
        Returns:
            Filtered DataFrame with only reliable GPS data
        """
        if df.empty:
            return df
        
        initial_count = len(df)
        
        # Apply fix type filter
        if 'fix_type' in df.columns:
            df = df[df['fix_type'] >= self.min_fix_type]
            logger.debug(f"After fix_type filter: {len(df)}/{initial_count} samples")
        
        # Apply HDOP filter
        if 'hdop' in df.columns:
            df = df[df['hdop'] <= self.max_hdop]
            logger.debug(f"After HDOP filter: {len(df)}/{initial_count} samples")
        
        # Apply VDOP filter
        if 'vdop' in df.columns:
            df = df[df['vdop'] <= self.max_vdop]
            logger.debug(f"After VDOP filter: {len(df)}/{initial_count} samples")
        
        filtered_count = len(df)
        filter_ratio = (initial_count - filtered_count) / initial_count if initial_count > 0 else 0
        
        if filter_ratio > 0.1:  # Warn if more than 10% filtered
            logger.warning(f"GPS reliability filter removed {filter_ratio:.1%} of data "
                         f"({initial_count - filtered_count}/{initial_count} samples)")
        
        return df.reset_index(drop=True)
    
    def select_best_gps_source(self, gps_sources: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
        """
        Select the best GPS source from multiple available sources.
        
        Args:
            gps_sources: Dictionary mapping source names to GPS DataFrames
            
        Returns:
            Tuple of (best_source_name, best_gps_dataframe)
        """
        if not gps_sources:
            raise ValueError("No GPS sources provided")
        
        if len(gps_sources) == 1:
            source_name, gps_data = next(iter(gps_sources.items()))
            filtered_data = self.filter_gps_data(gps_data)
            if filtered_data.empty:
                raise ValueError("No GPS sources have reliable data")
            return source_name, filtered_data
        
        # Evaluate each GPS source
        source_scores = {}
        
        for source_name, gps_data in gps_sources.items():
            if gps_data.empty:
                source_scores[source_name] = -1
                continue
            
            # Filter data first
            filtered_data = self.filter_gps_data(gps_data)
            
            if filtered_data.empty:
                source_scores[source_name] = -1
                continue
            
            # Calculate quality score
            score = self._calculate_gps_quality_score(filtered_data)
            source_scores[source_name] = score
            
            logger.debug(f"GPS source '{source_name}': score={score:.3f}, "
                        f"samples={len(filtered_data)}")
        
        # Select best source
        valid_sources = {k: v for k, v in source_scores.items() if v >= 0}
        
        if not valid_sources:
            raise ValueError("No GPS sources have reliable data")
        
        best_source = max(valid_sources.keys(), key=lambda k: source_scores[k])
        
        logger.info(f"Selected GPS source '{best_source}' with quality score "
                   f"{source_scores[best_source]:.3f}")
        
        return best_source, self.filter_gps_data(gps_sources[best_source])
    
    def _calculate_gps_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate a quality score for GPS data.
        
        Args:
            df: GPS DataFrame with quality metrics
            
        Returns:
            Quality score (higher is better)
        """
        if df.empty:
            return -1.0
        
        score = 0.0
        
        # Data availability score (0-40 points)
        data_availability = len(df) / max(len(df), 1000)  # Normalize to expected sample count
        score += min(data_availability * 40, 40)
        
        # Fix type score (0-20 points)
        if 'fix_type' in df.columns:
            avg_fix_type = df['fix_type'].mean()
            fix_score = min((avg_fix_type - 2) * 10, 20)  # 3D fix = 10 points, RTK = 20 points
            score += max(fix_score, 0)
        
        # HDOP score (0-20 points) - lower HDOP is better
        if 'hdop' in df.columns:
            avg_hdop = df['hdop'].mean()
            hdop_score = max(20 - avg_hdop * 4, 0)  # HDOP of 1 = 16 points, HDOP of 5 = 0 points
            score += hdop_score
        
        # VDOP score (0-10 points) - lower VDOP is better
        if 'vdop' in df.columns:
            avg_vdop = df['vdop'].mean()
            vdop_score = max(10 - avg_vdop * 2, 0)  # VDOP of 1 = 8 points, VDOP of 5 = 0 points
            score += vdop_score
        
        # Consistency score (0-10 points) - less variation in position is better
        if all(col in df.columns for col in ['gps_lat', 'gps_lon']):
            lat_std = df['gps_lat'].std()
            lon_std = df['gps_lon'].std()
            # Convert to approximate meters (rough approximation)
            lat_std_m = lat_std * 111000  # degrees to meters
            lon_std_m = lon_std * 111000 * np.cos(np.radians(df['gps_lat'].mean()))
            position_std = np.sqrt(lat_std_m**2 + lon_std_m**2)
            consistency_score = max(10 - position_std / 10, 0)  # 10m std = 9 points, 100m std = 0 points
            score += consistency_score
        
        return score
    
    def get_gps_quality_report(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate a quality report for GPS data.
        
        Args:
            df: GPS DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        if df.empty:
            return {
                'total_samples': 0,
                'reliable_samples': 0,
                'reliability_ratio': 0.0,
                'avg_hdop': np.nan,
                'avg_vdop': np.nan,
                'avg_fix_type': np.nan,
                'position_std_m': np.nan
            }
        
        # Filter data to get reliable samples
        reliable_df = self.filter_gps_data(df)
        
        # Calculate position standard deviation in meters
        position_std_m = np.nan
        if all(col in df.columns for col in ['gps_lat', 'gps_lon']) and not df.empty:
            lat_std = df['gps_lat'].std()
            lon_std = df['gps_lon'].std()
            if not (np.isnan(lat_std) or np.isnan(lon_std)):
                lat_std_m = lat_std * 111000
                lon_std_m = lon_std * 111000 * np.cos(np.radians(df['gps_lat'].mean()))
                position_std_m = np.sqrt(lat_std_m**2 + lon_std_m**2)
        
        return {
            'total_samples': len(df),
            'reliable_samples': len(reliable_df),
            'reliability_ratio': len(reliable_df) / len(df) if len(df) > 0 else 0.0,
            'avg_hdop': df['hdop'].mean() if 'hdop' in df.columns else np.nan,
            'avg_vdop': df['vdop'].mean() if 'vdop' in df.columns else np.nan,
            'avg_fix_type': df['fix_type'].mean() if 'fix_type' in df.columns else np.nan,
            'position_std_m': position_std_m
        }


def prioritize_gps_units(gps_data_dict: Dict[str, pd.DataFrame], 
                        config: Optional[Dict] = None) -> List[Tuple[str, pd.DataFrame]]:
    """
    Prioritize multiple GPS units based on reliability metrics.
    
    Args:
        gps_data_dict: Dictionary mapping GPS unit names to their data
        config: Optional configuration for filtering parameters
        
    Returns:
        List of (unit_name, filtered_data) tuples ordered by priority (best first)
    """
    if not gps_data_dict:
        return []
    
    # Initialize filter with config parameters
    filter_config = config or {}
    gps_filter = GPSReliabilityFilter(
        min_fix_type=filter_config.get('min_gps_fix_type', 3),
        max_hdop=filter_config.get('max_hdop', 5.0),
        max_vdop=filter_config.get('max_vdop', 10.0)
    )
    
    # Calculate quality scores for each GPS unit
    unit_scores = []
    
    for unit_name, gps_data in gps_data_dict.items():
        if gps_data.empty:
            continue
        
        filtered_data = gps_filter.filter_gps_data(gps_data)
        
        if filtered_data.empty:
            continue
        
        quality_score = gps_filter._calculate_gps_quality_score(filtered_data)
        unit_scores.append((quality_score, unit_name, filtered_data))
    
    # Sort by quality score (descending)
    unit_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return prioritized list
    return [(name, data) for _, name, data in unit_scores]


def detect_gps_outages(df: pd.DataFrame, max_gap_seconds: float = 5.0) -> List[Tuple[float, float]]:
    """
    Detect GPS data outages (gaps in reliable data).
    
    Args:
        df: GPS DataFrame with timestamp column
        max_gap_seconds: Maximum gap to consider as an outage
        
    Returns:
        List of (start_time, end_time) tuples for detected outages
    """
    if df.empty or 'timestamp' not in df.columns:
        return []
    
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Calculate time differences
    time_diffs = df_sorted['timestamp'].diff()
    
    # Find gaps larger than threshold
    gap_mask = time_diffs > max_gap_seconds
    gap_indices = df_sorted.index[gap_mask].tolist()
    
    outages = []
    for gap_idx in gap_indices:
        # Find the position in the sorted dataframe
        sorted_positions = df_sorted.index.tolist()
        sorted_idx = sorted_positions.index(gap_idx)
        
        if sorted_idx > 0:
            start_time = df_sorted.iloc[sorted_idx - 1]['timestamp']
            end_time = df_sorted.iloc[sorted_idx]['timestamp']
            outages.append((start_time, end_time))
    
    return outages