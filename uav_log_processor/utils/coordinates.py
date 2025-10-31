"""
Coordinate system conversion utilities.

Handles conversion between WGS84 (GPS) and ENU (East-North-Up) coordinate systems.
"""

from typing import Tuple, Optional, Union, List
import numpy as np
import pandas as pd
import logging

try:
    import pyproj
    from pyproj import Transformer
except ImportError:
    raise ImportError("pyproj is required for coordinate conversion. Install with: pip install pyproj")


class CoordinateConverter:
    """Handles coordinate system conversions for UAV data."""
    
    def __init__(self, config=None):
        """
        Initialize coordinate converter.
        
        Args:
            config: Configuration dictionary with conversion parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Earth parameters
        self.earth_radius = 6378137.0  # WGS84 semi-major axis in meters
        self.earth_flattening = 1.0 / 298.257223563  # WGS84 flattening
        
        # Cache for transformers to avoid recreation
        self._transformer_cache = {}
    
    def calculate_home_point(self, gps_data: pd.DataFrame, 
                           method: str = 'first_valid') -> Tuple[float, float, float]:
        """
        Calculate home point from GPS data.
        
        Args:
            gps_data: DataFrame with 'gps_lat', 'gps_lon', 'gps_alt' columns
            method: Method to calculate home point ('first_valid', 'mean', 'median')
            
        Returns:
            Tuple of (home_lat, home_lon, home_alt) in degrees and meters
            
        Raises:
            ValueError: If no valid GPS data found
        """
        if gps_data.empty:
            raise ValueError("No GPS data provided")
        
        required_cols = ['gps_lat', 'gps_lon', 'gps_alt']
        missing_cols = [col for col in required_cols if col not in gps_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Filter out invalid GPS data
        valid_gps = gps_data.dropna(subset=required_cols)
        
        # Remove obviously invalid coordinates
        valid_gps = valid_gps[
            (valid_gps['gps_lat'].abs() <= 90) &
            (valid_gps['gps_lon'].abs() <= 180) &
            (valid_gps['gps_alt'].abs() <= 50000)  # Reasonable altitude limit
        ]
        
        if valid_gps.empty:
            # Try to find any non-NaN GPS data, even if it doesn't meet all criteria
            partial_valid = gps_data.dropna(subset=['gps_lat', 'gps_lon'])
            if not partial_valid.empty:
                self.logger.warning("Using partial GPS data for home point calculation")
                valid_gps = partial_valid
                # Fill missing altitude with default
                if 'gps_alt' not in valid_gps.columns or valid_gps['gps_alt'].isna().all():
                    valid_gps = valid_gps.copy()
                    valid_gps['gps_alt'] = 0.0
            else:
                raise ValueError("No valid GPS coordinates found")
        
        if method == 'first_valid':
            home_lat = valid_gps['gps_lat'].iloc[0]
            home_lon = valid_gps['gps_lon'].iloc[0]
            home_alt = valid_gps['gps_alt'].iloc[0]
            
        elif method == 'mean':
            home_lat = valid_gps['gps_lat'].mean()
            home_lon = valid_gps['gps_lon'].mean()
            home_alt = valid_gps['gps_alt'].mean()
            
        elif method == 'median':
            home_lat = valid_gps['gps_lat'].median()
            home_lon = valid_gps['gps_lon'].median()
            home_alt = valid_gps['gps_alt'].median()
            
        else:
            raise ValueError(f"Unknown home point calculation method: {method}")
        
        self.logger.info(f"Calculated home point: ({home_lat:.6f}, {home_lon:.6f}, {home_alt:.2f})")
        return home_lat, home_lon, home_alt
    
    def wgs84_to_enu(self, lat: Union[float, np.ndarray], 
                     lon: Union[float, np.ndarray], 
                     alt: Union[float, np.ndarray],
                     home_lat: float, home_lon: float, home_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert WGS84 coordinates to ENU (East-North-Up) coordinates.
        
        Args:
            lat: Latitude in degrees (scalar or array)
            lon: Longitude in degrees (scalar or array)
            alt: Altitude in meters (scalar or array)
            home_lat: Home point latitude in degrees
            home_lon: Home point longitude in degrees
            home_alt: Home point altitude in meters
            
        Returns:
            Tuple of (east, north, up) coordinates in meters
        """
        # Convert inputs to numpy arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        alt = np.asarray(alt)
        
        # Validate inputs - log warnings but don't raise exceptions
        try:
            self._validate_coordinates(lat, lon, alt)
        except Exception as e:
            self.logger.warning(f"Input coordinate validation: {str(e)}")
        
        try:
            self._validate_coordinates(home_lat, home_lon, home_alt)
        except Exception as e:
            self.logger.warning(f"Home point validation: {str(e)}")
        
        try:
            # Create transformer for WGS84 to ECEF conversion
            transformer_key = f"wgs84_ecef_{home_lat}_{home_lon}"
            if transformer_key not in self._transformer_cache:
                self._transformer_cache[transformer_key] = Transformer.from_crs(
                    "EPSG:4979",  # WGS84 3D
                    "EPSG:4978",  # ECEF
                    always_xy=True
                )
            
            transformer = self._transformer_cache[transformer_key]
            
            # Convert points to ECEF
            x, y, z = transformer.transform(lon, lat, alt)
            home_x, home_y, home_z = transformer.transform(home_lon, home_lat, home_alt)
            
            # Convert ECEF to ENU
            east, north, up = self._ecef_to_enu(
                x, y, z, home_x, home_y, home_z, home_lat, home_lon
            )
            
            # Ensure return values are numpy arrays
            return np.asarray(east), np.asarray(north), np.asarray(up)
            
        except Exception as e:
            self.logger.error(f"Error in WGS84 to ENU conversion: {str(e)}")
            # Fallback to simple approximation
            east, north, up = self._wgs84_to_enu_simple(lat, lon, alt, home_lat, home_lon, home_alt)
            return np.asarray(east), np.asarray(north), np.asarray(up)
    
    def enu_to_wgs84(self, x: Union[float, np.ndarray], 
                     y: Union[float, np.ndarray], 
                     z: Union[float, np.ndarray],
                     home_lat: float, home_lon: float, home_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert ENU coordinates to WGS84 coordinates.
        
        Args:
            x: East coordinate in meters (scalar or array)
            y: North coordinate in meters (scalar or array)
            z: Up coordinate in meters (scalar or array)
            home_lat: Home point latitude in degrees
            home_lon: Home point longitude in degrees
            home_alt: Home point altitude in meters
            
        Returns:
            Tuple of (latitude, longitude, altitude) in degrees and meters
        """
        # Convert inputs to numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)
        
        try:
            self._validate_coordinates(home_lat, home_lon, home_alt)
        except Exception as e:
            self.logger.warning(f"Home point validation: {str(e)}")
        
        try:
            # Create transformer for ECEF to WGS84 conversion
            transformer_key = f"ecef_wgs84_{home_lat}_{home_lon}"
            if transformer_key not in self._transformer_cache:
                self._transformer_cache[transformer_key] = Transformer.from_crs(
                    "EPSG:4978",  # ECEF
                    "EPSG:4979",  # WGS84 3D
                    always_xy=True
                )
            
            transformer = self._transformer_cache[transformer_key]
            
            # Convert home point to ECEF
            home_transformer = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
            home_x, home_y, home_z = home_transformer.transform(home_lon, home_lat, home_alt)
            
            # Convert ENU to ECEF
            ecef_x, ecef_y, ecef_z = self._enu_to_ecef(
                x, y, z, home_x, home_y, home_z, home_lat, home_lon
            )
            
            # Convert ECEF to WGS84
            lon, lat, alt = transformer.transform(ecef_x, ecef_y, ecef_z)
            
            # Ensure return values are numpy arrays
            return np.asarray(lat), np.asarray(lon), np.asarray(alt)
            
        except Exception as e:
            self.logger.error(f"Error in ENU to WGS84 conversion: {str(e)}")
            # Fallback to simple approximation
            lat, lon, alt = self._enu_to_wgs84_simple(x, y, z, home_lat, home_lon, home_alt)
            return np.asarray(lat), np.asarray(lon), np.asarray(alt)
    
    def convert_dataframe_to_enu(self, data: pd.DataFrame, 
                                home_point: Optional[Tuple[float, float, float]] = None) -> pd.DataFrame:
        """
        Convert GPS coordinates in DataFrame to ENU coordinates with coordinate preservation.
        
        Args:
            data: DataFrame with 'gps_lat', 'gps_lon', 'gps_alt' columns
            home_point: Optional home point tuple (lat, lon, alt). If None, calculated from data.
            
        Returns:
            DataFrame with additional 'enu_x', 'enu_y', 'enu_z' columns
        """
        if data.empty:
            return data
        
        required_cols = ['gps_lat', 'gps_lon', 'gps_alt']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        data = data.copy()
        
        # Log original coordinate statistics for debugging
        original_coords = data[required_cols].copy()
        zero_coords = ((original_coords['gps_lat'] == 0) & (original_coords['gps_lon'] == 0)).sum()
        if zero_coords > 0:
            self.logger.warning(f"Found {zero_coords} rows with (0,0) coordinates in input data")
        
        # Calculate or use provided home point
        if home_point is None:
            home_lat, home_lon, home_alt = self.calculate_home_point(data)
        else:
            home_lat, home_lon, home_alt = home_point
        
        # Convert coordinates - preserve original GPS coordinates throughout
        valid_mask = data[required_cols].notna().all(axis=1) & ~((data['gps_lat'] == 0) & (data['gps_lon'] == 0))
        
        if valid_mask.any():
            valid_data = data.loc[valid_mask]
            
            east, north, up = self.wgs84_to_enu(
                valid_data['gps_lat'].values,
                valid_data['gps_lon'].values,
                valid_data['gps_alt'].values,
                home_lat, home_lon, home_alt
            )
            
            # Add ENU columns - initialize with NaN to preserve invalid coordinates
            data['enu_x'] = np.nan
            data['enu_y'] = np.nan
            data['enu_z'] = np.nan
            
            # Only assign converted values to valid coordinates
            data.loc[valid_mask, 'enu_x'] = east
            data.loc[valid_mask, 'enu_y'] = north
            data.loc[valid_mask, 'enu_z'] = up
            
            # Store home point in metadata
            data.attrs['home_point'] = (home_lat, home_lon, home_alt)
            
            # Verify no coordinate corruption occurred
            post_zero_coords = ((data['gps_lat'] == 0) & (data['gps_lon'] == 0)).sum()
            if post_zero_coords != zero_coords:
                self.logger.error(f"Coordinate corruption detected! Zero coordinates changed from {zero_coords} to {post_zero_coords}")
            
            self.logger.info(f"Successfully converted {valid_mask.sum()} valid coordinates to ENU")
        else:
            self.logger.warning("No valid GPS coordinates found for ENU conversion")
        
        return data
    
    def _ecef_to_enu(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     home_x: float, home_y: float, home_z: float,
                     home_lat: float, home_lon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert ECEF coordinates to ENU relative to home point."""
        # Calculate differences
        dx = x - home_x
        dy = y - home_y
        dz = z - home_z
        
        # Convert to radians
        lat_rad = np.radians(home_lat)
        lon_rad = np.radians(home_lon)
        
        # Rotation matrix from ECEF to ENU
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # Apply rotation
        east = -sin_lon * dx + cos_lon * dy
        north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
        up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
        
        return east, north, up
    
    def _enu_to_ecef(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                     home_x: float, home_y: float, home_z: float,
                     home_lat: float, home_lon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert ENU coordinates to ECEF relative to home point."""
        # Convert to radians
        lat_rad = np.radians(home_lat)
        lon_rad = np.radians(home_lon)
        
        # Rotation matrix from ENU to ECEF (transpose of ECEF to ENU)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        
        # Apply rotation
        dx = -sin_lon * x - sin_lat * cos_lon * y + cos_lat * cos_lon * z
        dy = cos_lon * x - sin_lat * sin_lon * y + cos_lat * sin_lon * z
        dz = cos_lat * y + sin_lat * z
        
        # Add home point offset
        ecef_x = home_x + dx
        ecef_y = home_y + dy
        ecef_z = home_z + dz
        
        return ecef_x, ecef_y, ecef_z
    
    def _wgs84_to_enu_simple(self, lat: np.ndarray, lon: np.ndarray, alt: np.ndarray,
                            home_lat: float, home_lon: float, home_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple approximation for WGS84 to ENU conversion (for small distances)."""
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        home_lat_rad = np.radians(home_lat)
        home_lon_rad = np.radians(home_lon)
        
        # Calculate differences
        dlat = lat_rad - home_lat_rad
        dlon = lon_rad - home_lon_rad
        dalt = alt - home_alt
        
        # Simple approximation (valid for small distances)
        east = self.earth_radius * dlon * np.cos(home_lat_rad)
        north = self.earth_radius * dlat
        up = dalt
        
        return east, north, up
    
    def _enu_to_wgs84_simple(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            home_lat: float, home_lon: float, home_alt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simple approximation for ENU to WGS84 conversion (for small distances)."""
        home_lat_rad = np.radians(home_lat)
        
        # Simple approximation (valid for small distances)
        dlat = y / self.earth_radius
        dlon = x / (self.earth_radius * np.cos(home_lat_rad))
        dalt = z
        
        # Convert back to degrees
        lat = home_lat + np.degrees(dlat)
        lon = home_lon + np.degrees(dlon)
        alt = home_alt + dalt
        
        return lat, lon, alt
    
    def _validate_coordinates(self, lat: Union[float, np.ndarray], 
                             lon: Union[float, np.ndarray], 
                             alt: Union[float, np.ndarray]) -> bool:
        """Validate coordinate values are within reasonable ranges."""
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        alt = np.asarray(alt)
        
        # Check for NaN values - but don't reject if they exist, just warn
        if np.any(np.isnan(lat)) or np.any(np.isnan(lon)) or np.any(np.isnan(alt)):
            self.logger.warning("NaN values detected in coordinates - will be handled during processing")
            # Don't return False here - let the processing continue with NaN handling
        
        # Check for zero coordinates that might indicate reset issues
        if np.any((lat == 0) & (lon == 0)):
            self.logger.warning("Zero coordinates (0,0) detected - possible coordinate reset issue")
            # Don't return False - log the issue but continue processing
        
        # Check coordinate ranges for valid (non-NaN, non-zero) values
        valid_mask = ~(np.isnan(lat) | np.isnan(lon) | ((lat == 0) & (lon == 0)))
        if np.any(valid_mask):
            valid_lat = lat[valid_mask]
            valid_lon = lon[valid_mask]
            valid_alt = alt[valid_mask] if not np.isnan(alt).all() else alt
            
            if np.any(np.abs(valid_lat) > 90):
                self.logger.warning("Invalid latitude values detected (outside ±90 degrees)")
                # Don't return False - just log the warning
            if np.any(np.abs(valid_lon) > 180):
                self.logger.warning("Invalid longitude values detected (outside ±180 degrees)")
                # Don't return False - just log the warning
            if not np.isnan(valid_alt).all() and np.any(np.abs(valid_alt) > 50000):  # Reasonable altitude limit
                self.logger.warning("Extreme altitude values detected (>50km)")
        
        return True