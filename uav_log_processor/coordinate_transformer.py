"""
Coordinate transformation utilities for converting between geographic coordinates
(latitude/longitude) and local Cartesian coordinates (meters).

Uses a local tangent plane approximation centered at an origin point.
"""

import numpy as np
from typing import Tuple, Union


class CoordinateTransformer:
    """
    Transforms coordinates between geographic (lat/lon) and local Cartesian (meters).
    
    Uses a local tangent plane approximation where:
    - Origin: A reference point (lat0, lon0)
    - X-axis: East direction (longitude)
    - Y-axis: North direction (latitude)
    
    The conversion accounts for Earth's curvature by using latitude-dependent
    scaling factors.
    """
    
    # Earth radius constants (meters)
    # These are approximate values for WGS84 ellipsoid
    EARTH_RADIUS_EQUATORIAL = 6378137.0  # meters at equator
    EARTH_RADIUS_POLAR = 6356752.3142  # meters at poles
    
    # Approximate conversion factors (meters per degree)
    # These are refined in the constructor based on actual latitude
    METERS_PER_DEGREE_LAT = 110649.0  # Approximately constant
    METERS_PER_DEGREE_LON_AT_EQUATOR = 111132.0  # Maximum at equator
    
    def __init__(self, origin_lat: float, origin_lon: float):
        """
        Initialize the coordinate transformer with an origin point.
        
        Args:
            origin_lat: Origin latitude in degrees (-90 to 90)
            origin_lon: Origin longitude in degrees (-180 to 180)
            
        Raises:
            ValueError: If latitude or longitude are out of valid range
        """
        # Validate inputs
        if not -90 <= origin_lat <= 90:
            raise ValueError(f"Latitude must be between -90 and 90 degrees, got {origin_lat}")
        if not -180 <= origin_lon <= 180:
            raise ValueError(f"Longitude must be between -180 and 180 degrees, got {origin_lon}")
        
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # Compute latitude-dependent scaling factor for longitude
        # At higher latitudes, longitude lines converge, so degrees cover less distance
        lat_rad = np.radians(origin_lat)
        self.meters_per_degree_lon = self.METERS_PER_DEGREE_LON_AT_EQUATOR * np.cos(lat_rad)
        self.meters_per_degree_lat = self.METERS_PER_DEGREE_LAT
        
        # Handle edge case: very high latitudes (near poles)
        # At latitudes > 89 degrees, longitude becomes poorly defined
        if abs(origin_lat) > 89.0:
            # Use a minimum scaling factor to avoid division by zero
            self.meters_per_degree_lon = max(self.meters_per_degree_lon, 1.0)
    
    def latlon_to_meters(
        self, 
        lat: Union[float, np.ndarray], 
        lon: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert latitude/longitude coordinates to local Cartesian coordinates in meters.
        
        Args:
            lat: Latitude in degrees (scalar or array)
            lon: Longitude in degrees (scalar or array)
            
        Returns:
            Tuple of (x, y) coordinates in meters relative to origin
            - x: East-West position (positive = East)
            - y: North-South position (positive = North)
            
        Note:
            This uses a local tangent plane approximation, which is accurate for
            distances up to ~100km from the origin. For larger distances, consider
            using a proper geodetic transformation.
        """
        # Convert to numpy arrays for consistent handling
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        
        # Compute differences from origin
        delta_lat = lat - self.origin_lat
        delta_lon = lon - self.origin_lon
        
        # Handle longitude wraparound (e.g., crossing the date line)
        # If difference is > 180 degrees, we crossed the date line
        delta_lon = np.where(delta_lon > 180, delta_lon - 360, delta_lon)
        delta_lon = np.where(delta_lon < -180, delta_lon + 360, delta_lon)
        
        # Convert to meters using local approximation
        x = delta_lon * self.meters_per_degree_lon
        y = delta_lat * self.meters_per_degree_lat
        
        return x, y
    
    def meters_to_latlon(
        self, 
        x: Union[float, np.ndarray], 
        y: Union[float, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert local Cartesian coordinates (meters) to latitude/longitude.
        
        Args:
            x: East-West position in meters (positive = East)
            y: North-South position in meters (positive = North)
            
        Returns:
            Tuple of (lat, lon) in degrees
            
        Note:
            This is the inverse of latlon_to_meters() and uses the same
            local tangent plane approximation.
        """
        # Convert to numpy arrays for consistent handling
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Convert meters back to degrees
        delta_lon = x / self.meters_per_degree_lon
        delta_lat = y / self.meters_per_degree_lat
        
        # Add to origin to get absolute coordinates
        lon = self.origin_lon + delta_lon
        lat = self.origin_lat + delta_lat
        
        # Normalize longitude to [-180, 180]
        lon = np.where(lon > 180, lon - 360, lon)
        lon = np.where(lon < -180, lon + 360, lon)
        
        # Clamp latitude to valid range [-90, 90]
        # (though in practice, our local approximation shouldn't produce invalid values)
        lat = np.clip(lat, -90, 90)
        
        return lat, lon
