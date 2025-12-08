#!/usr/bin/env python3
"""
Demonstration of CoordinateTransformer functionality.
"""

import sys
sys.path.insert(0, '.')

from uav_log_processor.coordinate_transformer import CoordinateTransformer
import numpy as np

def main():
    print("=" * 70)
    print("CoordinateTransformer Demonstration")
    print("=" * 70)
    
    # Example 1: Basic usage
    print("\n1. Basic Usage - San Francisco")
    print("-" * 70)
    origin_lat, origin_lon = 37.7749, -122.4194
    transformer = CoordinateTransformer(origin_lat, origin_lon)
    
    # Convert origin (should be 0, 0)
    x, y = transformer.latlon_to_meters(origin_lat, origin_lon)
    print(f"   Origin ({origin_lat}, {origin_lon}) -> ({x:.2f}m, {y:.2f}m)")
    
    # Convert a point 0.01 degrees north
    test_lat, test_lon = origin_lat + 0.01, origin_lon
    x, y = transformer.latlon_to_meters(test_lat, test_lon)
    print(f"   Point 0.01° north -> ({x:.2f}m, {y:.2f}m)")
    
    # Example 2: Round-trip conversion
    print("\n2. Round-Trip Conversion")
    print("-" * 70)
    test_lat, test_lon = 37.8, -122.3
    x, y = transformer.latlon_to_meters(test_lat, test_lon)
    lat_back, lon_back = transformer.meters_to_latlon(x, y)
    print(f"   Original: ({test_lat}, {test_lon})")
    print(f"   To meters: ({x:.2f}m, {y:.2f}m)")
    print(f"   Back to lat/lon: ({lat_back:.6f}, {lon_back:.6f})")
    print(f"   Error: {abs(test_lat - lat_back):.10f}° lat, {abs(test_lon - lon_back):.10f}° lon")
    
    # Example 3: Array processing
    print("\n3. Array Processing")
    print("-" * 70)
    lats = np.array([37.77, 37.78, 37.79, 37.80])
    lons = np.array([-122.42, -122.41, -122.40, -122.39])
    x_arr, y_arr = transformer.latlon_to_meters(lats, lons)
    print(f"   Input: {len(lats)} coordinate pairs")
    print(f"   Output: {len(x_arr)} meter pairs")
    for i in range(len(lats)):
        print(f"     ({lats[i]:.2f}, {lons[i]:.2f}) -> ({x_arr[i]:.1f}m, {y_arr[i]:.1f}m)")
    
    # Example 4: Edge cases
    print("\n4. Edge Cases")
    print("-" * 70)
    
    # Equator
    eq_transformer = CoordinateTransformer(0.0, 0.0)
    x, y = eq_transformer.latlon_to_meters(0.0, 1.0)
    print(f"   Equator: 1° longitude = {x:.1f}m")
    
    # High latitude
    high_transformer = CoordinateTransformer(80.0, 0.0)
    x, y = high_transformer.latlon_to_meters(80.0, 1.0)
    print(f"   80° latitude: 1° longitude = {x:.1f}m")
    
    # Near pole
    pole_transformer = CoordinateTransformer(89.5, 0.0)
    x, y = pole_transformer.latlon_to_meters(89.5, 1.0)
    print(f"   89.5° latitude: 1° longitude = {x:.1f}m")
    
    print("\n" + "=" * 70)
    print("✓ All demonstrations completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
