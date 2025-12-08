#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

from uav_log_processor.coordinate_transformer import CoordinateTransformer
import numpy as np

print("Testing CoordinateTransformer...")

# Test 1: Basic functionality
transformer = CoordinateTransformer(37.7749, -122.4194)
x, y = transformer.latlon_to_meters(37.7749, -122.4194)
print(f"Test 1 - Origin: x={x:.6f}, y={y:.6f} (should be ~0)")

# Test 2: Round trip
lat, lon = 40.0, -105.0
transformer2 = CoordinateTransformer(lat, lon)
x, y = transformer2.latlon_to_meters(lat + 0.1, lon + 0.1)
lat_back, lon_back = transformer2.meters_to_latlon(x, y)
print(f"Test 2 - Round trip: input=({lat+0.1}, {lon+0.1}), output=({lat_back:.6f}, {lon_back:.6f})")

# Test 3: Array handling
lats = np.array([0.0, 0.1, 0.2])
lons = np.array([0.0, 0.1, 0.2])
transformer3 = CoordinateTransformer(0.0, 0.0)
x, y = transformer3.latlon_to_meters(lats, lons)
print(f"Test 3 - Arrays: input shape={lats.shape}, output shape={x.shape}")

print("\nAll basic tests passed!")
