#!/usr/bin/env python3
"""
Simple test script for CoordinateTransformer.
Tests basic functionality without requiring a testing framework.
"""

import numpy as np
import sys
from uav_log_processor.coordinate_transformer import CoordinateTransformer


def test_basic_conversion():
    """Test basic lat/lon to meters conversion."""
    print("Test 1: Basic conversion...")
    
    # Create transformer at a known location (roughly San Francisco)
    origin_lat, origin_lon = 37.7749, -122.4194
    transformer = CoordinateTransformer(origin_lat, origin_lon)
    
    # Test conversion of origin point (should be 0, 0)
    x, y = transformer.latlon_to_meters(origin_lat, origin_lon)
    assert abs(x) < 1e-6, f"Origin x should be ~0, got {x}"
    assert abs(y) < 1e-6, f"Origin y should be ~0, got {y}"
    print("  ✓ Origin point converts to (0, 0)")
    
    # Test a point 1 degree north (should be ~110km north)
    x, y = transformer.latlon_to_meters(origin_lat + 1.0, origin_lon)
    assert abs(x) < 1e-6, f"Point directly north should have x~0, got {x}"
    assert 110000 < y < 111000, f"Point 1° north should be ~110km, got {y}"
    print(f"  ✓ Point 1° north is {y:.1f}m away")
    
    # Test a point 1 degree east (should be ~85km east at this latitude)
    x, y = transformer.latlon_to_meters(origin_lat, origin_lon + 1.0)
    assert 80000 < x < 90000, f"Point 1° east should be ~85km, got {x}"
    assert abs(y) < 1e-6, f"Point directly east should have y~0, got {y}"
    print(f"  ✓ Point 1° east is {x:.1f}m away")


def test_round_trip():
    """Test that converting to meters and back preserves coordinates."""
    print("\nTest 2: Round-trip conversion...")
    
    origin_lat, origin_lon = 40.0, -105.0
    transformer = CoordinateTransformer(origin_lat, origin_lon)
    
    # Test with various points
    test_points = [
        (40.0, -105.0),  # Origin
        (40.1, -105.0),  # North
        (40.0, -104.9),  # East
        (39.9, -105.1),  # Southwest
        (40.05, -104.95),  # Northeast
    ]
    
    for lat, lon in test_points:
        # Convert to meters and back
        x, y = transformer.latlon_to_meters(lat, lon)
        lat_back, lon_back = transformer.meters_to_latlon(x, y)
        
        # Check that we get back the original coordinates (within tolerance)
        lat_error = abs(lat - lat_back)
        lon_error = abs(lon - lon_back)
        
        assert lat_error < 1e-9, f"Lat round-trip error too large: {lat_error}"
        assert lon_error < 1e-9, f"Lon round-trip error too large: {lon_error}"
    
    print(f"  ✓ All {len(test_points)} points round-trip correctly")


def test_array_input():
    """Test that the transformer handles numpy arrays."""
    print("\nTest 3: Array input...")
    
    origin_lat, origin_lon = 0.0, 0.0  # Equator
    transformer = CoordinateTransformer(origin_lat, origin_lon)
    
    # Create arrays of coordinates
    lats = np.array([0.0, 0.1, 0.2, 0.3])
    lons = np.array([0.0, 0.1, 0.2, 0.3])
    
    # Convert to meters
    x, y = transformer.latlon_to_meters(lats, lons)
    
    # Check that we get arrays back
    assert isinstance(x, np.ndarray), "Should return numpy array"
    assert isinstance(y, np.ndarray), "Should return numpy array"
    assert len(x) == len(lats), "Output should have same length as input"
    assert len(y) == len(lats), "Output should have same length as input"
    
    print(f"  ✓ Array input/output works correctly")


def test_edge_cases():
    """Test edge cases like equator and high latitudes."""
    print("\nTest 4: Edge cases...")
    
    # Test at equator
    transformer_eq = CoordinateTransformer(0.0, 0.0)
    x, y = transformer_eq.latlon_to_meters(0.0, 1.0)
    # At equator, 1 degree longitude ≈ 111km
    assert 110000 < x < 112000, f"At equator, 1° lon should be ~111km, got {x}"
    print("  ✓ Equator conversion works")
    
    # Test at high latitude (but not too close to pole)
    transformer_high = CoordinateTransformer(80.0, 0.0)
    x, y = transformer_high.latlon_to_meters(80.0, 1.0)
    # At 80° latitude, longitude lines are much closer
    assert 0 < x < 30000, f"At 80° lat, 1° lon should be ~19km, got {x}"
    print("  ✓ High latitude conversion works")
    
    # Test near pole (should not crash)
    try:
        transformer_pole = CoordinateTransformer(89.5, 0.0)
        x, y = transformer_pole.latlon_to_meters(89.5, 1.0)
        print("  ✓ Near-pole conversion works without crashing")
    except Exception as e:
        print(f"  ✗ Near-pole conversion failed: {e}")
        raise


def test_validation():
    """Test input validation."""
    print("\nTest 5: Input validation...")
    
    # Test invalid latitude
    try:
        CoordinateTransformer(91.0, 0.0)
        assert False, "Should raise ValueError for lat > 90"
    except ValueError:
        print("  ✓ Rejects latitude > 90")
    
    try:
        CoordinateTransformer(-91.0, 0.0)
        assert False, "Should raise ValueError for lat < -90"
    except ValueError:
        print("  ✓ Rejects latitude < -90")
    
    # Test invalid longitude
    try:
        CoordinateTransformer(0.0, 181.0)
        assert False, "Should raise ValueError for lon > 180"
    except ValueError:
        print("  ✓ Rejects longitude > 180")
    
    try:
        CoordinateTransformer(0.0, -181.0)
        assert False, "Should raise ValueError for lon < -180"
    except ValueError:
        print("  ✓ Rejects longitude < -180")


def test_date_line_wraparound():
    """Test handling of date line crossing."""
    print("\nTest 6: Date line wraparound...")
    
    # Create transformer near date line
    transformer = CoordinateTransformer(0.0, 179.0)
    
    # Test point across date line
    x1, y1 = transformer.latlon_to_meters(0.0, -179.0)
    # These points are only 2 degrees apart (not 358 degrees)
    # So distance should be ~222km, not ~39,800km
    assert abs(x1) < 300000, f"Date line crossing should be handled, got {x1}m"
    print(f"  ✓ Date line crossing handled correctly ({x1:.1f}m)")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CoordinateTransformer")
    print("=" * 60)
    
    try:
        test_basic_conversion()
        test_round_trip()
        test_array_input()
        test_edge_cases()
        test_validation()
        test_date_line_wraparound()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        sys.exit(0)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
