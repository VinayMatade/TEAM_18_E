#!/usr/bin/env python3
"""Manual test of trajectory loader with real data."""

from uav_log_processor.trajectory_loader import (
    load_trajectory_csv,
    validate_trajectory_data,
    has_rtk_data,
    extract_gps_trajectory,
    extract_rtk_trajectory,
    extract_imu_data
)

# Test with a real CSV file
csv_path = "files/cleaned/test/00000001_cleaned.csv"

print("=" * 60)
print("Testing trajectory loader with real data")
print("=" * 60)

# Load the CSV
print(f"\n1. Loading CSV: {csv_path}")
df = load_trajectory_csv(csv_path)
print(f"   Loaded {len(df)} rows")
print(f"   Columns: {list(df.columns)}")

# Validate the data
print("\n2. Validating trajectory data")
is_valid, error_msg = validate_trajectory_data(df)
if is_valid:
    print("   ✓ Data is valid")
else:
    print(f"   ✗ Data is invalid: {error_msg}")

# Check for RTK data
print("\n3. Checking for RTK data")
if has_rtk_data(df):
    print("   ✓ RTK data is available")
else:
    print("   ✗ No RTK data available")

# Extract GPS trajectory
print("\n4. Extracting GPS trajectory")
lat, lon = extract_gps_trajectory(df)
print(f"   GPS trajectory: {len(lat)} points")
print(f"   Latitude range: [{lat.min():.6f}, {lat.max():.6f}]")
print(f"   Longitude range: [{lon.min():.6f}, {lon.max():.6f}]")

# Extract RTK trajectory if available
print("\n5. Extracting RTK trajectory")
rtk_result = extract_rtk_trajectory(df)
if rtk_result is not None:
    rtk_lat, rtk_lon = rtk_result
    print(f"   RTK trajectory: {len(rtk_lat)} points")
    print(f"   RTK Latitude range: [{rtk_lat.min():.6f}, {rtk_lat.max():.6f}]")
    print(f"   RTK Longitude range: [{rtk_lon.min():.6f}, {rtk_lon.max():.6f}]")
else:
    print("   No RTK data to extract")

# Extract IMU data
print("\n6. Extracting IMU data")
acc, gyr = extract_imu_data(df)
print(f"   Accelerometer data: {acc.shape}")
print(f"   Gyroscope data: {gyr.shape}")
print(f"   Acc X range: [{acc[:, 0].min():.3f}, {acc[:, 0].max():.3f}]")
print(f"   Acc Y range: [{acc[:, 1].min():.3f}, {acc[:, 1].max():.3f}]")
print(f"   Acc Z range: [{acc[:, 2].min():.3f}, {acc[:, 2].max():.3f}]")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
