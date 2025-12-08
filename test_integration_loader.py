#!/usr/bin/env python3
"""Integration test for trajectory loader with multiple files."""

import glob
from uav_log_processor.trajectory_loader import (
    load_trajectory_csv,
    validate_trajectory_data,
    has_rtk_data
)

# Test with multiple CSV files
csv_files = glob.glob("files/cleaned/test/*.csv")[:5]  # Test first 5 files

print("=" * 60)
print("Integration Test: Loading Multiple Trajectory Files")
print("=" * 60)

success_count = 0
fail_count = 0
rtk_count = 0

for csv_path in csv_files:
    print(f"\nProcessing: {csv_path}")
    try:
        # Load the CSV
        df = load_trajectory_csv(csv_path)
        
        # Validate
        is_valid, error_msg = validate_trajectory_data(df)
        if not is_valid:
            print(f"  ✗ Validation failed: {error_msg}")
            fail_count += 1
            continue
        
        # Check for RTK
        has_rtk = has_rtk_data(df)
        if has_rtk:
            rtk_count += 1
            print(f"  ✓ Loaded {len(df)} rows (with RTK)")
        else:
            print(f"  ✓ Loaded {len(df)} rows (no RTK)")
        
        success_count += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        fail_count += 1

print("\n" + "=" * 60)
print(f"Results: {success_count} succeeded, {fail_count} failed")
print(f"Files with RTK data: {rtk_count}/{success_count}")
print("=" * 60)

if fail_count == 0:
    print("\n✓ All integration tests passed!")
else:
    print(f"\n✗ {fail_count} files failed to load")
