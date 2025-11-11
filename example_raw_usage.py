#!/usr/bin/env python3
"""
Example usage of the raw TXT parser.

This demonstrates how to use TxtParserRaw to import all GPS data
without any filtering, then apply your own custom filtering logic.
"""

import pandas as pd
import numpy as np
from uav_log_processor.parsers import TxtParserRaw

def process_raw_gps_data(log_file_path: str):
    """Process GPS data using the raw parser with custom filtering."""
    
    print("Processing GPS data with raw parser...")
    
    # 1. Import all data without filtering
    parser = TxtParserRaw()
    df = parser.parse(log_file_path)
    
    print(f"Total records imported: {len(df)}")
    
    # 2. Examine GPS data
    gps_data = df.dropna(subset=['gps_lat', 'gps_lon'])
    print(f"Records with GPS coordinates: {len(gps_data)}")
    
    if len(gps_data) == 0:
        print("No GPS data found!")
        return df
    
    # 3. Analyze GPS data quality
    print("\nGPS Data Analysis:")
    print(f"Latitude range: {gps_data['gps_lat'].min():.6f} to {gps_data['gps_lat'].max():.6f}")
    print(f"Longitude range: {gps_data['gps_lon'].min():.6f} to {gps_data['gps_lon'].max():.6f}")
    print(f"Altitude range: {gps_data['gps_alt'].min():.2f} to {gps_data['gps_alt'].max():.2f}")
    
    # Check fix types
    fix_types = gps_data['fix_type'].value_counts().sort_index()
    print(f"\nGPS Fix Types:")
    for fix_type, count in fix_types.items():
        if pd.isna(fix_type):
            print(f"  Unknown/NaN: {count}")
        else:
            fix_quality = {0: "No Fix", 1: "Dead Reckoning", 2: "2D Fix", 3: "3D Fix", 4: "DGPS", 5: "RTK Float", 6: "RTK Fixed"}
            quality_name = fix_quality.get(int(fix_type), f"Type {int(fix_type)}")
            print(f"  {quality_name}: {count}")
    
    # 4. Apply custom filtering (example)
    print("\nApplying custom filtering...")
    
    # Example 1: Keep only 3D fix or better
    good_fix = gps_data[gps_data['fix_type'] >= 3]
    print(f"Records with 3D fix or better: {len(good_fix)}")
    
    # Example 2: Remove obvious outliers (global coordinate bounds)
    valid_coords = good_fix[
        (good_fix['gps_lat'] >= -90) & (good_fix['gps_lat'] <= 90) &
        (good_fix['gps_lon'] >= -180) & (good_fix['gps_lon'] <= 180)
    ]
    print(f"Records with valid global coordinates: {len(valid_coords)}")
    
    # Example 3: Detect coordinate clusters (find main flight area)
    if len(valid_coords) > 10:
        # Use median and MAD (Median Absolute Deviation) for robust outlier detection
        lat_median = valid_coords['gps_lat'].median()
        lon_median = valid_coords['gps_lon'].median()
        
        lat_mad = np.median(np.abs(valid_coords['gps_lat'] - lat_median))
        lon_mad = np.median(np.abs(valid_coords['gps_lon'] - lon_median))
        
        # Define reasonable bounds (e.g., 3 MAD from median)
        lat_bounds = (lat_median - 3*lat_mad, lat_median + 3*lat_mad)
        lon_bounds = (lon_median - 3*lon_mad, lon_median + 3*lon_mad)
        
        print(f"\nDetected flight area:")
        print(f"Latitude: {lat_bounds[0]:.6f} to {lat_bounds[1]:.6f}")
        print(f"Longitude: {lon_bounds[0]:.6f} to {lon_bounds[1]:.6f}")
        
        # Filter to main flight area
        main_area = valid_coords[
            (valid_coords['gps_lat'] >= lat_bounds[0]) & (valid_coords['gps_lat'] <= lat_bounds[1]) &
            (valid_coords['gps_lon'] >= lon_bounds[0]) & (valid_coords['gps_lon'] <= lon_bounds[1])
        ]
        print(f"Records in main flight area: {len(main_area)}")
        
        # 5. Create filtered dataset
        # Mark valid GPS records in the original dataframe
        df['gps_valid'] = False
        df.loc[main_area.index, 'gps_valid'] = True
        
        # Option A: Keep only records with valid GPS
        df_filtered = df[df['gps_valid'] | df[['gps_lat', 'gps_lon']].isna().all(axis=1)]
        
        # Option B: Set invalid GPS coordinates to NaN (preserves other sensor data)
        df_cleaned = df.copy()
        invalid_mask = ~df['gps_valid'] & ~df[['gps_lat', 'gps_lon']].isna().all(axis=1)
        df_cleaned.loc[invalid_mask, ['gps_lat', 'gps_lon', 'gps_alt']] = np.nan
        
        print(f"\nFiltering results:")
        print(f"Original GPS records: {len(gps_data)}")
        print(f"Valid GPS records: {len(main_area)}")
        print(f"Filtered dataset size: {len(df_filtered)}")
        print(f"Cleaned dataset size: {len(df_cleaned)}")
        
        return df_cleaned
    
    return df

def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed data to CSV."""
    df.to_csv(output_path, index=False, float_format='%.15g')
    print(f"\nSaved processed data to: {output_path}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_raw_usage.py <log_file_path> [output_file]")
        print("Example: python example_raw_usage.py 'files/logs/2025-08-04 16-17-25.txt' 'output/processed_raw.csv'")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output/processed_raw.csv"
    
    # Process the data
    processed_df = process_raw_gps_data(log_file)
    
    # Save results
    save_processed_data(processed_df, output_file)
    
    print("\nDone! You now have:")
    print("1. All original data preserved")
    print("2. Custom filtering applied based on your criteria")
    print("3. Full control over the filtering logic")