#!/usr/bin/env python3
"""
Compare filtered vs raw GPS data parsing.

This script demonstrates the difference between the filtered TxtParser 
and the raw TxtParserRaw for importing GPS coordinates.
"""

import pandas as pd
from uav_log_processor.parsers import TxtParser, TxtParserRaw

def compare_parsers(log_file_path: str):
    """Compare filtered vs raw parsing results."""
    
    print("=" * 60)
    print("GPS DATA PARSING COMPARISON")
    print("=" * 60)
    
    # Parse with filtered parser (validates GPS coordinates)
    print("\n1. FILTERED PARSER (TxtParser)")
    print("-" * 40)
    
    filtered_parser = TxtParser()
    filtered_df = filtered_parser.parse(log_file_path)
    filtered_gps = filtered_df.dropna(subset=['gps_lat', 'gps_lon'])
    
    print(f"Total records: {len(filtered_df)}")
    print(f"GPS records: {len(filtered_gps)}")
    if len(filtered_gps) > 0:
        print(f"Lat range: {filtered_gps['gps_lat'].min():.6f} to {filtered_gps['gps_lat'].max():.6f}")
        print(f"Lon range: {filtered_gps['gps_lon'].min():.6f} to {filtered_gps['gps_lon'].max():.6f}")
        print(f"Alt range: {filtered_gps['gps_alt'].min():.2f} to {filtered_gps['gps_alt'].max():.2f}")
        
        print("\nSample coordinates:")
        print(filtered_gps[['timestamp', 'gps_lat', 'gps_lon', 'gps_alt', 'fix_type']].head())
    
    # Parse with raw parser (imports all GPS data)
    print("\n\n2. RAW PARSER (TxtParserRaw)")
    print("-" * 40)
    
    raw_parser = TxtParserRaw()
    raw_df = raw_parser.parse(log_file_path)
    raw_gps = raw_df.dropna(subset=['gps_lat', 'gps_lon'])
    
    print(f"Total records: {len(raw_df)}")
    print(f"GPS records: {len(raw_gps)}")
    if len(raw_gps) > 0:
        print(f"Lat range: {raw_gps['gps_lat'].min():.6f} to {raw_gps['gps_lat'].max():.6f}")
        print(f"Lon range: {raw_gps['gps_lon'].min():.6f} to {raw_gps['gps_lon'].max():.6f}")
        print(f"Alt range: {raw_gps['gps_alt'].min():.2f} to {raw_gps['gps_alt'].max():.2f}")
        
        print("\nSample coordinates:")
        print(raw_gps[['timestamp', 'gps_lat', 'gps_lon', 'gps_alt', 'fix_type']].head())
        
        # Show invalid coordinates that were filtered out
        invalid_coords = raw_gps[
            (raw_gps['gps_lat'] < -90) | (raw_gps['gps_lat'] > 90) |
            (raw_gps['gps_lon'] < -180) | (raw_gps['gps_lon'] > 180) |
            (raw_gps['fix_type'] < 3)
        ]
        
        if len(invalid_coords) > 0:
            print(f"\nInvalid coordinates found: {len(invalid_coords)}")
            print("Sample invalid coordinates:")
            print(invalid_coords[['timestamp', 'gps_lat', 'gps_lon', 'gps_alt', 'fix_type']].head())
    
    # Summary
    print("\n\n3. SUMMARY")
    print("-" * 40)
    print(f"Filtered GPS records: {len(filtered_gps)}")
    print(f"Raw GPS records: {len(raw_gps)}")
    print(f"Filtered out: {len(raw_gps) - len(filtered_gps)} records")
    
    if len(filtered_gps) > 0 and len(raw_gps) > 0:
        print(f"Data retention: {len(filtered_gps)/len(raw_gps)*100:.1f}%")
    
    print("\nUse Cases:")
    print("- TxtParser (filtered): Production ML datasets, clean analysis")
    print("- TxtParserRaw (raw): Data exploration, debugging, manual filtering")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python compare_parsers.py <log_file_path>")
        print("Example: python compare_parsers.py 'files/logs/2025-08-04 16-17-25.txt'")
        sys.exit(1)
    
    log_file = sys.argv[1]
    compare_parsers(log_file)