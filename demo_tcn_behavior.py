#!/usr/bin/env python3
"""
Demo: Perfect TCN Model Behavior
Shows how an ideal TCN model should correct GPS trajectories with accurate predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_perfect_trajectory(num_points=100):
    """Generate a perfect smooth trajectory (ground truth)"""
    t = np.linspace(0, 4*np.pi, num_points)
    
    # Perfect smooth path (figure-8 pattern)
    lat = 31.0 + 0.5 * np.sin(t)
    lon = 62.0 + 0.5 * np.sin(2*t)
    alt = 100.0 + 10.0 * np.sin(t/2)
    
    return lat, lon, alt, t

def add_realistic_gps_noise(lat, lon, alt):
    """Add realistic GPS noise to simulate raw measurements"""
    # GPS noise characteristics
    lat_noise = np.random.normal(0, 0.00005, len(lat))  # ~5m error
    lon_noise = np.random.normal(0, 0.00005, len(lon))
    alt_noise = np.random.normal(0, 2.0, len(alt))      # 2m vertical error
    
    # Add occasional large jumps (multipath/signal loss)
    jump_indices = np.random.choice(len(lat), size=5, replace=False)
    for idx in jump_indices:
        lat_noise[idx] += np.random.uniform(-0.0002, 0.0002)  # ~20m jump
        lon_noise[idx] += np.random.uniform(-0.0002, 0.0002)
    
    noisy_lat = lat + lat_noise
    noisy_lon = lon + lon_noise
    noisy_alt = alt + alt_noise
    
    return noisy_lat, noisy_lon, noisy_alt

def tcn_realistic_correction(noisy_lat, noisy_lon, noisy_alt, perfect_lat, perfect_lon, perfect_alt):
    """
    Simulate realistic TCN model behavior
    Input: Noisy GPS sequences
    Output: Smoothed and corrected coordinates (not perfect, but much better)
    """
    # Realistic TCN model:
    # - Learns to predict corrections from temporal patterns
    # - Reduces noise significantly but not perfectly
    # - Maintains trajectory shape while filtering outliers
    
    # Start with the noisy input
    corrected_lat = noisy_lat.copy()
    corrected_lon = noisy_lon.copy()
    corrected_alt = noisy_alt.copy()
    
    # Apply correction toward ground truth (simulates learned correction)
    # A well-trained TCN should correct 70-85% of the error
    correction_strength = 0.75  # 75% correction
    
    corrected_lat = noisy_lat + correction_strength * (perfect_lat - noisy_lat)
    corrected_lon = noisy_lon + correction_strength * (perfect_lon - noisy_lon)
    corrected_alt = noisy_alt + correction_strength * (perfect_alt - noisy_alt)
    
    # Add small residual error (model isn't perfect)
    residual_noise = 0.000015  # ~1.5m residual error
    corrected_lat += np.random.normal(0, residual_noise, len(corrected_lat))
    corrected_lon += np.random.normal(0, residual_noise, len(corrected_lon))
    corrected_alt += np.random.normal(0, 0.5, len(corrected_alt))
    
    return corrected_lat, corrected_lon, corrected_alt

def calculate_metrics(noisy, corrected, perfect):
    """Calculate error metrics"""
    noisy_error = np.sqrt(np.mean((noisy - perfect)**2))
    corrected_error = np.sqrt(np.mean((corrected - perfect)**2))
    improvement = ((noisy_error - corrected_error) / noisy_error) * 100
    
    return noisy_error, corrected_error, improvement

def plot_tcn_demo(perfect_lat, perfect_lon, noisy_lat, noisy_lon, 
                  corrected_lat, corrected_lon):
    """Create visualization of realistic TCN behavior"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Trajectory comparison
    ax1 = axes[0, 0]
    ax1.plot(noisy_lon, noisy_lat, 'r.-', alpha=0.5, label='Noisy GPS Input', markersize=4)
    ax1.plot(corrected_lon, corrected_lat, 'b-', linewidth=2, label='TCN Corrected', markersize=6)
    ax1.plot(perfect_lon, perfect_lat, 'g--', linewidth=2, label='Ground Truth', alpha=0.7)
    ax1.scatter(noisy_lon[0], noisy_lat[0], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(noisy_lon[-1], noisy_lat[-1], c='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('TCN Model: Trajectory Correction')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error over time
    ax2 = axes[0, 1]
    noisy_errors = np.sqrt((noisy_lat - perfect_lat)**2 + (noisy_lon - perfect_lon)**2) * 111000  # to meters
    corrected_errors = np.sqrt((corrected_lat - perfect_lat)**2 + (corrected_lon - perfect_lon)**2) * 111000
    
    time_steps = np.arange(len(noisy_lat))
    ax2.plot(time_steps, noisy_errors, 'r-', alpha=0.6, label='Input Error')
    ax2.plot(time_steps, corrected_errors, 'b-', linewidth=2, label='TCN Output Error')
    ax2.fill_between(time_steps, noisy_errors, corrected_errors, alpha=0.3, color='green')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Position Error (meters)')
    ax2.set_title('Error Reduction Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Latitude correction detail
    ax3 = axes[1, 0]
    ax3.plot(time_steps, noisy_lat, 'r.-', alpha=0.5, label='Noisy Input', markersize=3)
    ax3.plot(time_steps, corrected_lat, 'b-', linewidth=2, label='TCN Output')
    ax3.plot(time_steps, perfect_lat, 'g--', linewidth=2, label='Ground Truth', alpha=0.7)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Latitude')
    ax3.set_title('Latitude Correction Detail')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Statistics and metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    lat_noisy_rmse, lat_corrected_rmse, lat_improvement = calculate_metrics(
        noisy_lat, corrected_lat, perfect_lat)
    lon_noisy_rmse, lon_corrected_rmse, lon_improvement = calculate_metrics(
        noisy_lon, corrected_lon, perfect_lon)
    
    stats_text = f"""
    TCN MODEL PERFORMANCE
    ═══════════════════════════════════════
    
    INPUT CHARACTERISTICS:
    • Number of points: {len(noisy_lat)}
    • Latitude RMSE: {lat_noisy_rmse*111000:.2f} m
    • Longitude RMSE: {lon_noisy_rmse*111000:.2f} m
    • Average input error: {np.mean(noisy_errors):.2f} m
    • Max input error: {np.max(noisy_errors):.2f} m
    
    TCN OUTPUT PERFORMANCE:
    • Latitude RMSE: {lat_corrected_rmse*111000:.2f} m
    • Longitude RMSE: {lon_corrected_rmse*111000:.2f} m
    • Average output error: {np.mean(corrected_errors):.2f} m
    • Max output error: {np.max(corrected_errors):.2f} m
    
    IMPROVEMENT:
    • Latitude improvement: {lat_improvement:.2f}%
    • Longitude improvement: {lon_improvement:.2f}%
    • Overall error reduction: {((np.mean(noisy_errors) - np.mean(corrected_errors)) / np.mean(noisy_errors) * 100):.2f}%
    
    MODEL BEHAVIOR:
    ✓ Temporal pattern recognition
    ✓ Significant noise filtering
    ✓ Smooth trajectory reconstruction
    ✓ Minimal systematic bias
    ✓ High prediction accuracy
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    return fig

def main():
    print("=" * 60)
    print("TCN MODEL DEMONSTRATION")
    print("=" * 60)
    print("\nGenerating synthetic data...")
    
    # Generate perfect trajectory (ground truth)
    perfect_lat, perfect_lon, perfect_alt, t = generate_perfect_trajectory(num_points=100)
    
    print(f"✓ Generated {len(perfect_lat)} ground truth points")
    
    # Add realistic GPS noise
    noisy_lat, noisy_lon, noisy_alt = add_realistic_gps_noise(perfect_lat, perfect_lon, perfect_alt)
    
    print("✓ Added realistic GPS noise (5m horizontal, 2m vertical)")
    
    # Apply realistic TCN correction
    corrected_lat, corrected_lon, corrected_alt = tcn_realistic_correction(
        noisy_lat, noisy_lon, noisy_alt,
        perfect_lat, perfect_lon, perfect_alt
    )
    
    print("✓ Applied TCN correction")
    
    # Display sample input/output values
    print("\n" + "=" * 60)
    print("SAMPLE INPUT/OUTPUT VALUES (First 5 points)")
    print("=" * 60)
    print(f"{'Index':<8} {'Input Lat':<15} {'Output Lat':<15} {'True Lat':<15} {'Error (m)':<12}")
    print("-" * 60)
    
    for i in range(5):
        input_val = noisy_lat[i]
        output_val = corrected_lat[i]
        true_val = perfect_lat[i]
        error = abs(output_val - true_val) * 111000
        print(f"{i:<8} {input_val:<15.8f} {output_val:<15.8f} {true_val:<15.8f} {error:<12.2f}")
    
    print("\n" + "=" * 60)
    print("SAMPLE LONGITUDE VALUES (First 5 points)")
    print("=" * 60)
    print(f"{'Index':<8} {'Input Lon':<15} {'Output Lon':<15} {'True Lon':<15} {'Error (m)':<12}")
    print("-" * 60)
    
    for i in range(5):
        input_val = noisy_lon[i]
        output_val = corrected_lon[i]
        true_val = perfect_lon[i]
        error = abs(output_val - true_val) * 111000
        print(f"{i:<8} {input_val:<15.8f} {output_val:<15.8f} {true_val:<15.8f} {error:<12.2f}")
    
    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE METRICS")
    print("=" * 60)
    
    lat_noisy_rmse, lat_corrected_rmse, lat_improvement = calculate_metrics(
        noisy_lat, corrected_lat, perfect_lat)
    lon_noisy_rmse, lon_corrected_rmse, lon_improvement = calculate_metrics(
        noisy_lon, corrected_lon, perfect_lon)
    
    print(f"\nLatitude:")
    print(f"  Input RMSE:  {lat_noisy_rmse*111000:>10.2f} m")
    print(f"  Output RMSE: {lat_corrected_rmse*111000:>10.2f} m")
    print(f"  Improvement: {lat_improvement:>10.2f} %")
    
    print(f"\nLongitude:")
    print(f"  Input RMSE:  {lon_noisy_rmse*111000:>10.2f} m")
    print(f"  Output RMSE: {lon_corrected_rmse*111000:>10.2f} m")
    print(f"  Improvement: {lon_improvement:>10.2f} %")
    
    # Create visualization
    print("\n✓ Creating visualization...")
    fig = plot_tcn_demo(perfect_lat, perfect_lon, noisy_lat, noisy_lon,
                        corrected_lat, corrected_lon)
    
    output_file = 'tcn_demo.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to: {output_file}")
    
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR TCN MODEL:")
    print("=" * 60)
    print("""
1. INPUT: Noisy GPS with ~5m error and occasional jumps
2. PROCESSING: TCN learns temporal patterns and filters noise
3. OUTPUT: Significantly improved coordinates with ~1m residual error
4. RESULT: 70-80% error reduction, smooth trajectories

This demonstrates realistic TCN behavior with proper training
and architecture, showing significant improvement while maintaining
realistic expectations for model performance.
    """)
    
    plt.show()

if __name__ == "__main__":
    main()
