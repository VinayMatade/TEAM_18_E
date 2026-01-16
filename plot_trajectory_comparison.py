#!/usr/bin/env python3
"""
Trajectory Visualization Script
Plots raw GPS coordinates, runs the model for correction, and compares both trajectories.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import joblib
from pathlib import Path

# Configuration
MODEL_PATH = "models/best_model(15.44, phy=0.03).pth"
INPUT_CSV = "files/cleaned/test/2025-08-04 17-03-35_cleaned.csv"
SEQ_LEN = 125  # Must match training configuration
REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']

# Device configuration
device_env = os.getenv("DEVICE", "auto")
if device_env == "auto":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(device_env)

print(f"Using device: {DEVICE}")

# Model Architecture (Must match train.py exactly)
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2, dilations=None):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilations[i] if dilations else 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.net(x)
        return self.linear(y[:, :, -1])

def load_model():
    """Load the trained model"""
    print(f"Loading model from {MODEL_PATH}...")
    
    # Create model with same architecture as training
    model = TCN(
        input_size=10,  # GPS(2) + Delta(2) + IMU(6) = 10
        output_size=2,
        num_channels=[128, 128, 128, 128, 128, 128],
        kernel_size=7,
        dropout=0.3,
        dilations=[1, 2, 4, 8, 16, 32]
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    return model

def load_scalers():
    """Load the fitted scalers"""
    print("Loading scalers...")
    try:
        scalers = joblib.load("scalers.save")
        return scalers
    except FileNotFoundError:
        print("Warning: scalers.save not found. Model predictions may be inaccurate.")
        return None

def plot_raw_trajectory(df):
    """Plot the raw GPS trajectory"""
    print("Plotting raw GPS trajectory...")
    
    plt.figure(figsize=(12, 10))
    
    # Plot raw trajectory
    plt.subplot(2, 2, 1)
    plt.plot(df['GPS_Lng'], df['GPS_Lat'], 'r-', alpha=0.7, linewidth=1, label='Raw GPS')
    plt.scatter(df['GPS_Lng'].iloc[0], df['GPS_Lat'].iloc[0], color='green', s=100, marker='o', label='Start', zorder=5)
    plt.scatter(df['GPS_Lng'].iloc[-1], df['GPS_Lat'].iloc[-1], color='red', s=100, marker='s', label='End', zorder=5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Raw GPS Trajectory')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot GPS accuracy over time
    plt.subplot(2, 2, 2)
    if 'GPA_HAcc' in df.columns:
        plt.plot(df.index, df['GPA_HAcc'], 'b-', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Horizontal Accuracy (m)')
        plt.title('GPS Horizontal Accuracy')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('GPS Accuracy (Not Available)')
    
    # Plot number of satellites
    plt.subplot(2, 2, 3)
    if 'GPS_NSats' in df.columns:
        plt.plot(df.index, df['GPS_NSats'], 'g-', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Number of Satellites')
        plt.title('GPS Satellite Count')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No satellite data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Satellite Count (Not Available)')
    
    # Plot speed
    plt.subplot(2, 2, 4)
    if 'GPS_Spd' in df.columns:
        plt.plot(df.index, df['GPS_Spd'], 'm-', alpha=0.7)
        plt.xlabel('Sample Index')
        plt.ylabel('Speed (m/s)')
        plt.title('GPS Speed')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No speed data available', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Speed (Not Available)')
    
    plt.tight_layout()
    plt.savefig('raw_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Raw trajectory analysis saved to raw_trajectory_analysis.png")
    plt.show()

def run_model_inference(df, model, scalers):
    """Run model inference to get corrected coordinates"""
    print("Running model inference...")
    
    # Prepare data
    data = df[REQUIRED_COLS].values.astype(np.float32)
    
    # Filter out invalid GPS points (null island filter)
    valid_mask = (data[:, 0] != 0) & (data[:, 1] != 0) & ~np.isnan(data).any(axis=1)
    data = data[valid_mask]
    
    if len(data) < SEQ_LEN:
        raise ValueError(f"Not enough valid data points. Need at least {SEQ_LEN}, got {len(data)}")
    
    # Add horizontal accuracy if available
    if 'GPA_HAcc' in df.columns:
        hacc = df['GPA_HAcc'].values[valid_mask].astype(np.float32)
    else:
        hacc = np.ones(len(data), dtype=np.float32)  # Default 1m accuracy
    
    predictions = []
    
    with torch.no_grad():
        for i in range(len(data) - SEQ_LEN + 1):
            # Extract window
            window = data[i:i + SEQ_LEN].copy()
            hacc_window = hacc[i:i + SEQ_LEN]
            
            # Store original coordinates for reference
            start_lat, start_lon = window[0, 0], window[0, 1]
            
            # Convert to meters (relative to start point)
            lat_rad = np.radians(start_lat)
            m_per_deg_lat = 110649.0
            m_per_deg_lon = 111132.0 * np.cos(lat_rad)
            
            # Convert GPS to relative meters
            gps_m = window[:, 0:2].copy()
            gps_m[:, 0] = (gps_m[:, 0] - start_lat) * m_per_deg_lat
            gps_m[:, 1] = (gps_m[:, 1] - start_lon) * m_per_deg_lon
            
            # Prepare input features
            if scalers is not None:
                # Normalize GPS coordinates
                gps_norm = scalers['gps_point'].transform(gps_m).astype(np.float32)
                
                # Calculate and normalize deltas
                delta_gps = np.diff(gps_m, axis=0, prepend=gps_m[:1])
                delta_norm = scalers['delta'].transform(delta_gps).astype(np.float32)
                
                # Normalize IMU data
                imu_norm = scalers['imu'].transform(window[:, 2:8]).astype(np.float32)
                
                # Combine features: GPS(2) + Delta(2) + IMU(6) = 10
                features = np.concatenate([gps_norm, delta_norm, imu_norm], axis=1)
            else:
                # Without scalers, use raw data (less accurate)
                delta_gps = np.diff(gps_m, axis=0, prepend=gps_m[:1])
                features = np.concatenate([gps_m, delta_gps, window[:, 2:8]], axis=1)
            
            # Convert to tensor and predict
            x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            pred_norm = model(x_tensor).cpu().numpy()[0]
            
            # Denormalize prediction if scalers available
            if scalers is not None:
                pred_m = scalers['target'].inverse_transform(pred_norm.reshape(1, -1))[0]
            else:
                pred_m = pred_norm
            
            # Convert back to absolute coordinates
            pred_lat = start_lat + pred_m[0] / m_per_deg_lat
            pred_lon = start_lon + pred_m[1] / m_per_deg_lon
            
            predictions.append([pred_lat, pred_lon])
    
    return np.array(predictions), valid_mask

def plot_comparison(df, predictions, valid_mask):
    """Plot comparison between raw and corrected trajectories"""
    print("Creating comparison plot...")
    
    # Filter original data to match predictions
    df_valid = df[valid_mask].reset_index(drop=True)
    
    plt.figure(figsize=(15, 10))
    
    # Main trajectory comparison
    plt.subplot(2, 2, 1)
    plt.plot(df_valid['GPS_Lng'], df_valid['GPS_Lat'], 'r-', alpha=0.6, linewidth=2, label='Raw GPS', zorder=1)
    
    # Only plot predictions where we have them (starting from SEQ_LEN)
    pred_start_idx = SEQ_LEN - 1
    plt.plot(predictions[:, 1], predictions[:, 0], 'b-', linewidth=2, label='AI Corrected', zorder=2)
    
    # Mark start and end points
    plt.scatter(df_valid['GPS_Lng'].iloc[0], df_valid['GPS_Lat'].iloc[0], 
                color='green', s=150, marker='o', label='Start', zorder=5, edgecolor='black')
    plt.scatter(df_valid['GPS_Lng'].iloc[-1], df_valid['GPS_Lat'].iloc[-1], 
                color='red', s=150, marker='s', label='End', zorder=5, edgecolor='black')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GPS Trajectory Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Zoomed view of a section
    plt.subplot(2, 2, 2)
    mid_idx = len(df_valid) // 2
    zoom_range = min(100, len(df_valid) // 4)
    start_zoom = max(0, mid_idx - zoom_range)
    end_zoom = min(len(df_valid), mid_idx + zoom_range)
    
    plt.plot(df_valid['GPS_Lng'].iloc[start_zoom:end_zoom], 
             df_valid['GPS_Lat'].iloc[start_zoom:end_zoom], 
             'r-', alpha=0.6, linewidth=2, label='Raw GPS')
    
    # Adjust prediction indices for zoom
    pred_zoom_start = max(0, start_zoom - pred_start_idx)
    pred_zoom_end = min(len(predictions), end_zoom - pred_start_idx)
    
    if pred_zoom_end > pred_zoom_start:
        plt.plot(predictions[pred_zoom_start:pred_zoom_end, 1], 
                 predictions[pred_zoom_start:pred_zoom_end, 0], 
                 'b-', linewidth=2, label='AI Corrected')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Zoomed View (Middle Section)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Error analysis (if we had ground truth, we'd calculate actual errors)
    plt.subplot(2, 2, 3)
    # Calculate displacement between consecutive points for both trajectories
    raw_displacements = []
    corrected_displacements = []
    
    for i in range(1, len(df_valid)):
        # Raw displacement
        lat1, lon1 = df_valid['GPS_Lat'].iloc[i-1], df_valid['GPS_Lng'].iloc[i-1]
        lat2, lon2 = df_valid['GPS_Lat'].iloc[i], df_valid['GPS_Lng'].iloc[i]
        
        # Approximate distance in meters
        dlat = (lat2 - lat1) * 110649.0
        dlon = (lon2 - lon1) * 111132.0 * np.cos(np.radians(lat1))
        raw_dist = np.sqrt(dlat**2 + dlon**2)
        raw_displacements.append(raw_dist)
    
    # For corrected trajectory
    for i in range(1, len(predictions)):
        lat1, lon1 = predictions[i-1, 0], predictions[i-1, 1]
        lat2, lon2 = predictions[i, 0], predictions[i, 1]
        
        dlat = (lat2 - lat1) * 110649.0
        dlon = (lon2 - lon1) * 111132.0 * np.cos(np.radians(lat1))
        corr_dist = np.sqrt(dlat**2 + dlon**2)
        corrected_displacements.append(corr_dist)
    
    plt.plot(raw_displacements, 'r-', alpha=0.7, label='Raw GPS Step Size')
    if corrected_displacements:
        # Align the arrays
        min_len = min(len(raw_displacements), len(corrected_displacements))
        plt.plot(corrected_displacements[:min_len], 'b-', alpha=0.7, label='Corrected Step Size')
    
    plt.xlabel('Step Index')
    plt.ylabel('Distance (m)')
    plt.title('Step Size Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics
    plt.subplot(2, 2, 4)
    stats_text = f"""
    Trajectory Statistics:
    
    Raw GPS Points: {len(df_valid)}
    Corrected Points: {len(predictions)}
    
    Raw GPS Stats:
    - Lat Range: {df_valid['GPS_Lat'].min():.6f} to {df_valid['GPS_Lat'].max():.6f}
    - Lon Range: {df_valid['GPS_Lng'].min():.6f} to {df_valid['GPS_Lng'].max():.6f}
    - Avg Step Size: {np.mean(raw_displacements):.2f} m
    
    Corrected GPS Stats:
    - Lat Range: {predictions[:, 0].min():.6f} to {predictions[:, 0].max():.6f}
    - Lon Range: {predictions[:, 1].min():.6f} to {predictions[:, 1].max():.6f}
    """
    
    if corrected_displacements:
        stats_text += f"- Avg Step Size: {np.mean(corrected_displacements):.2f} m"
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontfamily='monospace', fontsize=9)
    plt.axis('off')
    plt.title('Statistics')
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Trajectory comparison saved to trajectory_comparison.png")
    plt.show()

def save_corrected_data(df, predictions, valid_mask, output_file="corrected_trajectory.csv"):
    """Save the corrected trajectory data"""
    print(f"Saving corrected data to {output_file}...")
    
    # Create a copy of the original dataframe
    df_output = df.copy()
    
    # Initialize corrected columns with NaN
    df_output['Corrected_Lat'] = np.nan
    df_output['Corrected_Lng'] = np.nan
    
    # Fill in the corrected values where we have predictions
    valid_indices = df.index[valid_mask].tolist()
    pred_start_idx = SEQ_LEN - 1
    
    for i, pred in enumerate(predictions):
        if pred_start_idx + i < len(valid_indices):
            original_idx = valid_indices[pred_start_idx + i]
            df_output.loc[original_idx, 'Corrected_Lat'] = pred[0]
            df_output.loc[original_idx, 'Corrected_Lng'] = pred[1]
    
    df_output.to_csv(output_file, index=False)
    print(f"✅ Corrected data saved to {output_file}")

def main():
    """Main execution function"""
    print("🚁 UAV Trajectory Analysis and Correction")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Input file not found: {INPUT_CSV}")
        return
    
    # Load data
    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    print(f"✅ Loaded {len(df)} data points")
    
    # Check required columns
    missing_cols = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return
    
    # Step 1: Plot raw trajectory
    plot_raw_trajectory(df)
    
    # Step 2: Load model and scalers
    try:
        model = load_model()
        scalers = load_scalers()
        
        # Step 3: Run inference
        predictions, valid_mask = run_model_inference(df, model, scalers)
        print(f"✅ Generated {len(predictions)} corrected coordinates")
        
        # Step 4: Plot comparison
        plot_comparison(df, predictions, valid_mask)
        
        # Step 5: Save corrected data
        save_corrected_data(df, predictions, valid_mask)
        
        print("\n🎉 Analysis complete!")
        print("Generated files:")
        print("  - raw_trajectory_analysis.png: Raw GPS analysis")
        print("  - trajectory_comparison.png: Before/after comparison")
        print("  - corrected_trajectory.csv: Corrected coordinates")
        
    except Exception as e:
        print(f"❌ Error during model inference: {e}")
        print("Showing only raw trajectory analysis.")

if __name__ == "__main__":
    main()
