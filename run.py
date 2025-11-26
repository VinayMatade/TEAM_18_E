#!/usr/bin/env python3
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import torch.nn as nn

# --- CONFIGURATION ---
MODEL_PATH = "best_model.pth"
INPUT_CSV = "files/cleaned/test/2025-08-04 17-03-35_cleaned.csv" # Pick a BAD log
OUTPUT_CSV = "corrected_flight.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS (Must match training)
SEQ_LEN = 50
REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']

# --- MODEL ARCHITECTURE (Must be identical to train.py) ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
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
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1)*dilation_size, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.net(x)
        return self.linear(y[:, :, -1])

# --- INFERENCE LOGIC ---
def run_inference():
    print(f"Loading model from {MODEL_PATH}...")
    model = TCN(input_size=9, output_size=2, num_channels=[64, 128, 64], kernel_size=3, dropout=0.0)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"Loading flight log: {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Prepare Data
    # Note: We do NOT add noise here. We use the raw noisy data from the file.
    data = df[REQUIRED_COLS].values.astype(np.float32)
    
    # We also need 'HAcc' to feed the model. 
    # If the log has it, use it. If not, estimate it.
    if 'GPA_HAcc' in df.columns:
        hacc = df['GPA_HAcc'].values.astype(np.float32)
    else:
        # Fallback: Assume 1.0m accuracy if missing
        hacc = np.ones(len(data), dtype=np.float32)
        
    # Combine into (Lat, Lng, HAcc, IMU...)
    # [Lat, Lng] are cols 0,1. IMU are 2..7
    full_input = np.column_stack([data[:, 0:2], hacc, data[:, 2:8]])
    
    predictions = []
    
    print("Fixing flight path...")
    with torch.no_grad():
        # We need a sliding window of SEQ_LEN (50)
        for i in range(len(full_input) - SEQ_LEN):
            window = full_input[i : i + SEQ_LEN].copy()
            
            # 1. CENTER THE WINDOW (Crucial!)
            start_lat = window[0, 0]
            start_lon = window[0, 1]
            
            lat_rad = np.radians(start_lat)
            m_per_deg_lat = 110649.0
            m_per_deg_lon = 111132.0 * np.cos(lat_rad)
            
            # Relative Degrees
            window[:, 0] -= start_lat
            window[:, 1] -= start_lon
            
            # Meters
            window[:, 0] *= m_per_deg_lat
            window[:, 1] *= m_per_deg_lon
            
            # Convert to Tensor
            x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            # Predict Correction (in Meters Relative to Start)
            pred_m = model(x_tensor).cpu().numpy()[0]
            
            # Convert back to Absolute Degrees
            pred_deg_lat = pred_m[0] / m_per_deg_lat
            pred_deg_lon = pred_m[1] / m_per_deg_lon
            
            final_lat = start_lat + pred_deg_lat
            final_lon = start_lon + pred_deg_lon
            
            predictions.append([final_lat, final_lon])
            
    # Convert to DataFrame (Pad the start with NaNs or original values)
    # Because we lose the first 50 points due to windowing
    pad = np.full((SEQ_LEN, 2), np.nan)
    pred_array = np.vstack([pad, np.array(predictions)])
    
    df['Corrected_Lat'] = pred_array[:, 0]
    df['Corrected_Lng'] = pred_array[:, 1]
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved corrected log to {OUTPUT_CSV}")
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 10))
    plt.plot(df['GPS_Lng'], df['GPS_Lat'], label='Original (Noisy)', alpha=0.5, color='red')
    plt.plot(df['Corrected_Lng'], df['Corrected_Lat'], label='AI Corrected', linewidth=2, color='green')
    plt.title(f"GPS Correction Result\n{os.path.basename(INPUT_CSV)}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig("result_plot.png")
    print("📈 Plot saved to result_plot.png")

if __name__ == "__main__":
    run_inference()
