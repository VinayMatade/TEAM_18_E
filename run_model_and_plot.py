import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuration
MODEL_PATH = "models/best_model(15.44, phy=0.03).pth"
INPUT_CSV = "files/cleaned/test/2025-08-04 17-03-35_cleaned.csv"
SEQ_LEN = 125
REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model architecture (from train.py)
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

# Load data
df = pd.read_csv(INPUT_CSV)
data = df[REQUIRED_COLS].values.astype(np.float32)

# Filter valid GPS points
valid_mask = (data[:, 0] != 0) & (data[:, 1] != 0) & ~np.isnan(data).any(axis=1)
data = data[valid_mask]

# Load model
model = TCN(
    input_size=10,
    output_size=2,
    num_channels=[128, 128, 128, 128, 128, 128],
    kernel_size=7,
    dropout=0.3,
    dilations=[1, 2, 4, 8, 16, 32]
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Load scalers
scalers = joblib.load("scalers.save")

# Run inference
predictions = []
with torch.no_grad():
    for i in range(len(data) - SEQ_LEN + 1):
        window = data[i:i + SEQ_LEN].copy()
        
        # Convert to meters relative to start point
        start_lat, start_lon = window[0, 0], window[0, 1]
        lat_rad = np.radians(start_lat)
        m_per_deg_lat = 110649.0
        m_per_deg_lon = 111132.0 * np.cos(lat_rad)
        
        gps_m = window[:, 0:2].copy()
        gps_m[:, 0] = (gps_m[:, 0] - start_lat) * m_per_deg_lat
        gps_m[:, 1] = (gps_m[:, 1] - start_lon) * m_per_deg_lon
        
        # Prepare features
        gps_norm = scalers['gps_point'].transform(gps_m).astype(np.float32)
        delta_gps = np.diff(gps_m, axis=0, prepend=gps_m[:1])
        delta_norm = scalers['delta'].transform(delta_gps).astype(np.float32)
        imu_norm = scalers['imu'].transform(window[:, 2:8]).astype(np.float32)
        
        features = np.concatenate([gps_norm, delta_norm, imu_norm], axis=1)
        
        # Predict
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        pred_norm = model(x_tensor).cpu().numpy()[0]
        pred_m = scalers['target'].inverse_transform(pred_norm.reshape(1, -1))[0]
        
        # Convert back to coordinates
        pred_lat = start_lat + pred_m[0] / m_per_deg_lat
        pred_lon = start_lon + pred_m[1] / m_per_deg_lon
        
        predictions.append([pred_lat, pred_lon])

predictions = np.array(predictions)

# Create comparison plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Full trajectory comparison
ax1.plot(data[:, 1], data[:, 0], 'r-', alpha=0.6, linewidth=1, label='Raw GPS')
ax1.plot(predictions[:, 1], predictions[:, 0], 'b-', linewidth=1.5, label='AI Corrected')
ax1.scatter(data[0, 1], data[0, 0], color='green', s=100, marker='o', label='Start')
ax1.scatter(data[-1, 1], data[-1, 0], color='red', s=100, marker='s', label='End')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
ax1.set_title('GPS Trajectory Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axis('equal')

# Plot 2: Zoomed section
mid_idx = len(data) // 2
zoom_range = 1000
start_zoom = max(0, mid_idx - zoom_range)
end_zoom = min(len(data), mid_idx + zoom_range)

ax2.plot(data[start_zoom:end_zoom, 1], data[start_zoom:end_zoom, 0], 'r-', alpha=0.6, linewidth=1, label='Raw GPS')
pred_start = max(0, start_zoom - (SEQ_LEN - 1))
pred_end = min(len(predictions), end_zoom - (SEQ_LEN - 1))
if pred_end > pred_start:
    ax2.plot(predictions[pred_start:pred_end, 1], predictions[pred_start:pred_end, 0], 'b-', linewidth=1.5, label='AI Corrected')
ax2.set_xlabel('Longitude')
ax2.set_ylabel('Latitude')
ax2.set_title('Zoomed View (Middle Section)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

# Plot 3: Step sizes
raw_steps = []
for i in range(1, len(data)):
    dlat = (data[i, 0] - data[i-1, 0]) * 110649.0
    dlon = (data[i, 1] - data[i-1, 1]) * 111132.0 * np.cos(np.radians(data[i-1, 0]))
    raw_steps.append(np.sqrt(dlat**2 + dlon**2))

pred_steps = []
for i in range(1, len(predictions)):
    dlat = (predictions[i, 0] - predictions[i-1, 0]) * 110649.0
    dlon = (predictions[i, 1] - predictions[i-1, 1]) * 111132.0 * np.cos(np.radians(predictions[i-1, 0]))
    pred_steps.append(np.sqrt(dlat**2 + dlon**2))

ax3.plot(raw_steps[:len(pred_steps)], 'r-', alpha=0.7, label='Raw GPS Step Size')
ax3.plot(pred_steps, 'b-', alpha=0.7, label='Corrected Step Size')
ax3.set_xlabel('Step Index')
ax3.set_ylabel('Distance (m)')
ax3.set_title('Step Size Comparison')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Statistics
stats_text = f"""Trajectory Statistics:

Raw GPS Points: {len(data)}
Corrected Points: {len(predictions)}

Raw GPS:
- Lat Range: {data[:, 0].min():.6f} to {data[:, 0].max():.6f}
- Lon Range: {data[:, 1].min():.6f} to {data[:, 1].max():.6f}
- Avg Step: {np.mean(raw_steps):.2f} m

Corrected GPS:
- Lat Range: {predictions[:, 0].min():.6f} to {predictions[:, 0].max():.6f}
- Lon Range: {predictions[:, 1].min():.6f} to {predictions[:, 1].max():.6f}
- Avg Step: {np.mean(pred_steps):.2f} m"""

ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, verticalalignment='top', 
         fontfamily='monospace', fontsize=9)
ax4.axis('off')
ax4.set_title('Statistics')

plt.tight_layout()
plt.savefig('trajectory_comparison.png', dpi=300, bbox_inches='tight')

# Save corrected data
df_corrected = df.copy()
df_corrected['Corrected_Lat'] = np.nan
df_corrected['Corrected_Lng'] = np.nan

valid_indices = df.index[valid_mask].tolist()
for i, pred in enumerate(predictions):
    if SEQ_LEN - 1 + i < len(valid_indices):
        original_idx = valid_indices[SEQ_LEN - 1 + i]
        df_corrected.loc[original_idx, 'Corrected_Lat'] = pred[0]
        df_corrected.loc[original_idx, 'Corrected_Lng'] = pred[1]

df_corrected.to_csv('corrected_trajectory.csv', index=False)

# Write summary
with open('model_results.txt', 'w') as f:
    f.write("Model Inference Results\n")
    f.write("======================\n")
    f.write(f"Input data points: {len(df)}\n")
    f.write(f"Valid GPS points: {len(data)}\n")
    f.write(f"Corrected points: {len(predictions)}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Sequence length: {SEQ_LEN}\n")
    f.write(f"Device used: {DEVICE}\n")
    f.write("\nFiles created:\n")
    f.write("- trajectory_comparison.png: Visual comparison\n")
    f.write("- corrected_trajectory.csv: Data with corrections\n")
    f.write("- model_results.txt: This summary\n")