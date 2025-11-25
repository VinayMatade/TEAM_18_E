#!/usr/bin/env python3
import argparse
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# CONFIGURATION
# -------------------------
CSV_FOLDER = "/content/fast_data/" 
NOISE_BANK_PATH = "/content/noise_bank.npy"

# HYPERPARAMETERS
EPOCHS = 50
BATCH_SIZE = 1024       # Optimized for T4 GPU
SEQ_LEN = 50            # 1 second history
LR = 0.001
PATIENCE = 6            # Early stopping
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. METRIC CALCULATOR (Hubballi Optimized)
# -------------------------
def calculate_accuracy(pred_deg, target_deg):
    """
    Calculates error in Meters.
    Optimized for Hubballi, India (Lat ~15.36)
    """
    # 1 deg Lat is always ~111km
    m_per_deg_lat = 110649.0 
    
    # 1 deg Lon shrinks as you go north. 
    # At Hubballi (15.36 deg): 111132 * cos(15.36) = ~107100m
    m_per_deg_lon = 107100.0
    
    # Convert Relative Degrees -> Meters
    pred_lat_m = pred_deg[:, 0] * m_per_deg_lat
    pred_lon_m = pred_deg[:, 1] * m_per_deg_lon
    
    targ_lat_m = target_deg[:, 0] * m_per_deg_lat
    targ_lon_m = target_deg[:, 1] * m_per_deg_lon
    
    # Euclidean Distance in Meters
    error_m = torch.sqrt((pred_lat_m - targ_lat_m)**2 + (pred_lon_m - targ_lon_m)**2)
    avg_mae_m = torch.mean(error_m).item()
    
    # Accuracy Score (0% = 5m error, 100% = 0m error)
    max_tolerance_m = 5.0
    accuracy = 100.0 * (1.0 - (avg_mae_m / max_tolerance_m))
    
    return avg_mae_m, max(0.0, min(100.0, accuracy))

# -------------------------
# 2. OPTIMIZED DATASET (With Relative Coords)
# -------------------------
class HybridNoiseDataset(Dataset):
    def __init__(self, clean_data, seq_len, noise_bank_path):
        self.clean_data = clean_data.astype(np.float32)
        self.seq_len = int(seq_len)
        self.n_samples = len(self.clean_data) - self.seq_len
        
        # Load Noise Bank
        try:
            self.real_noise_bank = np.load(noise_bank_path)
            print(f"   Dataset loaded {len(self.real_noise_bank)} real vibration samples.")
        except:
            print("⚠️ Warning: noise_bank.npy not found! Using fallback white noise.")
            self.real_noise_bank = None

        # Pre-Calculate Drift
        print("   Pre-calculating drift buffers...")
        self.drift_buffer_lat = self._make_drift_buffer(1_000_000)
        self.drift_buffer_lon = self._make_drift_buffer(1_000_000)

    def _make_drift_buffer(self, size):
        dt = 0.02
        tau = 300.0
        sigma = 2.5
        alpha = np.exp(-dt / tau)
        beta = sigma * np.sqrt(1 - alpha**2)
        white = np.random.normal(0, beta, size)
        buffer = np.zeros(size, dtype=np.float32)
        curr = 0.0
        for i in range(size):
            curr = alpha * curr + white[i]
            buffer[i] = curr
        return buffer

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # A. Get Window & Make Copy
        window = self.clean_data[idx : idx + self.seq_len].copy()
        
        # --- CRITICAL FIX: WINDOW CENTERING (Relative Coords) ---
        start_lat = window[0, 0]
        start_lon = window[0, 1]
        
        # Determine Scale Factor for this specific window (Dynamic)
        # (Even though your dataset is Hubballi, this makes the code robust for any location)
        lat_rad = np.radians(start_lat)
        m_per_deg_lat = 110649.0
        m_per_deg_lon = 111132.0 * np.cos(lat_rad)
        
        # Shift GPS to start at 0,0
        window[:, 0] -= start_lat
        window[:, 1] -= start_lon
        
        clean_gps = window[:, 0:2] # Now Relative (small numbers)
        clean_imu = window[:, 2:8]
        n_rows = self.seq_len
        
        # B. Slice Math Drift (Meters)
        d_start = np.random.randint(0, len(self.drift_buffer_lat) - n_rows)
        drift_lat_m = self.drift_buffer_lat[d_start : d_start + n_rows]
        drift_lon_m = self.drift_buffer_lon[d_start : d_start + n_rows]
        
        # C. Slice Real Vibration (Meters)
        if self.real_noise_bank is not None and len(self.real_noise_bank) > n_rows:
            r_start = np.random.randint(0, len(self.real_noise_bank) - n_rows)
            real_vib = self.real_noise_bank[r_start : r_start + n_rows]
            vib_lat_m = real_vib[:, 0]
            vib_lon_m = real_vib[:, 1]
        else:
            vib_lat_m = np.random.normal(0, 0.05, n_rows) 
            vib_lon_m = np.random.normal(0, 0.05, n_rows)
            
        # D. Combine Noise (Meters) -> Degrees
        total_lat_m = drift_lat_m + vib_lat_m
        total_lon_m = drift_lon_m + vib_lon_m
        
        noise_deg_lat = total_lat_m / m_per_deg_lat
        noise_deg_lon = total_lon_m / m_per_deg_lon
        
        # Apply noise to Relative GPS
        noisy_lat = clean_gps[:, 0] + noise_deg_lat
        noisy_lon = clean_gps[:, 1] + noise_deg_lon
        
        # E. Fake HAcc
        xy_err = np.sqrt(total_lat_m**2 + total_lon_m**2)
        fake_hacc = xy_err * np.random.uniform(0.8, 1.2, size=n_rows)
        
        # F. Degrade IMU
        imu_noise = np.random.normal(0, 0.15, size=clean_imu.shape).astype(np.float32)
        noisy_imu = clean_imu + imu_noise
        
        # G. Stack
        noisy_gps_block = np.stack([noisy_lat, noisy_lon, fake_hacc], axis=1)
        x_data = np.concatenate([noisy_gps_block, noisy_imu], axis=1)
        y_target = clean_gps[-1, :] 
        
        return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# -------------------------
# 3. MODEL (TCN)
# -------------------------
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

# -------------------------
# 4. MAIN
# -------------------------
def main():
    print(f"Running on {DEVICE}")
    
    # 1. Load Data
    REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    print(f"Found {len(files)} logs in {CSV_FOLDER}")
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS): continue
            data = df[REQUIRED_COLS].values
            data = data[~np.isnan(data).any(axis=1)]
            if len(data) > SEQ_LEN:
                all_data.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not all_data:
        print("❌ No valid data found!")
        return
        
    full_dataset = np.vstack(all_data)
    print(f"Total Samples: {len(full_dataset)}")
    
    split_idx = int(len(full_dataset) * 0.8)
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]
    
    # 2. Setup Dataloaders (Optimized for Speed)
    train_ds = HybridNoiseDataset(train_data, SEQ_LEN, NOISE_BANK_PATH)
    val_ds = HybridNoiseDataset(val_data, SEQ_LEN, NOISE_BANK_PATH)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, 
                            num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    
    model = TCN(input_size=9, output_size=2, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    # 3. Training Loop with Early Stopping
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_loss = float('inf')
    patience_counter = 0
    
    print("\n🚀 Starting Training...")
    print(f"{'Epoch':<5} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val MAE (m)':<12} | {'Val Acc %':<10}")
    print("-" * 65)
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        val_accs = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
                
                # Metrics
                mae_m, acc_pct = calculate_accuracy(pred, y)
                val_maes.append(mae_m)
                val_accs.append(acc_pct)
                
        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)
        avg_val_mae = np.mean(val_maes)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        print(f"{epoch+1:<5} | {avg_train_loss:.8f}   | {avg_val_loss:.8f}   | {avg_val_mae:.4f}       | {avg_val_acc:.2f}%")

        # Early Stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early Stopping Triggered!")
            break

    torch.save(model.state_dict(), "final_model.pth")
    print("\n✅ Training Complete.")
    
    plt.figure(figsize=(10,5))
    plt.plot(history['val_acc'], label='Val Accuracy %', color='green')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    main()
