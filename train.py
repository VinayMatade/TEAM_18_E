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
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# CONFIGURATION
# -------------------------
CSV_FOLDER = "/content/TEAM_18_E/clean/train"
NOISE_BANK_PATH = "/content/TEAM_18_E/noise_bank.npy"

# HYPERPARAMETERS
EPOCHS = 50
BATCH_SIZE = 1024
SEQ_LEN = 50
LR = 0.001
PATIENCE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# 1. METRIC CALCULATOR
# -------------------------


def calculate_accuracy(pred_m, target_m):
    # Euclidean Distance in Meters
    error_m = torch.sqrt(torch.sum((pred_m - target_m)**2, dim=1))
    avg_mae_m = torch.mean(error_m).item()

    # Accuracy Score (0% = 5m error, 100% = 0m error)
    max_tolerance_m = 5.0
    accuracy = 100.0 * (1.0 - (avg_mae_m / max_tolerance_m))

    return avg_mae_m, max(0.0, min(100.0, accuracy))

# -------------------------
# 2. DATASET (Optimized & Scaled)
# -------------------------


class HybridNoiseDataset(Dataset):
    def __init__(self, clean_data, seq_len, noise_bank_path, scaler=None):
        self.raw_data = clean_data.astype(np.float32)
        self.seq_len = int(seq_len)
        self.n_samples = len(self.raw_data) - self.seq_len

        # Fit Scaler on IMU columns (2-7) if not provided
        if scaler is None:
            print("   Fitting StandardScaler on training data...")
            self.scaler = StandardScaler()
            self.scaler.fit(self.raw_data[:, 2:8])
        else:
            self.scaler = scaler

        # Transform IMU data immediately
        self.norm_imu = self.scaler.transform(
            self.raw_data[:, 2:8]).astype(np.float32)
        self.gps_data = self.raw_data[:, 0:2]

        try:
            self.real_noise_bank = np.load(noise_bank_path)
            print(
                f"   Dataset loaded {len(self.real_noise_bank)} real vibration samples.")
        except:
            print("⚠️ Warning: noise_bank.npy not found! Using fallback white noise.")
            self.real_noise_bank = None

        print("   Pre-calculating drift buffers...")
        self.drift_buffer_lat = self._make_drift_buffer(1_000_000)
        self.drift_buffer_lon = self._make_drift_buffer(1_000_000)

    def _make_drift_buffer(self, size):
        dt = 0.02
        tau = 300.0
        sigma = 2.5
        alpha = np.exp(-dt / tau)
        beta = sigma * np.sqrt(1 - alpha**2)

        buffer = np.zeros(size, dtype=np.float32)
        chunk_size = 1000  # Reset every 20 seconds

        for i in range(0, size, chunk_size):
            end = min(i + chunk_size, size)
            length = end - i
            white = np.random.normal(0, beta, length)
            curr = 0.0
            for j in range(length):
                curr = alpha * curr + white[j]
                buffer[i + j] = curr
        return buffer

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # A. Get Window
        norm_imu_window = self.norm_imu[idx: idx + self.seq_len]
        gps_window = self.gps_data[idx: idx + self.seq_len].copy()

        # B. Center & Scale GPS to Meters
        start_lat = gps_window[0, 0]
        start_lon = gps_window[0, 1]

        lat_rad = np.radians(start_lat)
        m_per_deg_lat = 110649.0
        m_per_deg_lon = 111132.0 * np.cos(lat_rad)

        # Relative Degrees
        gps_window[:, 0] -= start_lat
        gps_window[:, 1] -= start_lon

        # Convert to Meters
        gps_window[:, 0] *= m_per_deg_lat
        gps_window[:, 1] *= m_per_deg_lon

        clean_gps_m = gps_window
        n_rows = self.seq_len

        # C. Add Noise (Already in Meters)
        d_start = np.random.randint(0, len(self.drift_buffer_lat) - n_rows)
        drift_lat_m = self.drift_buffer_lat[d_start: d_start + n_rows]
        drift_lon_m = self.drift_buffer_lon[d_start: d_start + n_rows]

        if self.real_noise_bank is not None and len(self.real_noise_bank) > n_rows:
            r_start = np.random.randint(0, len(self.real_noise_bank) - n_rows)
            real_vib = self.real_noise_bank[r_start: r_start + n_rows]
            vib_lat_m = real_vib[:, 0]
            vib_lon_m = real_vib[:, 1]
        else:
            vib_lat_m = np.random.normal(0, 0.05, n_rows)
            vib_lon_m = np.random.normal(0, 0.05, n_rows)

        total_lat_m = drift_lat_m + vib_lat_m
        total_lon_m = drift_lon_m + vib_lon_m

        noisy_lat_m = clean_gps_m[:, 0] + total_lat_m
        noisy_lon_m = clean_gps_m[:, 1] + total_lon_m

        # Fake HAcc (Normalized approx 0-1 range)
        xy_err = np.sqrt(total_lat_m**2 + total_lon_m**2)
        fake_hacc = xy_err * np.random.uniform(0.8, 1.2, size=n_rows)
        fake_hacc = fake_hacc / 5.0

        # Add noise to normalized IMU
        imu_noise = np.random.normal(
            0, 0.1, size=norm_imu_window.shape).astype(np.float32)
        noisy_imu = norm_imu_window + imu_noise

        # Stack
        noisy_gps_block = np.stack(
            [noisy_lat_m, noisy_lon_m, fake_hacc], axis=1)
        x_data = np.concatenate([noisy_gps_block, noisy_imu], axis=1)
        y_target = clean_gps_m[-1, :]

        return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# -------------------------
# 3. MODEL (Updated Weight Norm)
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
        # Updated Weight Norm for modern PyTorch
        try:
            # Try new way (PyTorch 2.1+)
            from torch.nn.utils.parametrizations import weight_norm
            self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
            self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                               stride=stride, padding=padding, dilation=dilation))
        except ImportError:
            # Fallback to old way
            self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                        stride=stride, padding=padding, dilation=dilation))
            self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                        stride=stride, padding=padding, dilation=dilation))

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

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

    REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX',
                     'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    print(f"Found {len(files)} logs in {CSV_FOLDER}")

    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS):
                continue
            data = df[REQUIRED_COLS].values
            data = data[~np.isnan(data).any(axis=1)]
            valid_gps = (data[:, 0] != 0.0) & (data[:, 1] != 0.0)
            data = data[valid_gps]
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

    # 1. Create Train Dataset (Fits Scaler)
    train_ds = HybridNoiseDataset(
        train_data, SEQ_LEN, NOISE_BANK_PATH, scaler=None)
    joblib.dump(train_ds.scaler, "scaler.save")
    print("✅ Scaler saved.")

    # 3. Create Val Dataset (Uses fitted Scaler)
    val_ds = HybridNoiseDataset(
        val_data, SEQ_LEN, NOISE_BANK_PATH, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            num_workers=2, pin_memory=True, persistent_workers=True, drop_last=True)

    model = TCN(input_size=9, output_size=2, num_channels=[
        128, 256, 128], kernel_size=3, dropout=0.2)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.SmoothL1Loss()

    # Removed verbose=True to fix error
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_loss = float('inf')
    patience_counter = 0

    print("\n🚀 Starting Training...")
    print(f"{'Epoch':<5} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val MAE (m)':<10} | {'Val Acc %':<10}")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                mae_m, acc_pct = calculate_accuracy(pred, y)
                val_maes.append(mae_m)
                val_accs.append(acc_pct)

        avg_val_loss = np.mean(val_losses)
        avg_val_acc = np.mean(val_accs)
        avg_val_mae = np.mean(val_maes)

        # Manually check for LR drop since verbose=True is gone
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        print(f"{epoch+1:<5} | {avg_train_loss:.6f}   | {avg_val_loss:.6f}   | {avg_val_mae:.4f}     | {avg_val_acc:.2f}%")

        if new_lr < current_lr:
            print(f"   📉 Learning Rate dropped to {new_lr:.2e}")

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

    plt.figure(figsize=(10, 5))
    plt.plot(history['val_acc'], label='Val Accuracy %', color='green')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.savefig('training_curve.png')


if __name__ == "__main__":
    main()
