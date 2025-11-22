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
CSV_FOLDER = "files/cleaned/train/"
NOISE_BANK_PATH = "noise_bank.npy"

EPOCHS = 50
BATCH_SIZE = 64
SEQ_LEN = 50
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CONSTANTS
METERS_TO_DEG = 1.0 / 111139.0
DEG_TO_METERS = 111139.0

# -------------------------
# 1. METRIC CALCULATOR
# -------------------------


def calculate_accuracy(pred_deg, target_deg):
    """
    Calculates error in Meters and an Accuracy % score.
    """
    # Convert PyTorch tensors to Meters
    pred_m = pred_deg * DEG_TO_METERS
    target_m = target_deg * DEG_TO_METERS

    # 1. MAE (Mean Absolute Error in Meters)
    # Shape: (Batch, 2) -> Calculate distance for each point
    # Euclidean distance error per point
    error_m = torch.sqrt(torch.sum((pred_m - target_m)**2, dim=1))
    avg_mae_m = torch.mean(error_m).item()

    # 2. Accuracy % (Pseudo-mAP)
    # We define "0% Accuracy" as an error of 5.0 meters (since that's our max noise)
    # We define "100% Accuracy" as 0.0 meters error.
    max_tolerance_m = 5.0
    accuracy = 100.0 * (1.0 - (avg_mae_m / max_tolerance_m))

    # Clamp between 0 and 100
    return avg_mae_m, max(0.0, min(100.0, accuracy))

# -------------------------
# 2. DRIFT GENERATOR
# -------------------------


def generate_ou_drift(n_samples, dt=0.02, tau=300.0, sigma=2.5):
    noise = np.zeros(n_samples)
    white_noise = np.random.normal(0, 1, n_samples)
    alpha = np.exp(-dt / tau)
    beta = sigma * np.sqrt(1 - alpha**2)
    current = 0.0
    for i in range(n_samples):
        current = alpha * current + beta * white_noise[i]
        noise[i] = current
    return noise

# -------------------------
# 3. DATASET
# -------------------------


class HybridNoiseDataset(Dataset):
    def __init__(self, clean_data, seq_len, noise_bank_path):
        self.clean_data = clean_data.astype(np.float32)
        self.seq_len = int(seq_len)
        self.n_samples = len(self.clean_data) - self.seq_len

        try:
            self.real_noise_bank = np.load(noise_bank_path)
            print(
                f"   Dataset loaded {len(self.real_noise_bank)} real vibration samples.")
        except:
            print("⚠️ Warning: noise_bank.npy not found! Using fallback white noise.")
            self.real_noise_bank = None

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        window = self.clean_data[idx: idx + self.seq_len]
        clean_gps = window[:, 0:2]
        clean_imu = window[:, 2:8]
        n_rows = self.seq_len

        # Drift + Vibration
        drift_lat = generate_ou_drift(n_rows, tau=200, sigma=3.0)
        drift_lon = generate_ou_drift(n_rows, tau=200, sigma=3.0)

        if self.real_noise_bank is not None and len(self.real_noise_bank) > n_rows:
            rand_start = np.random.randint(
                0, len(self.real_noise_bank) - n_rows)
            real_vib = self.real_noise_bank[rand_start: rand_start + n_rows]
            vib_lat = real_vib[:, 0]
            vib_lon = real_vib[:, 1]
        else:
            vib_lat = np.random.normal(0, 0.05, n_rows)
            vib_lon = np.random.normal(0, 0.05, n_rows)

        total_lat_m = drift_lat + vib_lat
        total_lon_m = drift_lon + vib_lon

        noisy_lat = clean_gps[:, 0] + (total_lat_m * METERS_TO_DEG)
        noisy_lon = clean_gps[:, 1] + (total_lon_m * METERS_TO_DEG)

        xy_err = np.sqrt(total_lat_m**2 + total_lon_m**2)
        fake_hacc = xy_err * np.random.uniform(0.8, 1.2, size=n_rows)

        imu_noise = np.random.normal(0, 0.15, size=clean_imu.shape)
        noisy_imu = clean_imu + imu_noise

        noisy_gps_block = np.stack([noisy_lat, noisy_lon, fake_hacc], axis=1)
        x_data = np.concatenate([noisy_gps_block, noisy_imu], axis=1)
        y_target = clean_gps[-1, :]

        return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_target, dtype=torch.float32)

# -------------------------
# 4. MODEL
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
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
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
# 5. MAIN
# -------------------------


def main():
    print(f"Running on {DEVICE}")

    # Load Data
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
            if len(data) > SEQ_LEN:
                all_data.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not all_data:
        print("❌ No valid data found!")
        return

    full_dataset = np.vstack(all_data)
    split_idx = int(len(full_dataset) * 0.8)
    train_data = full_dataset[:split_idx]
    val_data = full_dataset[split_idx:]

    train_ds = HybridNoiseDataset(train_data, SEQ_LEN, NOISE_BANK_PATH)
    val_ds = HybridNoiseDataset(val_data, SEQ_LEN, NOISE_BANK_PATH)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, drop_last=True)

    model = TCN(input_size=9, output_size=2, num_channels=[
                64, 128, 64], kernel_size=3, dropout=0.2)
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0

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

                # Calculate Metrics
                mae_m, acc_pct = calculate_accuracy(pred, y)
                val_maes.append(mae_m)
                val_accs.append(acc_pct)

        avg_val_loss = np.mean(val_losses)
        avg_val_mae = np.mean(val_maes)
        avg_val_acc = np.mean(val_accs)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)

        print(f"{epoch+1:<5} | {avg_train_loss:.8f}   | {avg_val_loss:.8f}   | {avg_val_mae:.4f}       | {avg_val_acc:.2f}%")

        # Save Best Model based on ACCURACY, not just Loss
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), "best_model.pth")

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_acc'], label='Val Accuracy %', color='green')
    plt.title('Validation Accuracy (Higher is Better)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy %')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    print(f"\n✅ Best Validation Accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
