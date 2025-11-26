#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# CONFIGURATION
# -------------------------
CSV_FOLDER = "/content/TEAM_18_E/files/train"
NOISE_BANK_PATH = "/content/TEAM_18_E/noise_bank.npy"
REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']

EPOCHS = 60
BATCH_SIZE = 512
SEQ_LEN = 125
LR = 0.0005
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # explicit for determinism

# -------------------------
# HELPERS
# -------------------------
def make_drift_buffer(size):
    dt, tau, sigma = 0.02, 300.0, 2.5
    alpha = np.exp(-dt / tau)
    beta = sigma * np.sqrt(1 - alpha**2)
    buffer = np.zeros(size, dtype=np.float32)
    chunk = 5000
    for i in range(0, size, chunk):
        l = min(chunk, size - i)
        w = np.random.normal(0, beta, l)
        c = 0.0
        for j in range(l):
            c = alpha * c + w[j]
            buffer[i + j] = c
    return buffer

def imu_integrate_velocity(acc_window, dt=0.02):
    acc = acc_window.float()
    acc = acc - torch.mean(acc, dim=1, keepdim=True)
    if acc.size(1) < 2:
        # return mean accel * dt as a stable fallback (shape: (B,2))
        return acc.mean(dim=1) * dt
    # Trapezoidal → cumulative velocity
    vel = torch.cumsum((acc[:, :-1, :] + acc[:, 1:, :]) * 0.5 * dt, dim=1)
    return vel[:, -1, :]  # final velocity over window

# -------------------------
# DATASET
# -------------------------
class HybridNoiseDataset(Dataset):
    def __init__(self, file_list, seq_len, scalers, shared):
        self.seq_len = int(seq_len)
        self.imu_scaler = scalers['imu']
        self.gps_scaler = scalers['gps_point']
        self.target_scaler = scalers['target']
        self.drift_lat = shared['drift_lat']
        self.drift_lon = shared['drift_lon']
        self.real_noise = shared.get('real_noise', None)
        self.drift_len = len(self.drift_lat)
        self.noise_len = len(self.real_noise) if (self.real_noise is not None) else 0

        data_list = []
        for f in file_list:
            try:
                if isinstance(f, str):
                    df = pd.read_csv(f)
                    if not all(c in df.columns for c in REQUIRED_COLS): continue
                    arr = df[REQUIRED_COLS].values.astype(np.float32)
                else:
                    arr = f
                arr = arr[~np.isnan(arr).any(axis=1)]
                valid = (arr[:, 0] != 0) & (arr[:, 1] != 0)
                arr = arr[valid]
                if len(arr) > seq_len:
                    data_list.append(arr)
            except Exception:
                pass

        if not data_list:
            raise ValueError("No data!")
        self.raw_data = np.vstack(data_list)
        self.n_samples = len(self.raw_data) - self.seq_len

        self.norm_imu = self.imu_scaler.transform(self.raw_data[:, 2:8]).astype(np.float32)
        self.raw_imu = self.raw_data[:, 2:8].astype(np.float32)
        self.raw_gps = self.raw_data[:, 0:2].astype(np.float32)

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        norm_imu_window = self.norm_imu[idx: idx + self.seq_len]
        raw_imu_window = self.raw_imu[idx: idx + self.seq_len]
        gps_window = self.raw_gps[idx: idx + self.seq_len].copy()

        start_lat, start_lon = gps_window[0, 0], gps_window[0, 1]
        mlat = 110649.0
        mlon = 111132.0 * np.cos(np.radians(start_lat))

        gps_window[:, 0] = (gps_window[:, 0] - start_lat) * mlat
        gps_window[:, 1] = (gps_window[:, 1] - start_lon) * mlon
        clean_gps_m = gps_window

        # safe drift start (inclusive)
        max_start = max(0, self.drift_len - self.seq_len)
        if max_start > 0:
            d_start = np.random.randint(0, max_start + 1)  # inclusive upper bound
        else:
            d_start = 0
        drift = np.stack([self.drift_lat[d_start : d_start + self.seq_len],
                          self.drift_lon[d_start : d_start + self.seq_len]], axis=1)

        # vib / real-noise fallback with safe indexing
        if self.noise_len >= self.seq_len:
            max_r = self.noise_len - self.seq_len
            if max_r > 0:
                r = np.random.randint(0, max_r + 1)
            else:
                r = 0
            vib = self.real_noise[r : r + self.seq_len]
        else:
            vib = np.random.normal(0, 0.05, (self.seq_len, 2)).astype(np.float32)

        noisy_gps_m = clean_gps_m + drift + vib

        # Target: Displacement (end - start)
        displacement_m = clean_gps_m[-1] - clean_gps_m[0]
        target_norm = self.target_scaler.transform(displacement_m.reshape(1, -1)).flatten().astype(np.float32)

        # Features: gps, delta-gps, norm imu
        gps_norm = self.gps_scaler.transform(noisy_gps_m).astype(np.float32)
        delta_gps = np.diff(noisy_gps_m, axis=0, prepend=noisy_gps_m[:1])
        delta_norm = self.gps_scaler.transform(delta_gps).astype(np.float32)

        x_full = np.concatenate([gps_norm, delta_norm, norm_imu_window], axis=1).astype(np.float32)
        raw_accel_xy = raw_imu_window[:, 0:2].astype(np.float32)

        return torch.tensor(x_full, dtype=torch.float32), torch.tensor(target_norm, dtype=torch.float32), torch.tensor(raw_accel_xy, dtype=torch.float32)

# -------------------------
# MODEL
# -------------------------
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1, self.conv2, self.relu2, self.dropout2)
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
            dilation_size = dilations[i]
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

# -------------------------
# MAIN
# -------------------------
def main():
    print(f"Running on {DEVICE}")
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    np.random.shuffle(files)
    split_idx = int(len(files) * 0.8)
    train_files, val_files = files[:split_idx], files[split_idx:]

    # 1. Scalers
    print("Fitting Scalers...")
    temp_arrs = []
    for f in train_files:
        try:
            df = pd.read_csv(f)
            if all(c in df.columns for c in REQUIRED_COLS):
                arr = df[REQUIRED_COLS].values.astype(np.float32)
                arr = arr[~np.isnan(arr).any(axis=1)]
                if len(arr) > SEQ_LEN: temp_arrs.append(arr)
        except: pass

    if not temp_arrs:
        print("No training files found or too short.")
        return

    all_train = np.vstack(temp_arrs)
    imu_scaler = StandardScaler().fit(all_train[:, 2:8])

    gps_pts = []
    displacements = []
    for arr in temp_arrs:
        if len(arr) <= SEQ_LEN:
            continue
        indices = np.random.randint(0, len(arr)-SEQ_LEN, min(50, len(arr)-SEQ_LEN))
        for i in indices:
            w = arr[i:i+SEQ_LEN, 0:2].copy()
            slat, slon = w[0,0], w[0,1]
            mlat, mlon = 110649.0, 111132.0 * np.cos(np.radians(slat))
            w[:,0] = (w[:,0]-slat)*mlat
            w[:,1] = (w[:,1]-slon)*mlon
            gps_pts.append(w)
            displacements.append((w[-1] - w[0]).reshape(1,2))

    gps_point_scaler = StandardScaler().fit(np.vstack(gps_pts))
    target_scaler = StandardScaler().fit(np.vstack(displacements))
    scalers = {'imu': imu_scaler, 'gps_point': gps_point_scaler, 'target': target_scaler}
    joblib.dump(scalers, "scalers.save")

    # 2. Shared Resources
    shared = {
        'drift_lat': make_drift_buffer(200_000),
        'drift_lon': make_drift_buffer(200_000),
        'real_noise': None
    }
    try:
        rn = np.load(NOISE_BANK_PATH)
        shared['real_noise'] = rn if len(rn) >= SEQ_LEN else None
    except:
        shared['real_noise'] = None

    train_ds = ConcatDataset([HybridNoiseDataset([arr], SEQ_LEN, scalers, shared) for arr in temp_arrs])
    val_arrs = []
    for f in val_files:
        try:
            df = pd.read_csv(f)
            arr = df[REQUIRED_COLS].values.astype(np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if len(arr) > SEQ_LEN: val_arrs.append(arr)
        except: pass
    if not val_arrs:
        print("No validation data found.")
        return
    val_ds = ConcatDataset([HybridNoiseDataset([arr], SEQ_LEN, scalers, shared) for arr in val_arrs])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 3. Model
    model = TCN(
        input_size=10,
        output_size=2,
        num_channels=[128, 128, 128, 128, 128, 128],
        kernel_size=7,
        dropout=0.3,
        dilations=[1, 2, 4, 8, 16, 32]
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss(beta=0.5)

    t_scale = torch.tensor(target_scaler.scale_, device=DEVICE, dtype=torch.float32)
    t_mean = torch.tensor(target_scaler.mean_, device=DEVICE, dtype=torch.float32)

    print(f"Training {len(train_ds)} samples. Physics Enabled.")
    best_mae = float('inf')
    history = {'val_mae': []}

    lambda_phy = 0.01

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        if epoch > 5 and lambda_phy < 0.1:
            lambda_phy += 0.01

        # H curriculum for velocity integration if you later switch to short-horizon physics
        # (kept here for future extension)
        for x, y, raw_acc in train_loader:
            x, y, raw_acc = x.to(DEVICE), y.to(DEVICE), raw_acc.to(DEVICE)
            optimizer.zero_grad()
            pred_norm = model(x)  # (B, 2)

            # 1. Position loss (normalized)
            loss_pos = criterion(pred_norm, y)

            # 2. Physics / Velocity Loss (unified)
            pred_m = pred_norm * t_scale + t_mean  # displacement in meters
            pred_vel_m_s = pred_m / (SEQ_LEN * 0.02 + 1e-8)  # velocity estimate (m/s)

            # IMU velocity integrated over full window
            imu_vel_m_s = imu_integrate_velocity(raw_acc, dt=0.02)
            imu_vel_m_s = imu_vel_m_s.to(pred_vel_m_s.device).to(pred_vel_m_s.dtype)

            loss_phy = criterion(pred_vel_m_s, imu_vel_m_s)

            # Combine
            loss = loss_pos + lambda_phy * loss_phy

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        # Validation (use full-window prediction)
        model.eval()
        total_err = 0.0
        count = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                pred_m = pred * t_scale + t_mean
                targ_m = y * t_scale + t_mean
                err = torch.sqrt(torch.sum((pred_m - targ_m) ** 2, dim=1))
                total_err += torch.sum(err).item()
                count += err.size(0)

        avg_mae = total_err / max(1, count)  # guard division
        scheduler.step(avg_mae)
        history['val_mae'].append(avg_mae)

        print(f"{epoch+1:<3d} | Loss: {np.mean(batch_losses):.5f} | Val MAE: {avg_mae:.4f}m | Phy: {lambda_phy:.2f}")

        if avg_mae < best_mae:
            best_mae = avg_mae
            torch.save(model.state_dict(), "best_model.pth")

    print(f"✅ Best MAE: {best_mae:.4f}m")
    torch.save(model.state_dict(), "final_model.pth")
    plt.plot(history['val_mae'])
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    main()
