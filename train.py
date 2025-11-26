#!/usr/bin/env python3
import glob
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib

# -------------------------
# CONFIGURATION
# -------------------------
CSV_FOLDER = "/content/TEAM_18_E/files/cleaned/train"       # adjust if needed
NOISE_BANK_PATH = "/content/TEAM_18_E/noise_bank.npy"
REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng',
                 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ',
                 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']

EPOCHS = 50
BATCH_SIZE = 256  # reduced for stability
SEQ_LEN = 50
LR = 0.001
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Deterministic seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# HELPERS
# -------------------------
def make_drift_buffer(size):
    """Generate drift buffer once and share across all datasets"""
    dt = 0.02
    tau = 300.0
    sigma = 2.5
    alpha = np.exp(-dt / tau)
    beta = sigma * np.sqrt(1 - alpha**2)
    buffer = np.zeros(size, dtype=np.float32)
    chunk = 1000
    for i in range(0, size, chunk):
        l = min(chunk, size - i)
        w = np.random.normal(0, beta, l)
        c = 0.0
        for j in range(l):
            c = alpha * c + w[j]
            buffer[i + j] = c
    return buffer

def _safe_inv(scaler, arr):
    """Ensure arr is 2-D before inverse_transform; return (N, D) array."""
    arr2 = np.atleast_2d(arr)
    return scaler.inverse_transform(arr2)

# -------------------------
# DATASET
# -------------------------
class HybridNoiseDataset(Dataset):
    """
    file_list can contain either:
      - a string path to a CSV file, or
      - an already-loaded numpy.ndarray with the REQUIRED_COLS order.
    The dataset normalizes IMU using imu_scaler and normalizes GPS (meters) using target_scaler.
    Shared must contain 'drift_lat', 'drift_lon', and 'real_noise' (or None).
    """
    def __init__(self, file_list, seq_len, scalers, shared):
        self.seq_len = int(seq_len)
        self.imu_scaler = scalers['imu']
        self.target_scaler = scalers['target']

        self.drift_buffer_lat = shared['drift_lat']
        self.drift_buffer_lon = shared['drift_lon']
        self.real_noise_bank = shared['real_noise']

        # Load/accept arrays
        data_list = []
        for f in file_list:
            try:
                if isinstance(f, str):
                    df = pd.read_csv(f)
                    if not all(c in df.columns for c in REQUIRED_COLS):
                        continue
                    arr = df[REQUIRED_COLS].values.astype(np.float32)
                else:
                    # assume numpy array already in REQUIRED_COLS order
                    arr = np.asarray(f, dtype=np.float32)
                # drop NaNs and null island
                arr = arr[~np.isnan(arr).any(axis=1)]
                valid = (arr[:, 0] != 0) & (arr[:, 1] != 0)
                arr = arr[valid]
                if len(arr) > seq_len:
                    data_list.append(arr)
            except Exception:
                # silent skip
                pass

        if not data_list:
            raise ValueError("No valid data loaded for dataset!")

        self.raw_data = np.vstack(data_list)  # (N, 8)
        self.n_samples = len(self.raw_data) - self.seq_len

        # Pre-transform IMU once
        self.norm_imu = self.imu_scaler.transform(self.raw_data[:, 2:8]).astype(np.float32)
        self.raw_gps = self.raw_data[:, 0:2]  # degrees

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # Window slices
        imu_window = self.norm_imu[idx: idx + self.seq_len]  # (seq_len, 6)
        gps_window = self.raw_gps[idx: idx + self.seq_len].copy()  # degrees

        # Center & convert to meters (per-window lon scaling)
        start_lat = gps_window[0, 0]
        start_lon = gps_window[0, 1]
        lat_rad = np.radians(start_lat)
        m_per_deg_lat = 110649.0
        m_per_deg_lon = 111132.0 * np.cos(lat_rad)

        gps_window[:, 0] = (gps_window[:, 0] - start_lat) * m_per_deg_lat
        gps_window[:, 1] = (gps_window[:, 1] - start_lon) * m_per_deg_lon
        clean_gps_m = gps_window  # (seq_len, 2)

        # Noise (safe random starts)
        n_rows = self.seq_len
        max_start = max(1, len(self.drift_buffer_lat) - n_rows + 1)
        d_start = np.random.randint(0, max_start)
        drift = np.stack([
            self.drift_buffer_lat[d_start:d_start + n_rows],
            self.drift_buffer_lon[d_start:d_start + n_rows]
        ], axis=1)

        if self.real_noise_bank is not None and len(self.real_noise_bank) >= n_rows:
            max_start2 = len(self.real_noise_bank) - n_rows + 1
            r = np.random.randint(0, max_start2)
            vib = self.real_noise_bank[r:r + n_rows]
        else:
            vib = np.random.normal(0, 0.05, (n_rows, 2)).astype(np.float32)

        total_noise_m = drift + vib
        noisy_gps_m = clean_gps_m + total_noise_m

        # Normalize GPS to the same scaler used for target
        noisy_gps_norm = self.target_scaler.transform(noisy_gps_m).astype(np.float32)
        target_norm = self.target_scaler.transform(clean_gps_m[-1:].astype(np.float32)).flatten().astype(np.float32)

        # Final input: [lat_norm, lon_norm, imu(6)] -> shape (seq_len, 8)
        x_full = np.concatenate([noisy_gps_norm, imu_window], axis=1).astype(np.float32)

        return torch.tensor(x_full, dtype=torch.float32), torch.tensor(target_norm, dtype=torch.float32)

# -------------------------
# MODEL (TCN)
# -------------------------
class Chomp1d(nn.Module):
    """Remove the extra padding on the right side produced by Conv1d with 'padding'."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Standard TCN temporal block (no weight-norm). Uses two Conv1d layers with
    ReLU + Dropout, a residual connection, and a Chomp to remove causal padding.
    """

    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 kernel_size: int,
                 stride: int,
                 dilation: int,
                 padding: int,
                 dropout: float = 0.2):
        super().__init__()

        # conv layers with padding chosen so that length_out = length_in + padding*2 - ...
        # We'll remove the right-side padding using Chomp1d(padding)
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

        # Sequential block for the two convs
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # 1x1 downsample if channels differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        # Small gaussian init is common in TCN implementations
        nn.init.normal_(self.conv1.weight, 0.0, 0.01)
        nn.init.normal_(self.conv2.weight, 0.0, 0.01)
        if self.conv1.bias is not None:
            nn.init.zeros_(self.conv1.bias)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0.0, 0.01)
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels_in, seq_len)
        returns: (batch, channels_out, seq_len)
        """
        out = self.net(x)                          # Conv path -> (B, C_out, seq_len)
        res = x if self.downsample is None else self.downsample(x)
        # After chomp, out and res should have identical time dimension
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, dropout=dropout))
        self.net = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)
        y = self.net(x)
        return self.linear(y[:, :, -1])

# -------------------------
# MAIN
# -------------------------
def main():
    print(f"Running on {DEVICE}")

    # 1. File split
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    print(f"Found {len(files)} logs in {CSV_FOLDER}")
    np.random.shuffle(files)
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")

    # 2. Load train arrays once (avoid double reads) and build temp list for scalers
    temp_train_arrs = []
    for f in train_files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS):
                continue
            arr = df[REQUIRED_COLS].values.astype(np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            valid = (arr[:, 0] != 0) & (arr[:, 1] != 0)
            arr = arr[valid]
            if len(arr) > SEQ_LEN:
                temp_train_arrs.append(arr)
        except Exception:
            pass

    if not temp_train_arrs:
        print("❌ No valid training data found!")
        return

    # Fit IMU scaler on stacked IMU columns
    all_train_arr = np.vstack(temp_train_arrs)
    imu_scaler = StandardScaler().fit(all_train_arr[:, 2:8])

    # Fit target scaler on relative meters using per-window centering (sampled windows)
    print("Computing relative GPS stats for target scaler...")
    rel_samples = []
    for arr in temp_train_arrs:
        if len(arr) <= SEQ_LEN:
            continue
        picks = min(100, len(arr) - SEQ_LEN)
        indices = np.random.randint(0, len(arr) - SEQ_LEN, picks)
        for i in indices:
            window = arr[i:i + SEQ_LEN, 0:2].copy()
            start_lat, start_lon = window[0, 0], window[0, 1]
            lat_rad = np.radians(start_lat)
            mlat = 110649.0
            mlon = 111132.0 * np.cos(lat_rad)
            window[:, 0] = (window[:, 0] - start_lat) * mlat
            window[:, 1] = (window[:, 1] - start_lon) * mlon
            rel_samples.append(window)
    target_scaler = StandardScaler().fit(np.vstack(rel_samples))
    print("Target scaler mean:", target_scaler.mean_)

    scalers = {'imu': imu_scaler, 'target': target_scaler}
    joblib.dump(scalers, "scalers.save")
    print("Scalers saved to scalers.save")

    # 3. Create shared resources (one-time)
    shared = {}
    shared['drift_lat'] = make_drift_buffer(size=200_000)   # reduce if memory constrained
    shared['drift_lon'] = make_drift_buffer(size=200_000)
    try:
        shared['real_noise'] = np.load(NOISE_BANK_PATH)
        print(f"Loaded {len(shared['real_noise'])} real noise samples.")
    except Exception:
        shared['real_noise'] = None
        print("Warning: noise_bank.npy not found; using synthetic vib.")

    # 4. Create per-file datasets (pass arrays to avoid double reads)
    train_ds_list = []
    for arr in temp_train_arrs:
        train_ds_list.append(HybridNoiseDataset([arr], SEQ_LEN, scalers=scalers, shared=shared))

    # Build val arrays similarly (read once)
    val_arrs = []
    for f in val_files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS):
                continue
            arr = df[REQUIRED_COLS].values.astype(np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            valid = (arr[:, 0] != 0) & (arr[:, 1] != 0)
            arr = arr[valid]
            if len(arr) > SEQ_LEN:
                val_arrs.append(arr)
        except Exception:
            pass

    val_ds_list = [HybridNoiseDataset([arr], SEQ_LEN, scalers=scalers, shared=shared) for arr in val_arrs]

    train_ds = ConcatDataset(train_ds_list)
    val_ds = ConcatDataset(val_ds_list)
    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=min(8, os.cpu_count() or 1), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=min(8, os.cpu_count() or 1), pin_memory=True, drop_last=False)

    # Sanity check one sample
    sample_x, sample_y = train_ds[0]
    print("\nDATA SANITY CHECK:")
    print("Input shape (seq_len,input):", sample_x.shape)
    print("Target shape:", sample_y.shape)
    print("Input GPS (first sample, normalized):", sample_x[0, 0:2])
    print("Target (normalized):", sample_y)
    print("Starting training...\n")

    # Model + training setup
    model = TCN(input_size=8, output_size=2, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss()

    history = {'train_loss': [], 'val_mae': []}
    best_mae = float('inf')
    patience_counter = 0

    print("🚀 Training...")
    print(f"{'Epoch':<5} | {'LR':<9} | {'TrainLoss':<10} | {'Val MAE':<9} | Stats")
    print("-" * 80)

    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        avg_train_loss = float(np.mean(batch_losses)) if batch_losses else 0.0

        # Validation
        model.eval()
        val_maes_all = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)

                # pred and y are normalized -> inverse transform to meters safely
                pred_np = pred.cpu().numpy()
                targ_np = y.cpu().numpy()

                pred_m = _safe_inv(target_scaler, pred_np)
                targ_m = _safe_inv(target_scaler, targ_np)

                errs = np.sqrt(np.sum((pred_m - targ_m) ** 2, axis=1))
                val_maes_all.extend(errs)

        errs = np.array(val_maes_all)
        avg_val_mae = float(errs.mean()) if errs.size > 0 else float('nan')

        # Scheduler on MAE
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_train_loss)
        history['val_mae'].append(avg_val_mae)

        stats = f"med:{np.median(errs):.2f} 90p:{np.percentile(errs,90):.2f} %<1m:{np.mean(errs<1.0)*100:.1f}%"
        print(f"{epoch+1:<5} | {current_lr:.2e} | {avg_train_loss:<10.6f} | {avg_val_mae:<9.4f} | {stats}")

        # Checkpoint on MAE
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_by_mae.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early stopping — best MAE: {best_mae:.4f} m")
            break

    torch.save(model.state_dict(), "final_model.pth")
    print("\n✅ Training Complete. Best MAE:", best_mae)

    plt.figure(figsize=(8, 4))
    plt.plot(history['val_mae'], label='Val MAE (m)')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (m)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    main()
