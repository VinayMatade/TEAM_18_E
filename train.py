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
CSV_FOLDER = "/content/fast_data/"
NOISE_BANK_PATH = "/content/noise_bank.npy"
REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX', 'IMU_AccY', 'IMU_AccZ', 'IMU_GyrX', 'IMU_GyrY', 'IMU_GyrZ']

EPOCHS = 50
BATCH_SIZE = 256  # Reduced for stability
SEQ_LEN = 50
LR = 0.001
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set deterministic seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -------------------------
# HELPER: Drift Buffer Generator (Module Level)
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
            buffer[i+j] = c
    return buffer
    
# -------------------------
# 1. DATASET (Fixed Scaler Scope + Shared Resources)
# -------------------------
class HybridNoiseDataset(Dataset):
    def __init__(self, file_list, seq_len, scalers, shared):
        self.seq_len = int(seq_len)
        
        # MUST HAVE SCALERS PASSED IN
        self.imu_scaler = scalers['imu']
        self.target_scaler = scalers['target']
        
        # SHARED RESOURCES (avoid recreating for each file)
        self.drift_buffer_lat = shared['drift_lat']
        self.drift_buffer_lon = shared['drift_lon']
        self.real_noise_bank = shared['real_noise']
        
        # Load and Stack Data
        data_list = []
        if len(file_list) == 1:
            # Single file - silent loading
            pass
        else:
            print(f"   Loading {len(file_list)} logs...")
        
        for f in file_list:
            try:
                df = pd.read_csv(f)
                if not all(c in df.columns for c in REQUIRED_COLS): 
                    continue
                arr = df[REQUIRED_COLS].values.astype(np.float32)
                arr = arr[~np.isnan(arr).any(axis=1)]
                
                # Remove Null Island
                valid = (arr[:,0] != 0) & (arr[:,1] != 0)
                arr = arr[valid]
                
                if len(arr) > seq_len:
                    data_list.append(arr)
            except:
                pass
                
        if not data_list:
            raise ValueError("No valid data loaded!")
            
        self.raw_data = np.vstack(data_list)
        self.n_samples = len(self.raw_data) - self.seq_len

        # Pre-transform IMU
        self.norm_imu = self.imu_scaler.transform(self.raw_data[:, 2:8]).astype(np.float32)
        self.raw_gps = self.raw_data[:, 0:2]

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        # A. Get Window
        imu_window = self.norm_imu[idx : idx + self.seq_len]
        gps_window = self.raw_gps[idx : idx + self.seq_len].copy()
        
        # B. Make GPS Relative (Meters)
        start_lat = gps_window[0, 0]
        start_lon = gps_window[0, 1]
        
        lat_rad = np.radians(start_lat)
        m_per_deg_lat = 110649.0
        m_per_deg_lon = 111132.0 * np.cos(lat_rad)
        
        gps_window[:, 0] -= start_lat
        gps_window[:, 1] -= start_lon
        gps_window[:, 0] *= m_per_deg_lat
        gps_window[:, 1] *= m_per_deg_lon
        
        clean_gps_m = gps_window
        
        # C. Add Noise (Meters) - FIXED: Safe randint
        n_rows = self.seq_len
        
        # Safe randint for drift
        max_start = max(1, len(self.drift_buffer_lat) - n_rows + 1)
        d_start = np.random.randint(0, max_start)
        drift = np.stack([
            self.drift_buffer_lat[d_start:d_start+n_rows],
            self.drift_buffer_lon[d_start:d_start+n_rows]
        ], axis=1)
        
        # Safe randint for real_noise_bank
        if self.real_noise_bank is not None and len(self.real_noise_bank) >= n_rows:
            max_start = len(self.real_noise_bank) - n_rows + 1
            r = np.random.randint(0, max_start)
            vib = self.real_noise_bank[r:r+n_rows]
        else:
            vib = np.random.normal(0, 0.05, (n_rows, 2))
             
        total_noise_m = drift + vib
        noisy_gps_m = clean_gps_m + total_noise_m
        
        # D. Normalize GPS using target_scaler (FIXED)
        noisy_gps_norm = self.target_scaler.transform(noisy_gps_m).astype(np.float32)
        
        # E. Prepare Target (Clean Relative Meters, Normalized)
        target_norm = self.target_scaler.transform(clean_gps_m[-1:]).flatten().astype(np.float32)
        
        # F. Stack Input: [NormLat, NormLon, NormIMU...] (8 features total)
        x_full = np.concatenate([noisy_gps_norm, imu_window], axis=1)
        
        return torch.tensor(x_full, dtype=torch.float32), torch.tensor(target_norm, dtype=torch.float32)

# -------------------------
# 2. MODEL
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
# 3. MAIN (FIXED SCALER SCOPE)
# -------------------------
def main():
    print(f"Running on {DEVICE}")
    
    # 1. File Split
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    print(f"Found {len(files)} logs in {CSV_FOLDER}")
    np.random.shuffle(files)
    split = int(len(files) * 0.8)
    train_files = files[:split]
    val_files = files[split:]
    
    print(f"Train Files: {len(train_files)}, Val Files: {len(val_files)}")
    
    # 2. FIT SCALERS MANUALLY BEFORE DATASET CREATION (CRITICAL FIX)
    print("Fitting scalers (this may take a moment)...")
    
    # Load raw data just to fit scalers
    temp_data_list = []
    for f in train_files:
        try:
            df = pd.read_csv(f)
            if all(c in df.columns for c in REQUIRED_COLS):
                temp_data_list.append(df[REQUIRED_COLS].values)
        except:
            pass
    
    if not temp_data_list:
        print("❌ No valid training data found!")
        return
    
    all_train_arr = np.vstack(temp_data_list)
    
    # A. IMU Scaler
    imu_scaler = StandardScaler().fit(all_train_arr[:, 2:8])
    
    # B. Target Scaler (Relative Meters) - CRITICAL FIX
    print("   Computing relative GPS stats...")
    rel_samples = []
    for arr in temp_data_list:
        if len(arr) <= SEQ_LEN:
            continue
        # Sample 100 windows
        indices = np.random.randint(0, len(arr) - SEQ_LEN, min(100, len(arr)-SEQ_LEN))
        for i in indices:
            window = arr[i : i+SEQ_LEN, 0:2].copy()
            start_lat, start_lon = window[0,0], window[0,1]
            lat_rad = np.radians(start_lat)
            mlat = 110649.0
            mlon = 111132.0 * np.cos(lat_rad)
            window[:,0] = (window[:,0]-start_lat)*mlat
            window[:,1] = (window[:,1]-start_lon)*mlon
            rel_samples.append(window)
    
    target_scaler = StandardScaler().fit(np.vstack(rel_samples))
    print(f"   Target Scaler Mean: {target_scaler.mean_}, Var: {target_scaler.var_}")
    
    # Pack scalers
    scalers = {'imu': imu_scaler, 'target': target_scaler}
    joblib.dump(scalers, "scalers.save")
    print("✅ Scalers saved.")
    
    # 3. CREATE SHARED RESOURCES (one-time, reused across all datasets)
    print("Creating shared noise resources...")
    shared = {}
    shared['drift_lat'] = make_drift_buffer(size=200_000)
    shared['drift_lon'] = make_drift_buffer(size=200_000)
    try:
        shared['real_noise'] = np.load(NOISE_BANK_PATH)
        print(f"   Loaded {len(shared['real_noise'])} real noise samples.")
    except:
        print("⚠️ Warning: noise_bank.npy not found! Using fallback white noise.")
        shared['real_noise'] = None
    
    # 4. CREATE DATASETS - FIXED: Per-file to prevent train/val leakage
    print("Creating per-file datasets to prevent cross-file windows...")
    train_ds_list = []
    for f in train_files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS):
                continue
            arr = df[REQUIRED_COLS].values
            arr = arr[~np.isnan(arr).any(axis=1)]
            valid = (arr[:,0] != 0) & (arr[:,1] != 0)
            arr = arr[valid]
            if len(arr) > SEQ_LEN:
                # Pass single-file list to prevent cross-file windows
                train_ds_list.append(HybridNoiseDataset([f], SEQ_LEN, scalers=scalers, shared=shared))
        except:
            pass
    
    val_ds_list = []
    for f in val_files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS):
                continue
            arr = df[REQUIRED_COLS].values
            arr = arr[~np.isnan(arr).any(axis=1)]
            valid = (arr[:,0] != 0) & (arr[:,1] != 0)
            arr = arr[valid]
            if len(arr) > SEQ_LEN:
                val_ds_list.append(HybridNoiseDataset([f], SEQ_LEN, scalers=scalers, shared=shared))
        except:
            pass
    
    train_ds = ConcatDataset(train_ds_list)
    val_ds = ConcatDataset(val_ds_list)
    print(f"   Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, drop_last=False)
    
    # 4. SANITY CHECK
    sample_x, sample_y = train_ds[0]
    print("\n🔍 DATA SANITY CHECK:")
    print(f"Input Shape: {sample_x.shape}")
    print(f"Target Shape: {sample_y.shape}")
    print(f"Input GPS Range (Normalized): Min={sample_x[:, 0:2].min():.4f}, Max={sample_x[:, 0:2].max():.4f}")
    print(f"Target GPS (Normalized): {sample_y}")
    print("✅ Data looks good. Starting training...\n")
    
    # 5. MODEL
    model = TCN(input_size=8, output_size=2, num_channels=[64, 128, 64], kernel_size=3, dropout=0.2)
    model.to(DEVICE)
    
    # FIXED: Use AdamW + SmoothL1Loss + Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion = nn.SmoothL1Loss()  # More robust to outliers, better for MAE
    
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    best_mae = float('inf')
    patience_counter = 0
    
    print("🚀 Starting Training...")
    print(f"{'Epoch':<5} | {'LR':<9} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val MAE':<10} | {'Stats':<30}")
    print("-" * 100)
    
    for epoch in range(EPOCHS):
        model.train()
        batch_losses = []
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # FIXED: Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
            
        avg_train_loss = np.mean(batch_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
                
                # INVERSE TRANSFORM (FIXED - target_scaler now in scope)
                pred_np = pred.cpu().numpy()
                targ_np = y.cpu().numpy()
                
                # Convert back to Meters using the scaler
                pred_m = target_scaler.inverse_transform(pred_np)
                targ_m = target_scaler.inverse_transform(targ_np)
                
                # Calc MAE
                err = np.sqrt(np.sum((pred_m - targ_m)**2, axis=1))
                val_maes.extend(err)
                
        avg_val_loss = np.mean(val_losses)
        errs = np.array(val_maes)
        avg_val_mae = errs.mean()
        
        # FIXED: Schedule on MAE and get current LR
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        # FIXED: Richer validation stats
        stats = f"med:{np.median(errs):.2f} 90p:{np.percentile(errs,90):.2f} <1m:{np.mean(errs<1.0)*100:.1f}%"
        print(f"{epoch+1:<5} | {current_lr:.2e}  | {avg_train_loss:.6f}   | {avg_val_loss:.6f}   | {avg_val_mae:.4f}m    | {stats}")

        # FIXED: Early Stopping on MAE
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            patience_counter = 0
            torch.save(model.state_dict(), "best_model_by_mae.pth")
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print(f"\n⏹️ Early Stopping Triggered! Best MAE: {best_mae:.4f}m")
            break

    torch.save(model.state_dict(), "final_model.pth")
    print("\n✅ Training Complete.")
    
    plt.figure(figsize=(10,5))
    plt.plot(history['val_mae'], label='Val MAE (meters)', color='green')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (meters)')
    plt.legend()
    plt.savefig('training_curve.png')

if __name__ == "__main__":
    main()
