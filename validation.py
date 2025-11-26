#!/usr/bin/env python3
"""
validation.py — Analysis & Visualization Toolkit for TCN GPS-IMU Model

This script loads:
  • best_model.pth
  • scalers.save
  • validation dataset

And performs:
  1. Model summary
  2. Kernel visualization (time + frequency domain)
  3. Activation extraction via forward hooks
  4. Saliency map (gradient-based feature importance)
  5. PCA of layer activations

Place this in the same directory as train.py outputs.
"""

import os
import glob
import joblib
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from train import TCN, HybridNoiseDataset, make_drift_buffer

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
SEQ_LEN = 125
BATCH_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FOLDER = "/content/TEAM_18_E/files/cleaned/test"   # folder containing validation CSVs
NOISE_BANK_PATH = "TEAM_18_E/noise_bank.npy"
MODEL_PATH = "/content/best_model_by_mae.pth"
SCALER_PATH = "/content/scalers.save"
OUTPUT_DIR = "validation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX','IMU_AccY','IMU_AccZ','IMU_GyrX','IMU_GyrY','IMU_GyrZ']

# ------------------------------------------------------
# 1. ROBUST SCALER LOADER
# ------------------------------------------------------
def load_scalers_safe(path, train_files=None, required_cols=REQUIRED_COLS, seq_len=SEQ_LEN):
    """
    Robust scaler loader that handles multiple formats and can recompute if needed.
    """
    # Try joblib first (normal case)
    if os.path.exists(path):
        try:
            obj = joblib.load(path)
            # Accept either direct dict {'imu':..., 'target':...} or a single scaler
            if isinstance(obj, dict) and 'imu' in obj and 'target' in obj:
                print(f"✅ Loaded scalers via joblib from {path}")
                return obj
            else:
                print(f"⚠️ joblib.load returned {type(obj)}, attempting to interpret...")
                # If it's a single scaler object, wrap
                if hasattr(obj, 'transform') and hasattr(obj, 'mean_'):
                    print("Single scaler object detected — using as 'imu' scaler; 'target' will be computed from CSVs.")
                    return {'imu': obj, 'target': None}
        except Exception as e:
            print(f"⚠️ joblib.load failed: {e!r}")
        
        # Try pickle
        try:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and 'imu' in obj and 'target' in obj:
                print(f"✅ Loaded scalers via pickle from {path}")
                return obj
            else:
                print(f"⚠️ pickle.load returned {type(obj)}; interpreting fallback...")
                if hasattr(obj, 'transform') and hasattr(obj, 'mean_'):
                    return {'imu': obj, 'target': None}
        except Exception as e:
            print(f"⚠️ pickle.load failed: {e!r}")
        
        # Try numpy (sometimes people save arrays)
        try:
            arr = np.load(path, allow_pickle=True)
            print("np.load succeeded; inspecting content...")
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                arr0 = arr.item() if arr.size == 1 else arr
                if isinstance(arr0, dict) and 'imu' in arr0 and 'target' in arr0:
                    print(f"✅ Loaded scalers via numpy from {path}")
                    return arr0
        except Exception as e:
            print(f"⚠️ np.load failed: {e!r}")
    
    # Last resort: recompute scalers from training files (slow but reliable)
    print("⚠️ Could not load scalers file. Attempting to compute scalers from provided train_files (this may take a while).")
    if not train_files:
        raise RuntimeError(f"Scalers missing or unreadable at {path} and no train_files provided to recompute them.")
    
    # Gather arrays to fit scalers
    imu_data = []
    rel_samples = []
    for f in train_files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in required_cols):
                continue
            arr = df[required_cols].values.astype(np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            if len(arr) <= seq_len:
                continue
            imu_data.append(arr[:, 2:8])
            
            # Sample a few windows for target scaler
            indices = np.random.randint(0, len(arr)-seq_len, min(50, max(1, len(arr)-seq_len)))
            for i in indices:
                w = arr[i:i+seq_len, 0:2].copy()
                slat, slon = w[0,0], w[0,1]
                mlat = 110649.0
                mlon = 111132.0 * np.cos(np.radians(slat))
                w[:,0] = (w[:,0]-slat)*mlat
                w[:,1] = (w[:,1]-slon)*mlon
                rel_samples.append(w)
        except Exception as e:
            print(f"⚠️ Warning: failed to read {f}: {e}")
    
    if not imu_data or not rel_samples:
        raise RuntimeError("Unable to recompute scalers: no valid data found in train_files.")
    
    imu_all = np.vstack(imu_data)
    imu_scaler = StandardScaler().fit(imu_all)
    target_all = np.vstack(rel_samples)
    target_scaler = StandardScaler().fit(target_all)
    
    print("✅ Recomputed scalers from CSVs. Consider saving them to disk for faster future runs.")
    return {'imu': imu_scaler, 'target': target_scaler}

# ------------------------------------------------------
# 2. Receptive Field Calculator
# ------------------------------------------------------
def receptive_field(kernel_size, dilations):
    """Calculate theoretical receptive field of TCN"""
    return 1 + (kernel_size - 1) * sum(dilations)

def analyze_receptive_field(model, seq_len):
    """Analyze model's receptive field vs sequence length"""
    # Extract dilations from model architecture
    dilations = []
    kernel_size = 3  # default from TCN
    
    for block in model.net:
        if hasattr(block, 'conv1'):
            kernel_size = block.conv1.kernel_size[0]
            dilation = block.conv1.dilation[0]
            dilations.append(dilation)
    
    rf = receptive_field(kernel_size, dilations)
    
    print("\n" + "="*60)
    print("RECEPTIVE FIELD ANALYSIS")
    print("="*60)
    print(f"Kernel Size: {kernel_size}")
    print(f"Dilations: {dilations}")
    print(f"Receptive Field: {rf}")
    print(f"Sequence Length: {seq_len}")
    
    if rf >= seq_len:
        print(f"✅ RF covers full sequence (RF={rf} >= SEQ_LEN={seq_len})")
    else:
        print(f"⚠️ RF does NOT cover full sequence (RF={rf} < SEQ_LEN={seq_len})")
        print(f"   Model can only see {rf}/{seq_len} timesteps")
    
    coverage = min(100, (rf / seq_len) * 100)
    print(f"Coverage: {coverage:.1f}%")
    print("="*60 + "\n")
    
    return rf, dilations

# ------------------------------------------------------
# 3. Utility: Model Summary
# ------------------------------------------------------
def model_summary(model):
    total = 0
    lines = []
    for name, p in model.named_parameters():
        line = f"{name:35s} {tuple(p.shape)}  params={p.numel()}"
        lines.append(line)
        total += p.numel()
    lines.append(f"Total params: {total}")
    text = "\n".join(lines)
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
        f.write(text)
    print(text)

# ------------------------------------------------------
# 4. Kernel Visualization
# ------------------------------------------------------
def plot_conv_kernels(conv_layer, out_dir, max_plots=16, in_ch=0):
    W = conv_layer.weight.detach().cpu().numpy()  # (out_ch, in_ch, k)
    out_ch, _, k = W.shape
    n = min(out_ch, max_plots)
    fig, axs = plt.subplots(nrows=n, figsize=(6, n*1.5))
    if n == 1: axs = [axs]

    for i in range(n):
        axs[i].plot(W[i, in_ch], marker='o')
        axs[i].set_title(f"Kernel out_ch={i}, in_ch={in_ch}")
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "kernels_time.png"))
    plt.close()

# Frequency response
import numpy as np

def plot_kernel_freq(conv_layer, out_dir, in_ch=0):
    W = conv_layer.weight.detach().cpu().numpy()
    N = 512
    fig = plt.figure(figsize=(8,6))
    for i in range(min(6, W.shape[0])):
        fft = np.fft.rfft(W[i, in_ch], n=N)
        mag = np.abs(fft)
        plt.plot(mag, label=f"out{i}")
    plt.legend()
    plt.title("Kernel Frequency Response (Magnitude)")
    plt.savefig(os.path.join(out_dir, "kernels_freq.png"))
    plt.close()

# ------------------------------------------------------
# 5. Activation Extraction via Hooks
# ------------------------------------------------------
def extract_activations(model, x_tensor, block_idx=0):
    act_store = {}

    def hook_fn(m, inp, outp):
        act_store['act'] = outp.detach().cpu()

    handle = model.net[block_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(x_tensor)
    handle.remove()

    act = act_store['act'].squeeze(0).numpy()  # (ch, time)
    fig = plt.figure(figsize=(10,6))
    plt.imshow(act, aspect='auto')
    plt.colorbar()
    plt.title(f"Activations Block {block_idx}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"activations_block_{block_idx}.png"))
    plt.close()

    return act

# ------------------------------------------------------
# 6. Saliency Map (Enhanced with Modality Ablation)
# ------------------------------------------------------
def compute_saliency(model, x_tensor, out_index=0):
    x_var = x_tensor.clone().detach().requires_grad_(True)
    pred = model(x_var)
    scalar = pred[0, out_index]
    model.zero_grad()
    scalar.backward(retain_graph=True)
    grad = x_var.grad.detach().cpu().abs().squeeze(0).numpy()  # (seq, feat)

    plt.figure(figsize=(10,5))
    plt.imshow(grad.T, aspect='auto')
    plt.colorbar()
    plt.title(f"Saliency Map for Output {out_index}")
    plt.xlabel("Time Step")
    plt.ylabel("Feature")
    plt.yticks(range(8), ['GPS_Lat', 'GPS_Lon', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'])
    plt.savefig(os.path.join(OUTPUT_DIR, f"saliency_out{out_index}.png"))
    plt.close()

    return grad

def saliency_by_modality(model, x_tensor, gps_idx=[0,1], imu_idx=list(range(2,8)), out_index=0):
    """
    Ablation saliency: compare GPS vs IMU importance
    Returns total gradient magnitude for each modality
    """
    x = x_tensor.clone().detach().requires_grad_(True)
    pred = model(x)
    scalar = pred[0, out_index]
    model.zero_grad()
    scalar.backward(retain_graph=True)
    g = x.grad.detach().cpu().abs().squeeze(0).numpy()  # (seq, feat)
    
    gps_grad = g[:, gps_idx].sum()
    imu_grad = g[:, imu_idx].sum()
    
    return gps_grad, imu_grad

def analyze_modality_importance(model, val_ds, n_samples=100):
    """
    Analyze GPS vs IMU importance across multiple samples
    """
    gps_grads = []
    imu_grads = []
    
    for i in range(min(n_samples, len(val_ds))):
        x, y = val_ds[i]
        x_tensor = x.unsqueeze(0).to(DEVICE)
        
        gps_g, imu_g = saliency_by_modality(model, x_tensor, out_index=0)
        gps_grads.append(gps_g)
        imu_grads.append(imu_g)
    
    gps_mean = np.mean(gps_grads)
    imu_mean = np.mean(imu_grads)
    total = gps_mean + imu_mean
    
    print("\n" + "="*60)
    print("MODALITY IMPORTANCE ANALYSIS")
    print("="*60)
    print(f"GPS Gradient (avg): {gps_mean:.4f} ({gps_mean/total*100:.1f}%)")
    print(f"IMU Gradient (avg): {imu_mean:.4f} ({imu_mean/total*100:.1f}%)")
    
    if imu_mean < gps_mean * 0.1:
        print("⚠️ WARNING: Model heavily ignores IMU (IMU < 10% of GPS)")
        print("   Consider: increasing IMU weight, checking IMU normalization")
    elif imu_mean > gps_mean * 2:
        print("⚠️ WARNING: Model heavily ignores GPS (GPS < 33% of IMU)")
        print("   Consider: checking GPS normalization, increasing GPS weight")
    else:
        print("✅ Model uses both GPS and IMU reasonably")
    
    print("="*60 + "\n")
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['GPS', 'IMU'], [gps_mean, imu_mean], color=['blue', 'orange'])
    ax.set_ylabel('Average Gradient Magnitude')
    ax.set_title('Modality Importance (Saliency Analysis)')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "modality_importance.png"))
    plt.close()
    
    return gps_mean, imu_mean

# ------------------------------------------------------
# 7. PCA on Activations
# ------------------------------------------------------
def activation_pca(model, dataset, block_idx=1, n_samples=2000):
    activations = []

    def hook_fn(m, inp, outp):
        activations.append(outp.detach().cpu().numpy())

    handle = model.net[block_idx].register_forward_hook(hook_fn)

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    count = 0

    for x, y in loader:
        with torch.no_grad():
            _ = model(x.to(DEVICE))
        count += x.size(0)
        if count >= n_samples:
            break

    handle.remove()

    arr = np.concatenate(activations, axis=0)[:n_samples]   # (N, ch, time)
    N, C, T = arr.shape
    flat = arr.reshape(N, C*T)

    pca = PCA(n_components=2).fit_transform(flat)

    plt.figure(figsize=(8,6))
    plt.scatter(pca[:,0], pca[:,1], s=5, alpha=0.5)
    plt.title(f"Activation PCA (Block {block_idx})")
    plt.savefig(os.path.join(OUTPUT_DIR, f"pca_block_{block_idx}.png"))
    plt.close()

# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
def main():
    print("Loading model + scalers...")
    
    # Get validation files first (needed for fallback scaler computation)
    print("Loading validation data...")
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    if not files:
        print(f"⚠️ No CSV files found in {CSV_FOLDER}")
        return
    
    # Load scalers with robust fallback
    scalers = load_scalers_safe(SCALER_PATH, train_files=files)
    
    # If target scaler is still None, recompute
    if scalers.get('target', None) is None:
        print("⚠️ Target scaler missing from file; recomputing from validation files...")
        scalers = load_scalers_safe(SCALER_PATH, train_files=files)
    
    imu_scaler = scalers['imu']
    target_scaler = scalers['target']
    
    # Load validation data
    val_arrs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if not all(c in df.columns for c in REQUIRED_COLS):
                continue
            arr = df[REQUIRED_COLS].values.astype(np.float32)
            arr = arr[~np.isnan(arr).any(axis=1)]
            valid = (arr[:,0] != 0) & (arr[:,1] != 0)
            arr = arr[valid]
            if len(arr) > SEQ_LEN:
                val_arrs.append(arr)
        except Exception as e:
            print(f"⚠️ Error loading {f}: {e}")

    if not val_arrs:
        print("❌ No valid validation data found!")
        return
    
    print(f"✅ Loaded {len(val_arrs)} validation files")
    
    # Create shared resources
    print("Creating shared noise resources...")
    shared = {
        'drift_lat': make_drift_buffer(200_000),
        'drift_lon': make_drift_buffer(200_000),
        'real_noise': None
    }
    
    try:
        shared['real_noise'] = np.load(NOISE_BANK_PATH)
        print(f"   Loaded {len(shared['real_noise'])} real noise samples.")
    except:
        print("⚠️ Warning: noise_bank.npy not found! Using fallback white noise.")
    
    # Create per-file datasets
    val_ds_list = []
    for f in files:
        try:
            val_ds_list.append(HybridNoiseDataset([f], SEQ_LEN, scalers=scalers, shared=shared))
        except Exception as e:
            print(f"⚠️ Error creating dataset for {f}: {e}")
    
    if not val_ds_list:
        print("❌ No valid datasets created!")
        return
    
    val_ds = torch.utils.data.ConcatDataset(val_ds_list)
    print(f"✅ Created validation dataset with {len(val_ds)} windows")

    # Load model
    model = TCN(input_size=8, output_size=2, num_channels=[64,128,64]).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 1. Model Summary
    model_summary(model)

    # 2. Kernel Visualization
    first_block = model.net[0]
    first_conv = next(m for m in first_block.modules() if isinstance(m, nn.Conv1d))
    plot_conv_kernels(first_conv, OUTPUT_DIR)
    plot_kernel_freq(first_conv, OUTPUT_DIR)

    # 3. Single sample for activation & saliency
    x_sample, y_sample = val_ds[0]
    x_tensor = x_sample.unsqueeze(0).to(DEVICE)

    extract_activations(model, x_tensor, block_idx=0)
    extract_activations(model, x_tensor, block_idx=1)

    compute_saliency(model, x_tensor, out_index=0)
    compute_saliency(model, x_tensor, out_index=1)

    # 4. PCA
    activation_pca(model, val_ds, block_idx=1, n_samples=2000)

    print("Validation analysis complete. Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()