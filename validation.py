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
# Removed load_scalers_safe function - no longer needed since train.py saves all 4 scalers

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
    # 10 features: gps_norm(2) + delta_norm(2) + norm_imu(6)
    plt.yticks(range(10), ['GPS_Lat', 'GPS_Lon', 'Δ_Lat', 'Δ_Lon', 'AccX', 'AccY', 'AccZ', 'GyrX', 'GyrY', 'GyrZ'])
    plt.savefig(os.path.join(OUTPUT_DIR, f"saliency_out{out_index}.png"))
    plt.close()

    return grad

def saliency_by_modality(model, x_tensor, gps_idx=[0,1,2,3], imu_idx=list(range(4,10)), out_index=0):
    """
    Ablation saliency: compare GPS vs IMU importance
    Returns total gradient magnitude for each modality
    
    Feature indices (10 total):
    - GPS: [0,1] = gps_norm, [2,3] = delta_norm
    - IMU: [4,5,6,7,8,9] = norm_imu (AccX, AccY, AccZ, GyrX, GyrY, GyrZ)
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
        x, y, raw_acc = val_ds[i]  # Updated to match HybridNoiseDataset output
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

    for batch in loader:
        x, y, raw_acc = batch  # Updated to match HybridNoiseDataset output
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
    
    # Load scalers (matching train.py: imu, gps_point, target, delta)
    scalers = joblib.load(SCALER_PATH)
    
    if not all(k in scalers for k in ['imu', 'gps_point', 'target', 'delta']):
        print("❌ Missing required scalers! Expected: imu, gps_point, target, delta")
        print(f"   Found: {list(scalers.keys())}")
        return
    
    print(f"✅ Loaded scalers: {list(scalers.keys())}")
    
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

    # Load model (matching train.py parameters)
    model = TCN(
        input_size=10,  # gps_norm(2) + delta_norm(2) + norm_imu(6) = 10
        output_size=2, 
        num_channels=[128, 128, 128, 128, 128, 128], 
        kernel_size=7, 
        dropout=0.3,
        dilations=[1, 2, 4, 8, 16, 32]
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 1. Model Summary
    model_summary(model)

    # 2. Kernel Visualization
    first_block = model.net[0]
    first_conv = next(m for m in first_block.modules() if isinstance(m, nn.Conv1d))
    plot_conv_kernels(first_conv, OUTPUT_DIR)
    plot_kernel_freq(first_conv, OUTPUT_DIR)

    # 3. Receptive Field Analysis
    analyze_receptive_field(model, SEQ_LEN)

    # 4. Single sample for activation & saliency
    x_sample, y_sample, raw_acc_sample = val_ds[0]
    x_tensor = x_sample.unsqueeze(0).to(DEVICE)

    extract_activations(model, x_tensor, block_idx=0)
    extract_activations(model, x_tensor, block_idx=1)

    compute_saliency(model, x_tensor, out_index=0)
    compute_saliency(model, x_tensor, out_index=1)

    # 5. Modality Importance Analysis
    analyze_modality_importance(model, val_ds, n_samples=100)

    # 6. PCA
    activation_pca(model, val_ds, block_idx=1, n_samples=2000)

    print("Validation analysis complete. Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()