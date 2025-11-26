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
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from train import TCN, HybridNoiseDataset, make_drift_buffer

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
SEQ_LEN = 50
BATCH_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV_FOLDER = "./val_data"   # folder containing validation CSVs
NOISE_BANK_PATH = "./noise_bank.npy"
MODEL_PATH = "best_model.pth"
SCALER_PATH = "scalers.save"
OUTPUT_DIR = "validation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_COLS = ['GPS_Lat', 'GPS_Lng', 'IMU_AccX','IMU_AccY','IMU_AccZ','IMU_GyrX','IMU_GyrY','IMU_GyrZ']

# ------------------------------------------------------
# 1. Utility: Model Summary
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
# 2. Kernel Visualization
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
# 3. Activation Extraction via Hooks
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
# 4. Saliency Map
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
    plt.savefig(os.path.join(OUTPUT_DIR, f"saliency_out{out_index}.png"))
    plt.close()

    return grad

# ------------------------------------------------------
# 5. PCA on Activations
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

    scalers = joblib.load(SCALER_PATH)
    imu_scaler = scalers['imu']
    target_scaler = scalers['target']

    print("Loading validation data...")
    files = glob.glob(os.path.join(CSV_FOLDER, "*.csv"))
    val_arrs = []
    for f in files:
        df = pd.read_csv(f)
        if not all(c in df.columns for c in REQUIRED_COLS):
            continue
        arr = df[REQUIRED_COLS].values.astype(np.float32)
        arr = arr[~np.isnan(arr).any(axis=1)]
        if len(arr) > SEQ_LEN:
            val_arrs.append(arr)

    shared = {
        'drift_lat': make_drift_buffer(200_000),
        'drift_lon': make_drift_buffer(200_000),
        'real_noise': None
    }

    val_ds = torch.utils.data.ConcatDataset([
        HybridNoiseDataset([arr], SEQ_LEN, scalers, shared) for arr in val_arrs
    ])

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