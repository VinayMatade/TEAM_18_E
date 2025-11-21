#!/usr/bin/env python3
"""
TCN (Temporal Convolutional Network) Training Script for Time Series Prediction

This script trains a TCN model on one or multiple CSV files containing time series data.
It supports adaptive sequence lengths, multi-file training with global scaling, and
comprehensive metrics tracking.
"""

# Standard library imports
import argparse
import glob
import os
import sys
from typing import List, Tuple, Optional, Dict

# Third-party imports for data processing and visualization
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Scikit-learn imports for preprocessing and metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PyTorch imports for deep learning
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# TCN building blocks
# -------------------------
class Chomp1d(nn.Module):
    """
    Chomping layer to remove padding from the end of the sequence.
    
    This is used in causal convolutions to ensure that the output at time t
    only depends on inputs up to time t (not future values), 
    i.e., at t, data only upto T-1 is used 
    """
    def __init__(self, chomp_size: int):
        """
        Initialize the Chomp1d layer.
        
        Args:
            chomp_size: Number of elements to remove from the end of the sequence
        """
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        """
        Forward pass: remove the last chomp_size elements from the sequence.
        
        Args:
            x: Input tensor of shape (batch, channels, seq_len)
            
        Returns:
            Tensor with last chomp_size elements removed
        """
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single temporal block in the TCN architecture.
    
    This block consists of two dilated causal convolutions with residual connections.
    Each convolution is followed by chomping, ReLU activation, and dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        """
        Initialize a temporal block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride for convolution
            dilation: Dilation factor for dilated convolution
            padding: Padding size (will be chomped to maintain causality)
            dropout: Dropout probability for regularization
        """
        super().__init__()
        # First convolutional layer with dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Remove padding to maintain causality
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second convolutional layer with dilation
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)  # Remove padding to maintain causality
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Sequential network combining all layers
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # Downsample residual connection if channel dimensions don't match
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        """Initialize convolutional weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        """
        Forward pass with residual connection.
        
        Args:
            x: Input tensor of shape (batch, in_channels, seq_len)
            
        Returns:
            Output tensor of shape (batch, out_channels, seq_len)
        """
        out = self.net(x)
        # Apply residual connection (with optional downsampling)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series prediction.
    
    The TCN uses stacked temporal blocks with exponentially increasing dilation
    to capture long-range dependencies in time series data. The final prediction
    is made using the last time step's representation.
    """
    def __init__(self, input_size: int, output_size: int, num_channels: List[int],
                 kernel_size: int = 3, dropout: float = 0.2):
        """
        Initialize the TCN model.
        
        Args:
            input_size: Number of input features per time step
            output_size: Number of output features to predict
            num_channels: List of channel sizes for each temporal block
            kernel_size: Size of convolutional kernels (default: 3)
            dropout: Dropout probability for regularization (default: 0.2)
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        # Build stacked temporal blocks with exponentially increasing dilation
        for i in range(num_levels):
            # First layer takes input_size channels, subsequent layers take previous layer's output
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            # Exponentially increase dilation: 1, 2, 4, 8, ...
            dilation_size = 2 ** i
            # Calculate padding to maintain sequence length
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, stride=1,
                                        dilation=dilation_size, padding=padding, dropout=dropout))
        
        self.network = nn.Sequential(*layers)
        # Final linear layer to map from last temporal block's channels to output size
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """
        Forward pass through the TCN.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Predictions of shape (batch, output_size)
        """
        # Transpose to (batch, features, seq_len) for Conv1d
        x = x.transpose(1, 2)
        # Pass through temporal blocks: (batch, channels, seq_len)
        y = self.network(x)
        # Take only the last time step: (batch, channels)
        y = y[:, :, -1]
        # Map to output size: (batch, output_size)
        return self.linear(y)


# -------------------------
# Dataset: sliding windows
# -------------------------
class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset for creating sliding windows from time series data.
    
    This dataset creates overlapping windows of historical data (input) and
    corresponding future values (target) for time series prediction.
    """
    def __init__(self, data_array: np.ndarray, seq_len: int, pred_len: int, target_indices: List[int]):
        """
        Initialize the sliding window dataset.
        
        Args:
            data_array: Numpy array of shape (time_steps, features)
            seq_len: Length of input sequence (lookback window)
            pred_len: Length of prediction horizon (how many steps ahead to predict)
            target_indices: Indices of features to predict (subset of all features)
            
        Raises:
            ValueError: If data is too short for the specified window sizes
        """
        self.data = data_array
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.target_indices = list(target_indices)
        # Calculate number of valid windows that can be created
        self.n_samples = len(self.data) - self.seq_len - self.pred_len + 1
        if self.n_samples <= 0:
            raise ValueError("Not enough data for seq_len/pred_len -> n_samples <= 0")

    def __len__(self):
        """Return the total number of samples (windows) in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get a single sample (input window and target).
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple of (input_tensor, target_tensor):
                - input_tensor: shape (seq_len, features)
                - target_tensor: shape (pred_len, target_features) or (target_features,) if pred_len=1
        """
        # Extract input sequence: seq_len consecutive time steps
        x = self.data[idx: idx + self.seq_len]  # (seq_len, features)
        # Extract target: pred_len future time steps, only for target features
        y = self.data[idx + self.seq_len: idx + self.seq_len + self.pred_len, self.target_indices]
        # If predicting only 1 step ahead, squeeze the time dimension
        if self.pred_len == 1:
            y = y[0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -------------------------
# Helpers: load + cleaning
# -------------------------
def load_and_clean_csv(csv_path: str, time_col: str = "timestamp") -> pd.DataFrame:
    """
    Load a CSV file and clean it for time series processing.
    
    This function performs several cleaning steps:
    1. Removes BOM characters from column names
    2. Sorts data by time column
    3. Extracts only numeric columns
    4. Interpolates missing values
    5. Handles infinite values
    
    Args:
        csv_path: Path to the CSV file
        time_col: Name of the timestamp column for sorting (case-insensitive)
        
    Returns:
        DataFrame containing only numeric columns, sorted by time, with cleaned values
        
    Raises:
        ValueError: If the time column is not found in the CSV
    """
    # Load CSV file
    df = pd.read_csv(csv_path)
    # Clean column names: strip whitespace and remove BOM (Byte Order Mark) characters
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    
    # Find time column (case-insensitive matching)
    if time_col not in df.columns:
        matches = [c for c in df.columns if c.lower() == time_col.lower()]
        if matches:
            time_col = matches[0]
    if time_col not in df.columns:
        raise ValueError(f"time column '{time_col}' not found in {csv_path}. Available: {list(df.columns)}")
    
    # Sort by time column to ensure chronological order
    df = df.sort_values(by=time_col).reset_index(drop=True)
    
    # Extract only numeric columns for processing
    df_numeric = df.select_dtypes(include=[np.number]).copy()
    
    # Interpolate missing values bidirectionally, then forward/backward fill any remaining
    df_numeric = df_numeric.interpolate(limit_direction='both').ffill().bfill()
    
    # Replace infinite values with NaN
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Final fallback: fill any remaining NaN with 0.0 (rare edge case)
    df_numeric = df_numeric.fillna(0.0)
    
    return df_numeric


def stabilize_zero_variance(array: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, List[int]]:
    """
    Add tiny noise to zero-variance columns to prevent scaling issues.
    
    StandardScaler will fail or produce NaN/Inf values when a column has zero variance.
    This function detects such columns and adds minimal Gaussian noise to stabilize them.
    
    Args:
        array: Input array of shape (samples, features)
        eps: Standard deviation of noise to add (default: 1e-6)
        
    Returns:
        Tuple of (stabilized_array, list_of_fixed_column_indices)
    """
    # Calculate standard deviation for each column
    stds = np.nanstd(array, axis=0)
    # Identify columns with zero variance
    zero_mask = stds == 0
    fixed = []
    
    if np.any(zero_mask):
        # Get indices of zero-variance columns
        fixed = np.where(zero_mask)[0].tolist()
        # Add tiny Gaussian noise to each zero-variance column
        for c in fixed:
            array[:, c] = array[:, c] + np.random.normal(0.0, eps, size=array.shape[0])
    
    return array, fixed


# -------------------------
# Metrics utilities
# -------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics for predictions.
    
    Calculates MAE, RMSE, R², and MAP% (Mean Accuracy Percentage) metrics
    with robust error handling.
    
    Args:
        y_true: Ground truth values (any shape)
        y_pred: Predicted values (same shape as y_true)
        
    Returns:
        Dictionary containing:
            - MAE: Mean Absolute Error
            - RMSE: Root Mean Squared Error
            - R2: R-squared (coefficient of determination)
            - MAP%: Mean Accuracy Percentage (100% - normalized MAE)
    """
    # Flatten arrays to 1D for metric calculation
    try:
        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)
    except Exception:
        return {"MAE": None, "RMSE": None, "R2": None, "MAP%": None}
    
    # Check for empty arrays
    if yt.size == 0 or yp.size == 0:
        return {"MAE": None, "RMSE": None, "R2": None, "MAP%": None}
    
    # Calculate MAE with fallback to manual computation
    try:
        mae = float(mean_absolute_error(yt, yp))
    except Exception:
        mae = float(np.mean(np.abs(yt - yp)))
    
    # Calculate RMSE with fallback to manual computation
    try:
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    except Exception:
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    
    # Calculate R² with fallback to NaN on error
    try:
        r2 = float(r2_score(yt, yp))
    except Exception:
        r2 = float(np.nan)
    
    # Calculate MAP% (Mean Accuracy Percentage)
    # This represents accuracy as a percentage of the value range
    val_range = float(np.max(yt) - np.min(yt)) if yt.size > 0 else 0.0
    map_pct = 0.0 if val_range == 0.0 else float(max(0.0, 100.0 * (1.0 - mae / val_range)))
    
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAP%": map_pct}


# -------------------------
# Train single file (robust)
# -------------------------
def train_one_file(model: nn.Module,
                   file_path: str,
                   global_feature_scaler: Optional[StandardScaler],
                   args,
                   device: torch.device,
                   optimizer=None):
    """
    Train the model on a single CSV file with comprehensive error handling.
    
    This function handles the complete training pipeline for one file:
    1. Load and clean the CSV data
    2. Apply adaptive sequence length if configured
    3. Split into train/validation sets
    4. Scale features using global scaler
    5. Create sliding window datasets
    6. Train with early stopping and learning rate scheduling
    7. Compute metrics and save loss curves
    
    Args:
        model: PyTorch model to train
        file_path: Path to the CSV file
        global_feature_scaler: Pre-fitted StandardScaler for feature normalization
        args: Argument namespace containing all hyperparameters
        device: PyTorch device (CPU or CUDA)
        optimizer: Optional optimizer (creates new one if None)
        
    Returns:
        Tuple containing:
            - model: Updated model
            - scaler: Feature scaler used
            - last_train_loss: Final training loss
            - best_val_loss: Best validation loss achieved
            - n_train_windows: Number of training windows
            - n_val_windows: Number of validation windows
            - metrics_dict: Dictionary of evaluation metrics
            - (train_curve, val_curve): Loss history for plotting
    """
    # ===== Step 1: Load and validate CSV data =====
    try:
        df_numeric = load_and_clean_csv(file_path, time_col=args.time_col)
    except Exception as e:
        print(f"⚠️ Skipping {os.path.basename(file_path)} — load error: {e}")
        return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])

    # Check if we have any numeric columns
    if df_numeric.shape[1] == 0:
        print(f"⚠️ Skipping {os.path.basename(file_path)} — no numeric columns.")
        return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])

    # If user specified a target column, use only that column
    if args.target:
        if args.target not in df_numeric.columns:
            print(f"⚠️ Skipping {os.path.basename(file_path)} — target '{args.target}' not found.")
            return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])
        df_numeric = df_numeric[[args.target]]

    # Convert to numpy array for processing
    data_array = df_numeric.values.astype(float)
    num_rows = len(data_array)

    # ===== Step 2: Determine adaptive sequence length =====
    # If seq_len is 0, compute it adaptively as 10% of file length (clamped to min/max)
    if args.seq_len == 0:
        computed_len = max(1, int(max(1, num_rows * 0.1)))  # 10% of rows
        seq_len_for_file = int(max(args.min_seq_len, min(args.max_seq_len, computed_len)))
    else:
        # Use fixed sequence length from arguments
        seq_len_for_file = int(max(1, args.seq_len))

    print(f"Adaptive seq-len for {os.path.basename(file_path)} -> {seq_len_for_file} (rows={num_rows})")

    # Verify we have enough data for the sequence and prediction lengths
    if num_rows < seq_len_for_file + args.pred_len:
        print(f"⚠️ Skipping {os.path.basename(file_path)}: too short for seq_len={seq_len_for_file}, pred_len={args.pred_len}")
        return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])

    # ===== Step 3: Split data into train and validation sets =====
    val_count = max(1, int(num_rows * args.test_split))
    train_array = data_array[:-val_count]  # All but last val_count rows
    val_array = data_array[-val_count:]     # Last val_count rows

    # ===== Step 4: Stabilize zero-variance columns =====
    # Add tiny noise to prevent scaling issues with constant columns
    train_array, fixed_columns = stabilize_zero_variance(train_array, eps=1e-6)
    if fixed_columns:
        print(f"ℹ️ {os.path.basename(file_path)}: tiny noise added to zero-variance columns: {fixed_columns}")

    # ===== Step 5: Apply feature scaling =====
    # Use global scaler if provided, otherwise fit a new one (not recommended)
    if global_feature_scaler is None:
        global_feature_scaler = StandardScaler().fit(train_array)
        print("ℹ️ Fitted fallback global scaler from this file (not recommended)")

    # Transform both train and validation data using the scaler
    try:
        train_scaled = global_feature_scaler.transform(train_array)
        val_scaled = global_feature_scaler.transform(val_array)
    except Exception as e:
        print(f"⚠️ Scaling error for {os.path.basename(file_path)}: {e}")
        return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])

    # Verify all values are finite (no NaN or Inf)
    if not np.isfinite(train_scaled).all() or not np.isfinite(val_scaled).all():
        print(f"⚠️ Skipping {os.path.basename(file_path)}: non-finite values after scaling.")
        return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])

    # All features are targets (predict all columns)
    target_indices = list(range(train_scaled.shape[1]))

    # ===== Step 6: Create sliding window datasets =====
    try:
        train_dataset = SlidingWindowDataset(train_scaled, seq_len_for_file, args.pred_len, target_indices)
        val_dataset = SlidingWindowDataset(val_scaled, seq_len_for_file, args.pred_len, target_indices)
    except Exception as e:
        print(f"⚠️ Skipping {os.path.basename(file_path)}: cannot build sliding windows: {e}")
        return model, global_feature_scaler, 0.0, 0.0, 0, 0, None, ([], [])

    # Verify we have at least some windows
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print(f"⚠️ Skipping {os.path.basename(file_path)}: zero windows (train={len(train_dataset)}, val={len(val_dataset)})")
        return model, global_feature_scaler, 0.0, 0.0, len(train_dataset), len(val_dataset), None, ([], [])

    # Create data loaders for batching
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ===== Step 7: Setup optimizer and scheduler =====
    # Optionally reset optimizer for each file (fresh start)
    if args.reset_optimizer_per_file or optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create learning rate scheduler (reduces LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.scheduler_patience
    )

    # ===== Step 8: Training loop with early stopping =====
    model.to(device)
    best_val_loss = float("inf")
    best_epoch = 0
    no_improvement_count = 0
    train_loss_history, val_loss_history = [], []
    map_pct_history = []  # Track MAP% per epoch to monitor overfitting

    for epoch in range(1, args.epochs + 1):
        # ----- Training phase -----
        model.train()
        batch_loss_values = []

        for xb, yb in train_loader:
            # Move batch to device (GPU/CPU)
            xb, yb = xb.to(device), yb.to(device)
            
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(xb)
            
            # Handle shape mismatch if predicting single feature
            if predictions.shape[-1] == 1 and yb.dim() == 1:
                predictions = predictions.squeeze(-1)
            
            # Compute loss
            loss_value = args.loss_fn(predictions, yb)
            
            # Check for numerical instability
            if torch.isnan(loss_value) or torch.isinf(loss_value):
                print(f"⚠️ NaN/Inf loss at {os.path.basename(file_path)} epoch {epoch}. Aborting file.")
                return model, global_feature_scaler, 0.0, 0.0, len(train_dataset), len(val_dataset), None, ([], [])
            
            # Backward pass
            loss_value.backward()
            
            # Gradient clipping to prevent exploding gradients
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            # Update weights
            optimizer.step()
            batch_loss_values.append(float(loss_value.item()))

        # Calculate average training loss for this epoch
        epoch_train_loss = float(np.mean(batch_loss_values)) if batch_loss_values else 0.0
        train_loss_history.append(epoch_train_loss)

        # ----- Validation phase -----
        model.eval()
        val_batch_losses = []
        preds_on_val = []
        trues_on_val = []
        
        # Disable gradient computation for validation (saves memory and computation)
        with torch.no_grad():
            for xb, yb in val_loader:
                # Move batch to device
                xb, yb = xb.to(device), yb.to(device)
                
                # Forward pass
                predictions = model(xb)
                
                # Handle shape mismatch
                if predictions.shape[-1] == 1 and yb.dim() == 1:
                    predictions = predictions.squeeze(-1)
                
                # Compute validation loss
                val_loss_val = float(args.loss_fn(predictions, yb).item())
                val_batch_losses.append(val_loss_val)

                # Store predictions and ground truth for metrics calculation
                np_pred = predictions.detach().cpu().numpy()
                np_true = yb.detach().cpu().numpy()
                
                # Ensure 2D shape for consistent stacking
                if np_pred.ndim == 1:
                    np_pred = np_pred.reshape(-1, 1)
                if np_true.ndim == 1:
                    np_true = np_true.reshape(-1, 1)
                preds_on_val.append(np_pred)
                trues_on_val.append(np_true)

        # Calculate average validation loss for this epoch
        epoch_val_loss = float(np.mean(val_batch_losses)) if val_batch_losses else float("inf")
        val_loss_history.append(epoch_val_loss)

        # ----- Calculate MAP% for this epoch -----
        # Compute MAP% to monitor overfitting (similar to mAP in YOLO)
        epoch_map_pct = None
        try:
            if preds_on_val and trues_on_val:
                # Concatenate all predictions and ground truth from this epoch
                epoch_preds = np.vstack(preds_on_val) if len(preds_on_val) > 1 else preds_on_val[0]
                epoch_trues = np.vstack(trues_on_val) if len(trues_on_val) > 1 else trues_on_val[0]
                
                # Inverse transform to original scale for meaningful MAP%
                if global_feature_scaler is not None:
                    epoch_preds_inv = global_feature_scaler.inverse_transform(epoch_preds)
                    epoch_trues_inv = global_feature_scaler.inverse_transform(epoch_trues)
                else:
                    epoch_preds_inv = epoch_preds
                    epoch_trues_inv = epoch_trues
                
                # Calculate MAP% for this epoch
                epoch_metrics = compute_metrics(epoch_trues_inv, epoch_preds_inv)
                epoch_map_pct = epoch_metrics.get("MAP%", None)
        except Exception:
            epoch_map_pct = None
        
        # Store MAP% in history
        map_pct_history.append(epoch_map_pct if epoch_map_pct is not None else 0.0)

        # ----- Learning rate scheduling -----
        # Reduce learning rate if validation loss plateaus
        try:
            scheduler.step(epoch_val_loss)
        except Exception:
            pass

        # ----- Early stopping logic -----
        # Save model if validation loss improved
        if epoch_val_loss < best_val_loss - 1e-12:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            no_improvement_count = 0
            try:
                torch.save({'model_state_dict': model.state_dict()}, args.save)
            except Exception as e:
                print(f"⚠️ Could not save model snapshot: {e}")
        else:
            no_improvement_count += 1

        # Print epoch summary with MAP% to monitor overfitting
        if epoch_map_pct is not None:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f} | MAP%: {epoch_map_pct:.2f}%")
        else:
            print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

        # Stop training if no improvement for 'patience' epochs
        if no_improvement_count >= args.patience:
            print(f"Early stopping for {os.path.basename(file_path)} after {epoch} epochs (best epoch = {best_epoch}, best val = {best_val_loss:.6f}).")
            break

    # ===== Step 9: Concatenate all validation predictions =====
    if preds_on_val:
        try:
            preds_arr = np.vstack(preds_on_val)
            trues_arr = np.vstack(trues_on_val)
        except Exception:
            preds_arr = np.concatenate(preds_on_val, axis=0)
            trues_arr = np.concatenate(trues_on_val, axis=0)
    else:
        preds_arr = np.empty((0, train_scaled.shape[1]))
        trues_arr = np.empty((0, train_scaled.shape[1]))

    # ===== Step 10: Compute evaluation metrics =====
    # Inverse-transform predictions back to original scale for meaningful metrics
    metrics_dict = None
    try:
        if preds_arr.size and global_feature_scaler is not None:
            # Transform back to original units
            preds_inv = global_feature_scaler.inverse_transform(preds_arr)
            trues_inv = global_feature_scaler.inverse_transform(trues_arr)
            metrics_dict = compute_metrics(trues_inv, preds_inv)
        else:
            # Use scaled values if inverse transform not possible
            metrics_dict = compute_metrics(trues_arr, preds_arr) if preds_arr.size else None
    except Exception:
        metrics_dict = compute_metrics(trues_arr, preds_arr) if preds_arr.size else None

    # ===== Step 11: Save loss curve and MAP% plots =====
    os.makedirs("plots", exist_ok=True)
    try:
        # Create figure with two subplots: Loss and MAP%
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Loss curves
        ax1.plot(range(1, len(train_loss_history) + 1), train_loss_history, label="Train Loss", color='blue')
        ax1.plot(range(1, len(val_loss_history) + 1), val_loss_history, label="Val Loss", color='orange')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Loss Curve - {os.path.basename(file_path)}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: MAP% curve (overfitting indicator)
        if map_pct_history and any(v > 0 for v in map_pct_history):
            ax2.plot(range(1, len(map_pct_history) + 1), map_pct_history, label="MAP%", color='green', linewidth=2)
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("MAP% (Mean Accuracy Percentage)")
            ax2.set_title("MAP% - Overfitting Monitor")
            ax2.set_ylim([0, 105])  # MAP% ranges from 0-100%
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            # Add horizontal line at best MAP%
            if map_pct_history:
                best_map = max(map_pct_history)
                ax2.axhline(y=best_map, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_map:.2f}%')
                ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'MAP% not available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title("MAP% - Overfitting Monitor")
        
        plt.tight_layout()
        plot_file = os.path.join("plots", f"{os.path.basename(file_path).replace('.csv','')}_metrics.png")
        plt.savefig(plot_file, dpi=150)
        plt.close()
        print(f"📈 Saved metrics plot to: {plot_file}")
    except Exception as e:
        print(f"⚠️ Could not save plot for {file_path}: {e}")

    # Return all results
    return model, global_feature_scaler, train_loss_history[-1] if train_loss_history else 0.0, best_val_loss, len(train_dataset), len(val_dataset), metrics_dict, (train_loss_history, val_loss_history)

# -------------------------
# Main orchestration
# -------------------------
def main():
    """
    Main entry point for the training script.
    
    This function orchestrates the entire training pipeline:
    1. Parse command-line arguments
    2. Discover CSV files to train on
    3. Build a global feature scaler from all files
    4. Initialize the TCN model
    5. Train on each file sequentially
    6. Aggregate and save results
    """
    # ===== Step 1: Parse command-line arguments =====
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--csv', required=True, help='Path to CSV file or folder')
    parser.add_argument('--time-col', default='timestamp', help='Time column used for sorting')
    parser.add_argument('--target', default=None, help='Optional single target column; if omitted, all numeric columns used')
    
    # Sequence length arguments
    parser.add_argument('--seq-len', type=int, default=0, help='Sequence length; 0 = adaptive per-file (10%% of rows clamped)')
    parser.add_argument('--min-seq-len', type=int, default=30, help='Minimum sequence length for adaptive mode')
    parser.add_argument('--max-seq-len', type=int, default=300, help='Maximum sequence length for adaptive mode')
    parser.add_argument('--pred-len', type=int, default=1, help='Prediction horizon (steps ahead)')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs per file')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='L2 regularization weight decay')
    parser.add_argument('--lr-factor', type=float, default=0.5, help='LR reduction factor for scheduler')
    parser.add_argument('--scheduler-patience', type=int, default=3, help='Scheduler patience for ReduceLROnPlateau')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability')
    
    # Model architecture
    parser.add_argument('--hidden-channels', nargs='+', type=int, default=[64, 64], help='List of channel sizes for TCN blocks')
    parser.add_argument('--kernel-size', type=int, default=3, help='Convolutional kernel size')
    
    # Training configuration
    parser.add_argument('--test-split', type=float, default=0.2, help='Fraction of data for validation')
    parser.add_argument('--patience', type=int, default=8, help='Early stopping patience per-file')
    parser.add_argument('--clip-grad', type=float, default=1.0, help='Gradient clipping threshold')
    
    # Output and device
    parser.add_argument('--save', default='best_tcn_all.pth', help='Path to save best model')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda/cpu)')
    parser.add_argument('--loss', choices=['mse', 'smoothl1'], default='smoothl1', help='Loss function')
    
    # Multi-file training
    parser.add_argument('--sample-per-file', type=int, default=2000, help='Rows per file used when fitting global scaler (sample)')
    parser.add_argument('--summary-csv', default='training_summary.csv', help='Path to save training summary')
    parser.add_argument('--metrics-csv', default='metrics_summary.csv', help='Path to save metrics summary')
    parser.add_argument('--reset-optimizer-per-file', action='store_true', help='If set, optimizer state is reset before each file training')
    
    args = parser.parse_args()

    # ===== Step 2: Process and validate arguments =====
    # Ensure certain arguments have proper types (defensive programming)
    args.lr_factor = float(args.lr_factor) if hasattr(args, 'lr_factor') else 0.5
    args.scheduler_patience = int(getattr(args, 'scheduler_patience', 3))
    args.reset_optimizer_per_file = bool(getattr(args, 'reset_optimizer_per_file', False))

    # Set loss function on args for easy access in training
    if args.loss == 'mse':
        args.loss_fn = nn.MSELoss()
    else:
        args.loss_fn = nn.SmoothL1Loss()

    # ===== Step 3: Discover CSV files =====
    if os.path.isdir(args.csv):
        # If directory provided, find all CSV files (case-insensitive)
        csv_files = sorted(glob.glob(os.path.join(args.csv, "*.[cC][sS][vV]")))
    else:
        # Single file provided
        csv_files = [args.csv]
    
    if not csv_files:
        raise RuntimeError(f"No CSV files found at {args.csv}")

    print(f"\n🔍 Found {len(csv_files)} CSV files to train on.\n")
    
    # Setup device (GPU if available, otherwise CPU)
    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # ===== Step 4: Build global feature scaler =====
    # Sample rows from each file to create a representative scaler
    # This prevents any single file from dominating the scaling statistics
    collected_samples = []
    for path in csv_files:
        try:
            # Load and clean the file
            df_tmp = load_and_clean_csv(path, time_col=args.time_col)
            
            # Filter to target column if specified
            if args.target:
                if args.target not in df_tmp.columns:
                    continue
                df_tmp = df_tmp[[args.target]]
            
            nrows = len(df_tmp)
            if nrows == 0:
                continue
            
            # Sample up to sample_per_file rows uniformly
            sample_count = min(nrows, args.sample_per_file)
            if nrows > sample_count:
                # Use linspace to get evenly distributed indices
                idxs = np.linspace(0, nrows-1, sample_count).astype(int)
                sampled = df_tmp.iloc[idxs].values.astype(float)
            else:
                # Use all rows if file is small
                sampled = df_tmp.values.astype(float)
            collected_samples.append(sampled)
        except Exception:
            # Skip files that fail to load
            continue

    # Fit global scaler on collected samples
    if collected_samples:
        all_sampled = np.vstack(collected_samples)
        global_feature_scaler = StandardScaler().fit(all_sampled)
        print(f"Fitted global scaler on {all_sampled.shape[0]} sampled rows from {len(collected_samples)} files.")
    else:
        global_feature_scaler = None
        print("⚠️ No samples for global scaler; scaler will be fitted per-file (not recommended).")

    # ===== Step 5: Infer feature dimensions from first usable file =====
    sample_df = None
    for p in csv_files:
        try:
            df_s = load_and_clean_csv(p, time_col=args.time_col)
            
            # Filter to target column if specified
            if args.target:
                if args.target not in df_s.columns:
                    continue
                df_s = df_s[[args.target]]
            
            # Use first file with at least one numeric column
            if df_s.shape[1] > 0:
                sample_df = df_s
                break
        except Exception:
            continue
    
    if sample_df is None:
        raise RuntimeError("No usable CSV to infer feature count.")

    # Number of input and output features (same for autoregressive prediction)
    input_feature_count = sample_df.shape[1]
    output_feature_count = input_feature_count

    # ===== Step 6: Initialize model and optimizer =====
    # Create TCN model (shared across all files for transfer learning)
    model = TCN(input_size=input_feature_count, output_size=output_feature_count,
                num_channels=list(args.hidden_channels), kernel_size=args.kernel_size, dropout=args.dropout)
    
    # Initialize optimizer (may be reset per-file if flag is set)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ===== Step 7: Initialize accumulators for overall summary =====
    training_summary_rows = []      # Per-file training statistics
    metrics_summary_rows = []       # Per-file evaluation metrics
    overall_train_weighted_sum = 0.0  # Weighted sum of training losses
    overall_val_weighted_sum = 0.0    # Weighted sum of validation losses
    overall_train_count = 0           # Total training windows across all files
    overall_val_count = 0             # Total validation windows across all files
    collected_train_curves = []       # Loss curves for averaging
    collected_val_curves = []

    # ===== Step 8: Train on each CSV file sequentially =====
    for index, csv_path in enumerate(csv_files, start=1):
        print(f"🔹 [{index}/{len(csv_files)}] {os.path.basename(csv_path)}")
        
        # Determine whether to reset optimizer for this file
        opt_for_file = None
        if args.reset_optimizer_per_file:
            opt_for_file = None  # train_one_file will create a fresh optimizer
        else:
            opt_for_file = optimizer  # reuse optimizer (transfer learning)

        # Train on this file
        model, global_feature_scaler, last_train_loss, best_val_loss, n_train_windows, n_val_windows, metrics_dict, curves = train_one_file(
            model=model,
            file_path=csv_path,
            global_feature_scaler=global_feature_scaler,
            args=args,
            device=device,
            optimizer=opt_for_file
        )

        # If optimizer was reset, create a new global optimizer for next file
        if args.reset_optimizer_per_file:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # Record training statistics for this file
        training_summary_rows.append({
            "file": os.path.basename(csv_path),
            "last_train_loss": last_train_loss,
            "best_val_loss": best_val_loss,
            "train_windows": int(n_train_windows),
            "val_windows": int(n_val_windows)
        })
        
        # Record metrics for this file
        if metrics_dict:
            metrics_summary_rows.append({"file": os.path.basename(csv_path), **metrics_dict})
        else:
            metrics_summary_rows.append({"file": os.path.basename(csv_path), "MAE": None, "RMSE": None, "R2": None, "MAP%": None})

        # Accumulate weighted losses for overall average
        if n_train_windows > 0 and not np.isnan(last_train_loss):
            overall_train_weighted_sum += float(last_train_loss) * int(n_train_windows)
            overall_train_count += int(n_train_windows)
        if n_val_windows > 0 and not np.isnan(best_val_loss):
            overall_val_weighted_sum += float(best_val_loss) * int(n_val_windows)
            overall_val_count += int(n_val_windows)

        # Collect loss curves for averaging
        if curves and curves[0] and curves[1]:
            collected_train_curves.append(curves[0])
            collected_val_curves.append(curves[1])

    # ===== Step 9: Calculate overall weighted averages =====
    overall_train = (overall_train_weighted_sum / overall_train_count) if overall_train_count else None
    overall_val = (overall_val_weighted_sum / overall_val_count) if overall_val_count else None

    # Print overall summary
    print("\n📊 ===== OVERALL SUMMARY =====")
    if overall_train is not None:
        print(f"Weighted Avg Train Loss : {overall_train:.6f} (over {overall_train_count} windows)")
    else:
        print("No valid train samples processed.")
    if overall_val is not None:
        print(f"Weighted Avg Val Loss   : {overall_val:.6f} (over {overall_val_count} windows)")
    else:
        print("No valid validation samples processed.")
    print("==============================\n")

    # ===== Step 10: Save summary CSVs =====
    # Save per-file training statistics
    try:
        pd.DataFrame(training_summary_rows).to_csv(args.summary_csv, index=False)
        print(f"Saved training summary to {args.summary_csv}")
    except Exception as e:
        print(f"⚠️ Could not save training summary: {e}")
    
    # Save per-file evaluation metrics
    try:
        pd.DataFrame(metrics_summary_rows).to_csv(args.metrics_csv, index=False)
        print(f"Saved metrics summary to {args.metrics_csv}")
    except Exception as e:
        print(f"⚠️ Could not save metrics summary: {e}")

    # ===== Step 11: Save final model and scaler =====
    # Save model state dictionary
    try:
        torch.save({'model_state_dict': model.state_dict()}, args.save)
        print(f"Saved final model to {args.save}")
    except Exception as e:
        print(f"⚠️ Could not save final model: {e}")

    # Save scaler and column information for inference
    scaler_out_path = os.path.splitext(args.save)[0] + "_scaler.joblib"
    try:
        joblib.dump({'feature_scaler': global_feature_scaler, 'columns': sample_df.columns.tolist()}, scaler_out_path)
        print(f"Saved scaler metadata to {scaler_out_path}")
    except Exception as e:
        print(f"⚠️ Could not save scaler metadata: {e}")

    # ===== Step 12: Create overall average loss curve =====
    if collected_train_curves:
        # Find minimum length across all curves
        min_len = min(len(c) for c in collected_train_curves)
        # Truncate all curves to minimum length and average
        mean_train = np.mean(np.array([np.array(c[:min_len]) for c in collected_train_curves]), axis=0)
        mean_val = np.mean(np.array([np.array(c[:min_len]) for c in collected_val_curves]), axis=0)
        
        # Plot and save average loss curve
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.plot(range(1, len(mean_train) + 1), mean_train, label="Avg Train Loss")
        plt.plot(range(1, len(mean_val) + 1), mean_val, label="Avg Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Average Loss Curve (All CSVs)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join("plots", "overall_average_loss.png"), dpi=150)
        plt.close()
        print("Saved overall average loss curve to plots/overall_average_loss.png")

    print("✅ Training complete.")


# Entry point: run main function when script is executed
if __name__ == "__main__":
    main()