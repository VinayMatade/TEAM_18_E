# Physics Loss Integration - Summary

## Critical Fixes Applied ✅

### 1. Dataset Returns Raw Accelerations
**File**: `train.py` - `HybridNoiseDataset.__getitem__()`

The dataset now returns a 3-tuple:
```python
return (x_full, target_norm, raw_acc)
```
- `x_full`: (seq_len, 8) - normalized inputs [GPS(2) + IMU(6)]
- `target_norm`: (2,) - normalized target position
- `raw_acc`: (seq_len, 2) - raw AccX, AccY in m/s² for physics loss

### 2. IMU Double Integration Function
**File**: `train.py` - Module level function

```python
def imu_double_integrate(raw_acc, dt=0.02):
    """
    Double integrate horizontal accelerations to get displacement.
    - Removes mean bias (assumes zero net acceleration)
    - First integration: velocity
    - Second integration: position
    Returns: (B, 2) final position in meters
    """
```

**Key Features**:
- Uses short horizon (25 steps = 0.5s) to minimize drift
- Removes mean bias before integration
- Returns displacement relative to window start

### 3. Physics Loss in Training Loop
**File**: `train.py` - `main()` training loop

**Loss Components**:
```python
loss_pos = criterion(pred, y)           # Main position loss (normalized)
loss_phy = criterion(pred_m, imu_disp)  # Physics loss (meters)
loss = loss_pos + lambda_phy * loss_phy # Combined
```

**Lambda Schedule**:
- Start: 0.01 (epoch 0)
- Ramp: Linear increase over 10 epochs
- Max: 0.1 (epoch 10+)

This gradual ramp prevents physics loss from dominating early training.

### 4. Gradient Monitoring
**File**: `train.py` - Validation loop

Every 5 epochs, monitors gradient contribution:
- GPS gradient sum
- IMU gradient sum
- Percentage breakdown

**Warnings**:
- If IMU < 10% of GPS: Model ignoring IMU
- If GPS < 33% of IMU: Model ignoring GPS

### 5. Noise Handling Robustness
**File**: `train.py` - `HybridNoiseDataset.__getitem__()`

Safe handling when `real_noise_bank` is None:
```python
if self.real_noise_bank is not None and len(self.real_noise_bank) >= n_rows:
    # Use real noise
else:
    # Fallback to synthetic
    vib = np.random.normal(0, 0.05, (n_rows, 2))
```

## Training Output Format

```
Epoch | LR        | TrainLoss  | Val MAE   | λ_phy   | Stats
------+----------+------------+-----------+---------+---------------------------
1     | 1.00e-03 | 0.123456   | 2.3456    | 0.010   | med:2.1 90p:4.5 %<1m:15.2%
       Grad: GPS=123.45 (85.2%), IMU=21.43 (14.8%)
```

## Physics Loss Benefits

1. **Forces IMU Usage**: Model must learn to use IMU signals to match physics
2. **Reduces GPS Shortcut**: Can't just memorize GPS patterns
3. **Improves Generalization**: Physics constraints are universal
4. **Better Uncertainty**: Model learns when to trust GPS vs IMU

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `IMU_HORIZON` | 25 steps | ~0.5s at 50Hz, reduces drift |
| `LAMBDA_PHY_START` | 0.01 | Initial physics weight |
| `LAMBDA_PHY_MAX` | 0.1 | Final physics weight |
| `LAMBDA_PHY_RAMP_EPOCHS` | 10 | Ramp duration |
| `dt` | 0.02s | 50Hz sampling rate |

## Expected Behavior

### Early Training (Epochs 1-5)
- GPS gradient >> IMU gradient (80-90% GPS)
- Physics loss helps but doesn't dominate
- Model learns basic GPS patterns

### Mid Training (Epochs 6-15)
- IMU gradient increases (20-40% IMU)
- Physics loss weight reaches maximum
- Model starts integrating IMU signals

### Late Training (Epochs 16+)
- Balanced GPS/IMU usage (40-60% each)
- Physics loss stabilizes
- Model uses both modalities effectively

## Troubleshooting

### If IMU gradient stays < 10%
1. Increase `LAMBDA_PHY_MAX` to 0.2-0.5
2. Check IMU normalization (should be ~[-3, 3])
3. Verify raw_acc units are m/s²
4. Try longer `IMU_HORIZON` (30-40 steps)

### If training becomes unstable
1. Reduce `LAMBDA_PHY_MAX` to 0.05
2. Increase `LAMBDA_PHY_RAMP_EPOCHS` to 20
3. Check for NaN in IMU data
4. Reduce learning rate

### If physics loss >> position loss
1. Check scaler denormalization (t_mean, t_scale)
2. Verify IMU integration units match GPS units (meters)
3. Reduce `LAMBDA_PHY_MAX`

## Next Steps (Optional Improvements)

1. **Seq-to-seq prediction**: Predict full trajectory, not just endpoint
2. **Orientation correction**: Use gyro to rotate body-frame accel to world-frame
3. **Velocity auxiliary loss**: Single integration is more stable than double
4. **Adaptive lambda**: Adjust physics weight based on GPS quality/noise level
5. **Causal convolutions**: For real-time deployment

## Memory Considerations

Current config uses:
- Batch size: 256
- Channels: [64, 128, 64]
- Sequence length: 50

If OOM occurs:
- Reduce batch size to 128
- Reduce channels to [64, 96, 64]
- Reduce `IMU_HORIZON` to 20

## Validation

The validation script (`validation.py`) now includes:
- Receptive field analysis
- Modality importance analysis (GPS vs IMU)
- Enhanced saliency maps with feature labels
- Gradient monitoring across samples

Run after training:
```bash
python validation.py
```

Check `validation_output/modality_importance.png` to verify IMU usage.
