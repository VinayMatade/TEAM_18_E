# GPS-IMU Fusion Training

Physics-informed TCN model for GPS denoising using IMU data.

## Quick Start

### Using the CLI (Recommended)

```bash
# Install the package
pip install -e .

# Complete workflow
uav-log-processor clean -i files/logs -o files/cleaned
uav-log-processor noise-bank -i files/cleaned/train -o noise_bank.npy
uav-log-processor train -i files/cleaned/train -n noise_bank.npy -e 60
uav-log-processor run -m best_model.pth -i test_flight.csv -o corrected.csv --plot
```

See [CLI_USAGE.md](CLI_USAGE.md) for detailed CLI documentation.

### Using Scripts Directly

```bash
python train.py
```

## Configuration

Edit `train.py` lines 17-26:
- `CSV_FOLDER`: Path to your CSV files
- `BATCH_SIZE`: 128 (reduce to 64 if OOM)
- `SEQ_LEN`: 125 timesteps (2.5 seconds at 50Hz)
- `EPOCHS`: 60

## Required CSV Columns

- GPS_Lat, GPS_Lng
- IMU_AccX, IMU_AccY, IMU_AccZ
- IMU_GyrX, IMU_GyrY, IMU_GyrZ

## Output Files

- `best_model.pth` - Best checkpoint (lowest validation MAE)
- `scalers.save` - Fitted scalers (4 total)
- `training_curve.png` - MAE over epochs

## Features

- **Physics Loss**: Compares predicted vs IMU-integrated velocity (speed magnitude)
- **Rotation**: Gyro-based attitude correction for horizontal acceleration
- **Noise Augmentation**: Drift + vibration added to GPS
- **Delta Features**: Explicit velocity signal (GPS frame-to-frame change)

## Model Architecture

- **Input**: 10 features (GPS position + GPS velocity + IMU)
- **Network**: 6-layer dilated TCN (128 channels, kernel=7)
- **Dilations**: [1, 2, 4, 8, 16, 32]
- **Receptive Field**: 379 timesteps (covers 125-step window 3×)
- **Output**: 2D displacement (relative meters)

## Training Details

- **Optimizer**: AdamW (lr=0.0005, weight_decay=1e-5)
- **Loss**: SmoothL1Loss (position) + Physics Loss (velocity magnitude)
- **Physics Weight**: Ramps from 0.01 → 0.1 over first 10 epochs
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: Patience=15 epochs

## Expected Performance

- **Validation MAE**: 0.8-1.5 meters
- **Training Time**: 15-30 min (GPU) / 2-4 hours (CPU)
- **GPU Memory**: ~3-4 GB

## Troubleshooting

**Out of Memory**: Reduce `BATCH_SIZE` to 64 or 32

**Poor MAE**: Check that:
- CSV files have all required columns
- GPS coordinates are valid (not 0,0)
- IMU data is in correct units (m/s² for accel, rad/s for gyro)

**Validation Analysis**:
```bash
python validation.py
```
Generates comprehensive analysis in `validation_output/` folder.

## Key Implementation Details

### Physics Loss (Speed Magnitude)
Compares speed (not velocity vector) to avoid yaw alignment issues:
```python
pred_speed = torch.norm(pred_avg_vel, dim=1)
imu_speed = torch.norm(imu_avg_vel, dim=1)
loss_phy = criterion(pred_speed, imu_speed)
```

### Gyro-Based Rotation
Integrates gyro to estimate attitude, rotates accelerometer to level frame:
- Removes gravity component
- Extracts horizontal acceleration
- Used for physics loss

### Per-File Datasets
Each CSV becomes separate dataset, concatenated to prevent windows crossing file boundaries (no train/val leakage).

---

**Status**: Production-ready. All critical bugs fixed.
