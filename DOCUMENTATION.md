# UAV Log Processor - Complete Documentation

## Table of Contents
1. [Quick Start](#quick-start)
2. [CLI Usage](#cli-usage)
3. [Changelog](#changelog)
4. [Implementation Details](#implementation-details)

---

## Quick Start

### Installation
```bash
pip install -e .
```

### Complete Workflow

#### 1. Clean Raw Logs
Convert raw .log files to cleaned CSV format:
```bash
uav-log-processor clean -i files/logs -o files/cleaned
```

#### 2. Organize Data
Split your cleaned files into train/test folders:
- `files/cleaned/train/` - Training data
- `files/cleaned/test/` - Test data

#### 3. Build Noise Bank
Extract real GPS noise patterns from training data:
```bash
uav-log-processor noise-bank -i files/cleaned/train -o noise_bank.npy
```

#### 4. Train Model
Train the GPS correction model:
```bash
uav-log-processor train -i files/cleaned/train -n noise_bank.npy -e 60
```

This creates:
- `best_model.pth` - Best model checkpoint
- `scalers.save` - Data scalers
- `training_curve.png` - Training visualization

#### 5. Run Inference
Correct GPS drift on test flights:
```bash
uav-log-processor run -m best_model.pth -i files/cleaned/test/flight.csv -o corrected.csv --plot
```

This creates:
- `corrected.csv` - Corrected flight data
- `result_plot.png` - Before/after comparison

---

## CLI Usage

### Commands

#### 1. Clean Logs
Clean and convert raw log files to CSV format.

```bash
uav-log-processor clean -i files/logs -o files/cleaned
```

**Options:**
- `-i, --input`: Input folder with raw .log files (required)
- `-o, --output`: Output folder for cleaned CSVs (required)

#### 2. Build Noise Bank
Build a noise bank from cleaned logs for training augmentation.

```bash
uav-log-processor noise-bank -i files/cleaned -o noise_bank.npy
```

**Options:**
- `-i, --input`: Input folder with cleaned CSVs (required)
- `-o, --output`: Output noise bank file (default: noise_bank.npy)

#### 3. Train Model
Train the GPS correction model using cleaned logs.

```bash
uav-log-processor train -i files/cleaned/train -n noise_bank.npy -e 60
```

**Options:**
- `-i, --input`: Input folder with cleaned training CSVs (required)
- `-n, --noise-bank`: Noise bank file (default: noise_bank.npy)
- `-e, --epochs`: Number of training epochs (default: 60)
- `-b, --batch-size`: Batch size (default: 128)
- `-s, --seq-len`: Sequence length (default: 125)
- `-lr, --learning-rate`: Learning rate (default: 0.0005)
- `--patience`: Early stopping patience (default: 15)
- `--device`: Device to use: cuda, cpu, or auto (default: auto)

#### 4. Run Inference
Run inference on a flight log to correct GPS drift.

```bash
uav-log-processor run -m best_model.pth -i input.csv -o corrected.csv
```

**Options:**
- `-m, --model`: Path to trained model (.pth file) (required)
- `-i, --input`: Input CSV file to correct (required)
- `-o, --output`: Output corrected CSV (default: corrected_flight.csv)
- `-s, --seq-len`: Sequence length (default: 50)
- `--plot`: Generate comparison plot
- `--device`: Device to use: cuda, cpu, or auto (default: auto)

### Direct Script Usage
You can also run the underlying scripts directly:

```bash
python clean_and_convert_logs.py
python build_noise_bank.py
python train.py
python run.py
```

Note: When running scripts directly, edit the configuration variables at the top of each script.

---

## Changelog

### [1.0.0] - New CLI Implementation

#### Added
- **New unified CLI** with 4 main commands:
  - `clean` - Clean and convert raw log files to CSV
  - `noise-bank` - Build noise bank from cleaned logs
  - `train` - Train GPS correction model
  - `run` - Run inference to correct GPS drift

#### Changed
- **Updated all scripts** to support environment variable configuration
- **CLI integration** - All scripts can now be called via CLI or directly
- **Device selection** - Added auto/cuda/cpu device selection for train and run commands

#### Technical Details
- CLI uses subprocess to call existing scripts with environment variables
- Maintains backward compatibility - scripts can still be run directly
- Entry point configured in `setup.py` as `uav-log-processor`

---

## Implementation Details

### CoordinateTransformer

**Location:** `uav_log_processor/coordinate_transformer.py`

Converts between geographic coordinates (latitude/longitude) and local Cartesian coordinates (meters).

**Features:**
- Local tangent plane approximation
- Handles edge cases (equator, high latitudes, date line crossing)
- Supports scalar and array inputs
- Accurate for distances up to ~100km from origin

**Usage:**
```python
from uav_log_processor import CoordinateTransformer

transformer = CoordinateTransformer(37.7749, -122.4194)
x, y = transformer.latlon_to_meters(37.8, -122.3)
lat, lon = transformer.meters_to_latlon(x, y)
```

**Conversion Formulas:**
- Latitude to meters: `y = (lat - lat₀) × 110,649 m/degree`
- Longitude to meters: `x = (lon - lon₀) × 111,132 × cos(lat₀) m/degree`

### Trajectory Loader

**Location:** `uav_log_processor/trajectory_loader.py`

Loads and preprocesses trajectory data from CSV files.

**Functions:**
- `load_trajectory_csv(csv_path)` - Load and preprocess trajectory data
- `validate_trajectory_data(df)` - Validate data meets requirements
- `has_rtk_data(df)` - Check if RTK ground truth is available
- `extract_gps_trajectory(df)` - Extract GPS lat/lon arrays
- `extract_rtk_trajectory(df)` - Extract RTK ground truth
- `extract_imu_data(df)` - Extract IMU accelerometer and gyroscope data

**Required Columns:**
- GPS: Latitude, Longitude
- IMU: AccX, AccY, AccZ, GyrX, GyrY, GyrZ

**Optional Columns:**
- RTK: RTK_Latitude, RTK_Longitude
- Accuracy: HAcc

**Usage:**
```python
from uav_log_processor.trajectory_loader import (
    load_trajectory_csv,
    extract_gps_trajectory,
    extract_imu_data
)

df = load_trajectory_csv('trajectory.csv')
gps_lat, gps_lon = extract_gps_trajectory(df)
acc, gyr = extract_imu_data(df)
```

### Project Structure

```
TEAM_18_E/
├── uav_log_processor/          # CLI package
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── coordinate_transformer.py
│   ├── trajectory_loader.py
│   └── data_structures.py
│
├── train.py                    # Training script
├── run.py                      # Inference script
├── validation.py               # Model analysis
├── clean_and_convert_logs.py  # Log cleaning
├── build_noise_bank.py         # Noise generation
├── setup.py                    # Package setup
│
├── requirements.txt            # Dependencies
├── README.md                   # Main documentation
├── DOCUMENTATION.md            # This file
│
├── noise_bank.npy              # Noise data
├── files/                      # Log files
│   ├── logs/                   # Raw logs
│   └── cleaned/                # Cleaned CSVs
│
└── models/                     # Trained models
    └── best_model.pth
```

### Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for full list

