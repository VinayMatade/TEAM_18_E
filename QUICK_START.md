# Quick Start Guide

## Installation

```bash
pip install -e .
```

## Complete Workflow

### 1. Clean Raw Logs
Convert raw .log files to cleaned CSV format:

```bash
python -m uav_log_processor.cli clean -i files/logs -o files/cleaned
```

### 2. Organize Data
Split your cleaned files into train/test folders:
- `files/cleaned/train/` - Training data
- `files/cleaned/test/` - Test data

### 3. Build Noise Bank
Extract real GPS noise patterns from training data:

```bash
python -m uav_log_processor.cli noise-bank -i files/cleaned/train -o noise_bank.npy
```

### 4. Train Model
Train the GPS correction model:

```bash
python -m uav_log_processor.cli train -i files/cleaned/train -n noise_bank.npy -e 60
```

This will create:
- `best_model.pth` - Best model checkpoint
- `scalers.save` - Data scalers
- `training_curve.png` - Training visualization

### 5. Run Inference
Correct GPS drift on test flights:

```bash
python -m uav_log_processor.cli run -m best_model.pth -i files/cleaned/test/flight.csv -o corrected.csv --plot
```

This will create:
- `corrected.csv` - Corrected flight data
- `result_plot.png` - Before/after comparison (if --plot is used)

## Alternative: Direct Script Usage

You can also run the scripts directly:

```bash
# Edit configuration at top of each script, then run:
python clean_and_convert_logs.py
python build_noise_bank.py
python train.py
python run.py
```

## Help

Get help for any command:

```bash
python -m uav_log_processor.cli --help
python -m uav_log_processor.cli clean --help
python -m uav_log_processor.cli train --help
```

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- See `requirements.txt` for full list
