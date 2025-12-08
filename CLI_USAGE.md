# UAV Log Processor CLI

A unified command-line interface for processing UAV logs, training GPS correction models, and running inference.

## Installation

```bash
pip install -e .
```

## Commands

### 1. Clean Logs

Clean and convert raw log files to CSV format.

```bash
uav-log-processor clean -i files/logs -o files/cleaned
```

**Options:**
- `-i, --input`: Input folder with raw .log files (required)
- `-o, --output`: Output folder for cleaned CSVs (required)

**Example:**
```bash
uav-log-processor clean -i files/logs -o files/cleaned
```

### 2. Build Noise Bank

Build a noise bank from cleaned logs for training augmentation.

```bash
uav-log-processor noise-bank -i files/cleaned -o noise_bank.npy
```

**Options:**
- `-i, --input`: Input folder with cleaned CSVs (required)
- `-o, --output`: Output noise bank file (default: noise_bank.npy)

**Example:**
```bash
uav-log-processor noise-bank -i files/cleaned/train -o noise_bank.npy
```

### 3. Train Model

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

**Example:**
```bash
uav-log-processor train -i files/cleaned/train -n noise_bank.npy -e 60 -b 128
```

### 4. Run Inference

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

**Example:**
```bash
uav-log-processor run -m best_model.pth -i test_flight.csv -o corrected.csv --plot
```

## Complete Workflow Example

Here's a complete workflow from raw logs to corrected GPS:

```bash
# Step 1: Clean raw logs
uav-log-processor clean -i files/logs -o files/cleaned

# Step 2: Split data into train/test (manual or script)
# Organize files into files/cleaned/train and files/cleaned/test

# Step 3: Build noise bank from training data
uav-log-processor noise-bank -i files/cleaned/train -o noise_bank.npy

# Step 4: Train the model
uav-log-processor train -i files/cleaned/train -n noise_bank.npy -e 60

# Step 5: Run inference on test data
uav-log-processor run -m best_model.pth -i files/cleaned/test/flight.csv -o corrected.csv --plot
```

## Direct Script Usage

You can also run the underlying scripts directly:

```bash
# Clean logs
python clean_and_convert_logs.py

# Build noise bank
python build_noise_bank.py

# Train model
python train.py

# Run inference
python run.py
```

Note: When running scripts directly, you need to edit the configuration variables at the top of each script.

## Environment Variables

The CLI passes configuration through environment variables. You can also set these manually:

```bash
# For cleaning
export INPUT_FOLDER="files/logs"
export OUTPUT_FOLDER="files/cleaned"
python clean_and_convert_logs.py

# For training
export CSV_FOLDER="files/cleaned/train"
export NOISE_BANK_PATH="noise_bank.npy"
export EPOCHS=60
export BATCH_SIZE=128
python train.py

# For inference
export MODEL_PATH="best_model.pth"
export INPUT_CSV="test_flight.csv"
export OUTPUT_CSV="corrected.csv"
python run.py
```
