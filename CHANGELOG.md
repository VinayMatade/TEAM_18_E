# Changelog

## [1.0.0] - New CLI Implementation

### Added
- **New unified CLI** (`uav_log_processor/cli.py`) with 4 main commands:
  - `clean` - Clean and convert raw log files to CSV
  - `noise-bank` - Build noise bank from cleaned logs
  - `train` - Train GPS correction model
  - `run` - Run inference to correct GPS drift

### Changed
- **Updated all scripts** to support environment variable configuration:
  - `clean_and_convert_logs.py` - Reads INPUT_FOLDER, OUTPUT_FOLDER from env
  - `build_noise_bank.py` - Reads INPUT_FOLDER, OUTPUT_FILE from env
  - `train.py` - Reads CSV_FOLDER, NOISE_BANK_PATH, EPOCHS, BATCH_SIZE, etc. from env
  - `run.py` - Reads MODEL_PATH, INPUT_CSV, OUTPUT_CSV, SEQ_LEN, SKIP_PLOT from env

- **CLI integration** - All scripts can now be called via CLI or directly
- **Device selection** - Added auto/cuda/cpu device selection for train and run commands

### Documentation
- Added `CLI_USAGE.md` - Comprehensive CLI documentation
- Added `QUICK_START.md` - Quick start guide for new users
- Updated `README.md` - Added CLI quick start section

### Removed
- Old CLI implementation that was focused on log processing pipeline

### Technical Details
- CLI uses subprocess to call existing scripts with environment variables
- Maintains backward compatibility - scripts can still be run directly
- Entry point configured in `setup.py` as `uav-log-processor`

### Testing
- All CLI commands tested and working
- Version command works: `uav-log-processor 1.0.0`
- Help system functional for all commands
