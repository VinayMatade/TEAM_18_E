# UAV Log Processor - File Documentation

This document explains all files and directories in the UAV Log Processor project, categorizing them as required, optional, or development-only.

## Project Structure Overview

```
uav_log_processor/
├── uav_log_processor/          # Main package (REQUIRED)
├── tests/                      # Test suite (DEVELOPMENT)
├── files/                      # Sample log files (OPTIONAL)
├── csv_output/                 # Generated output directory (GENERATED)
├── .kiro/                      # Kiro IDE specifications (DEVELOPMENT)
├── setup.py                    # Package installation (REQUIRED)
├── requirements.txt            # Dependencies (REQUIRED)
├── README.md                   # Project documentation (REQUIRED)
├── CHANGELOG.md               # Version history (OPTIONAL)
└── files.md                   # This documentation (OPTIONAL)
```

## Core Package Files (REQUIRED)

### Main Package: `uav_log_processor/`
The core Python package containing all processing logic.

#### Root Module Files
- **`__init__.py`** - Package initialization and version info
- **`__main__.py`** - Entry point for `python -m uav_log_processor`
- **`cli.py`** - Command-line interface implementation
- **`config.py`** - Configuration management system
- **`pipeline.py`** - Main processing pipeline orchestration

#### Parsers: `uav_log_processor/parsers/`
Handles different UAV log file formats.

- **`__init__.py`** - Parser package initialization
- **`base.py`** - Base parser class and interfaces
- **`tlog_parser.py`** - TLOG (telemetry log) format parser using pymavlink
- **`bin_parser.py`** - BIN (ArduPilot binary) format parser
- **`rlog_parser.py`** - RLOG (custom format) parser
- **`txt_parser.py`** - TXT (text-based log) format parser

#### Processors: `uav_log_processor/processors/`
Core data processing components.

- **`__init__.py`** - Processor package initialization
- **`base.py`** - Base processor class and interfaces
- **`synchronizer.py`** - Time synchronization and data alignment
- **`dataset_formatter.py`** - ML dataset formatting and normalization
- **`error_calculator.py`** - GPS error vector computation
- **`ground_truth_generator.py`** - Ground truth position estimation
- **`metadata_generator.py`** - Metadata and documentation generation
- **`motion_classifier.py`** - Motion segment classification
- **`reproducibility_manager.py`** - Reproducibility and audit trail management

#### Utilities: `uav_log_processor/utils/`
Supporting utility functions.

- **`__init__.py`** - Utils package initialization
- **`coordinates.py`** - Coordinate system transformations (WGS84 to ENU)
- **`data_quality.py`** - Data quality assessment and validation
- **`error_handling.py`** - Error handling and logging utilities
- **`gps_filter.py`** - GPS data filtering and reliability assessment
- **`io_utils.py`** - File I/O operations and utilities
- **`validation.py`** - Data validation functions
- **`visualization.py`** - Plotting and visualization functions

## Test Suite (DEVELOPMENT)

### Organized Test Structure: `tests/`
Comprehensive test suite reorganized by component type for better maintainability.

**Previous Structure**: All test files were in the root `tests/` directory
**New Structure**: Organized into logical subdirectories

#### Unit Tests: `tests/unit/`
Tests for individual components in isolation.

##### Parser Tests: `tests/unit/parsers/`
- **`test_parsers.py`** - Tests for all log format parsers (TLOG, BIN, RLOG, TXT)

##### Processor Tests: `tests/unit/processors/`
- **`test_synchronizer.py`** - Time synchronization and data alignment tests
- **`test_dataset_formatter.py`** - ML dataset formatting and normalization tests
- **`test_error_calculator.py`** - GPS error vector computation tests
- **`test_ground_truth_generator.py`** - Ground truth position estimation tests
- **`test_metadata_generator.py`** - Metadata and documentation generation tests
- **`test_motion_classifier.py`** - Motion segment classification tests
- **`test_reproducibility_manager.py`** - Reproducibility and audit trail tests

##### Utility Tests: `tests/unit/utils/`
- **`test_data_quality.py`** - Data quality validation and assessment tests
- **`test_error_handling.py`** - Error handling and logging utility tests
- **`test_gps_filter.py`** - GPS filtering and reliability assessment tests
- **`test_visualization.py`** - Plotting and visualization function tests

#### Integration Tests: `tests/integration/`
- **`test_integration.py`** - End-to-end pipeline tests with real flight data

**Benefits of New Structure**:
- Easier navigation and maintenance
- Clear separation of concerns
- Matches the main package structure
- Facilitates targeted test execution (e.g., `pytest tests/unit/parsers/`)

## Sample Data (OPTIONAL)

### Log Files: `files/`
Sample UAV log files for testing and development.

#### Log Formats: `files/logs/`
- **`.log files`** - Raw log files (auto-converted to .txt)
- **`.txt files`** - Text-based log files

#### Telemetry Logs: `files/rlog/`
- **`.rlog files`** - Custom format telemetry logs

#### Binary Logs: `files/tlog/`
- **`.tlog files`** - MAVLink telemetry logs

## Generated Output (GENERATED)

### CSV Output: `csv_output/`
Generated during processing - created automatically.

- **`train.csv`** - Training dataset (70% of data)
- **`valid.csv`** - Validation dataset (15% of data)
- **`test.csv`** - Test dataset (15% of data)
- **`aligned_full.csv`** - Complete synchronized raw data
- **`metadata.json`** - Dataset metadata and feature descriptions
- **`visualization.png`** - Trajectory comparison plots
- **`processing_log.txt`** - Processing audit trail

## Development Files (DEVELOPMENT)

### Kiro IDE Specifications: `.kiro/specs/uav-log-processor/`
Development specifications and planning documents.

- **`requirements.md`** - Formal requirements using EARS syntax
- **`design.md`** - System architecture and design decisions
- **`tasks.md`** - Implementation task breakdown

### Development Scripts
- **`check_imports.py`** - Import validation utility
- **`fix_code_issues.py`** - Code quality improvement script

### Package Metadata
- **`uav_log_processor.egg-info/`** - Generated package metadata (auto-created)

## Cleanup Recommendations (REMOVE/ARCHIVE)

### Duplicate Directories
- **`TEAM_18_E/`** - Duplicate project directory that should be removed
- **`.pytest_cache/`** - Generated pytest cache (can be safely deleted)

### Temporary Files
Any `.pyc` files, `__pycache__` directories, or other temporary files generated during development should be cleaned up regularly.

## Installation and Configuration (REQUIRED)

### Core Installation Files
- **`setup.py`** - Package installation configuration
- **`requirements.txt`** - Python dependencies list

### Documentation
- **`README.md`** - Main project documentation and usage guide
- **`CHANGELOG.md`** - Version history and changes (optional)
- **`files.md`** - This file documentation (optional)

## File Categories Summary

### REQUIRED for Production Use
- All files in `uav_log_processor/` package
- `setup.py`
- `requirements.txt`
- `README.md`

### OPTIONAL for Enhanced Functionality
- Sample log files in `files/`
- `CHANGELOG.md`
- `files.md`

### DEVELOPMENT Only
- All files in `tests/` (now organized by component type)
- All files in `.kiro/`
- Development scripts (`check_imports.py`, `fix_code_issues.py`)

### GENERATED During Processing
- All files in `csv_output/`
- `uav_log_processor.egg-info/`

### SHOULD BE REMOVED
- `TEAM_18_E/` directory (duplicate project)
- `.pytest_cache/` directory (generated cache)
- Any `__pycache__/` directories (generated bytecode)

## Usage Notes

1. **Minimum Installation**: Only the `uav_log_processor/` package, `setup.py`, `requirements.txt`, and `README.md` are needed for basic functionality.

2. **Development Setup**: Include the `tests/` directory and development scripts for contributing to the project.

3. **Sample Data**: The `files/` directory contains example log files for testing but is not required for production use.

4. **Output Location**: The system automatically creates `csv_output/` directory for generated files.

5. **IDE Integration**: The `.kiro/` directory contains development specifications for the Kiro IDE but is not needed for standalone use.

## File Dependencies

### Critical Dependencies
- Parser modules depend on `pymavlink` for .tlog and .bin parsing
- Coordinate utilities require `pyproj` for WGS84/ENU transformations
- All processors depend on `pandas` and `numpy` for data manipulation

### Optional Dependencies
- Visualization requires `matplotlib` and `seaborn`
- Advanced processing may use `scipy` for signal processing

This documentation ensures clear understanding of which files are essential for different use cases and deployment scenarios.