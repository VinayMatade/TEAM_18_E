# Design Document

## Overview

This design specifies a codebase cleanup and visualization system for the GPS-IMU fusion project. The system will identify and remove unused code, then create a comprehensive visualization script that displays raw GPS data, model-corrected trajectories, and ground truth paths for performance evaluation.

The project uses a Temporal Convolutional Network (TCN) to denoise GPS coordinates using IMU sensor data. The visualization script will enable researchers to visually assess model performance by comparing multiple trajectory representations on a single plot.

## Architecture

### Component Structure

```
uav_log_processor/
├── cli.py                    # CLI interface (KEEP - used)
├── __main__.py              # Module entry point (KEEP - used)
├── pipeline.py              # Full pipeline (ANALYZE - may be unused)
├── config.py                # Configuration (ANALYZE)
├── parsers/                 # Log parsers (KEEP - used by CLI)
│   ├── txt_parser.py       # Text log parser (KEEP)
│   └── ...
├── processors/              # Data processors (ANALYZE)
└── utils/                   # Utilities (ANALYZE)

scripts/
├── train.py                 # Training script (KEEP - core)
├── run.py                   # Inference script (KEEP - core)
├── clean_and_convert_logs.py  # Log cleaning (KEEP - used by CLI)
├── build_noise_bank.py      # Noise generation (KEEP - used by CLI)
├── validation.py            # Model analysis (KEEP - useful)
├── visualize_trajectory.py  # NEW - comprehensive visualization
├── example_raw_usage.py     # Example (REMOVE - documentation only)
├── compare_parsers.py       # Example (REMOVE - documentation only)
└── check_scalers.py         # Diagnostic (REMOVE - one-time use)
```

### Cleanup Strategy

1. **Identify Core Workflows**: Trace imports from CLI commands and main scripts
2. **Mark Unused Code**: Flag files/functions not in the dependency tree
3. **Safe Removal**: Remove only files that are:
   - Not imported by any core script
   - Not referenced in CLI commands
   - Not part of the package API
4. **Preserve Documentation**: Keep README, CHANGELOG, and usage guides

### Visualization Architecture

The new `visualize_trajectory.py` script will:

1. Load a trained model and input CSV file
2. Run inference to generate corrected GPS coordinates
3. Compute or load ground truth trajectory
4. Plot all three trajectories on a single figure
5. Calculate and display error metrics
6. Save the visualization to a file

## Components and Interfaces

### 1. Codebase Analyzer

**Purpose**: Identify unused files and functions

**Interface**:
```python
class CodebaseAnalyzer:
    def analyze_imports(self, root_dir: str) -> Dict[str, Set[str]]
    def find_unused_files(self, root_dir: str, entry_points: List[str]) -> List[str]
    def generate_report(self) -> str
```

**Behavior**:
- Parses Python files to extract import statements
- Builds dependency graph from entry points (CLI, train.py, run.py)
- Identifies files not in the dependency tree
- Generates human-readable report

### 2. Trajectory Visualizer

**Purpose**: Create comprehensive trajectory comparison plots

**Interface**:
```python
class TrajectoryVisualizer:
    def __init__(self, model_path: str, scaler_path: str)
    def load_data(self, csv_path: str) -> pd.DataFrame
    def run_inference(self, data: pd.DataFrame) -> np.ndarray
    def compute_ground_truth(self, data: pd.DataFrame) -> np.ndarray
    def plot_trajectories(self, raw_gps, corrected_gps, ground_truth) -> plt.Figure
    def calculate_metrics(self, raw_gps, corrected_gps, ground_truth) -> Dict[str, float]
    def save_plot(self, fig: plt.Figure, output_path: str)
```

**Behavior**:
- Loads trained TCN model and scalers
- Processes input CSV to extract GPS and IMU data
- Runs model inference with proper windowing and coordinate transforms
- Computes ground truth from smoothed GPS or RTK data
- Creates matplotlib figure with all trajectories
- Calculates MAE, RMSE, and max error metrics
- Saves high-resolution plot to file

### 3. Coordinate Transformer

**Purpose**: Convert between lat/lon degrees and local meters

**Interface**:
```python
class CoordinateTransformer:
    def __init__(self, origin_lat: float, origin_lon: float)
    def latlon_to_meters(self, lat: np.ndarray, lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    def meters_to_latlon(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
```

**Behavior**:
- Uses local tangent plane approximation
- Converts degrees to meters: `x = (lon - lon0) * 111132 * cos(lat0)`, `y = (lat - lat0) * 110649`
- Handles coordinate system alignment for plotting

## Data Models

### Input Data Format

```python
# CSV columns required for visualization
required_columns = [
    'GPS_Lat',      # Latitude in degrees
    'GPS_Lng',      # Longitude in degrees
    'IMU_AccX',     # Accelerometer X (m/s²)
    'IMU_AccY',     # Accelerometer Y (m/s²)
    'IMU_AccZ',     # Accelerometer Z (m/s²)
    'IMU_GyrX',     # Gyroscope X (rad/s)
    'IMU_GyrY',     # Gyroscope Y (rad/s)
    'IMU_GyrZ',     # Gyroscope Z (rad/s)
]

# Optional columns
optional_columns = [
    'GPA_HAcc',     # Horizontal accuracy (meters)
    'RTK_Lat',      # RTK ground truth latitude
    'RTK_Lng',      # RTK ground truth longitude
]
```

### Trajectory Data Structure

```python
@dataclass
class TrajectoryData:
    """Container for trajectory information"""
    timestamps: np.ndarray      # Time in seconds
    x: np.ndarray              # X position in meters
    y: np.ndarray              # Y position in meters
    label: str                 # Trajectory label for legend
    color: str                 # Plot color
    style: str                 # Line style ('-', '--', ':')
```

### Error Metrics

```python
@dataclass
class ErrorMetrics:
    """GPS error statistics"""
    mae: float                 # Mean Absolute Error (meters)
    rmse: float                # Root Mean Square Error (meters)
    max_error: float           # Maximum error (meters)
    median_error: float        # Median error (meters)
    std_error: float           # Standard deviation (meters)
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Acceptance Criteria Testing Prework

1.1 WHEN analyzing the codebase THEN the system SHALL identify files that are not imported or used by the main workflows
Thoughts: This is about analyzing the dependency graph of the codebase. We can test this by creating a small test codebase with known unused files, running the analyzer, and checking that it correctly identifies them.
Testable: yes - example

1.2 WHEN removing files THEN the system SHALL preserve all files required for training, inference, and data processing
Thoughts: This is about ensuring we don't break the system. We can test this by running the cleanup, then verifying that all CLI commands still work.
Testable: yes - example

1.3 WHEN removing code THEN the system SHALL ensure that the CLI commands (clean, noise-bank, train, run) continue to function correctly
Thoughts: This is the same as 1.2 - testing that CLI commands work after cleanup.
Testable: yes - example

1.4 WHEN cleanup is complete THEN the system SHALL maintain all functionality described in README.md
Thoughts: This is a high-level integration test. We can verify by running the example commands from README.
Testable: yes - example

2.1 WHEN the visualization script runs THEN the system SHALL plot the raw GPS latitude and longitude coordinates as a 2D trajectory
Thoughts: This is testing that the plot contains raw GPS data. We can generate random GPS coordinates, run the visualizer, and check that the plot contains those coordinates.
Testable: yes - property

2.2 WHEN displaying raw GPS THEN the system SHALL use a distinct color and line style to differentiate it from other trajectories
Thoughts: This is testing visual properties of the plot. We can check that different trajectories have different colors/styles.
Testable: yes - property

2.3 WHEN plotting coordinates THEN the system SHALL convert latitude/longitude to meters for accurate distance representation
Thoughts: This is testing the coordinate transformation. We can test that the conversion is correct by checking known lat/lon pairs convert to expected meter values.
Testable: yes - property

2.4 WHEN the raw GPS trajectory is displayed THEN the system SHALL include axis labels showing distance in meters
Thoughts: This is testing plot metadata. We can check that the axis labels contain "meters".
Testable: yes - example

3.1 WHEN the visualization script runs THEN the system SHALL plot the model's corrected GPS coordinates as a 2D trajectory
Thoughts: Similar to 2.1, testing that corrected coordinates appear in the plot.
Testable: yes - property

3.2 WHEN displaying model output THEN the system SHALL use a distinct color and line style to differentiate it from raw GPS
Thoughts: Same as 2.2, testing visual differentiation.
Testable: yes - property

3.3 WHEN plotting model output THEN the system SHALL align the corrected trajectory with the same coordinate system as raw GPS
Thoughts: This is testing that both trajectories use the same origin and scale. We can verify that the coordinate transformer is applied consistently.
Testable: yes - property

3.4 WHEN the model output is displayed THEN the system SHALL show the trajectory for all timesteps where predictions are available
Thoughts: This is testing that we don't drop data. We can count the number of points in the plot vs the input.
Testable: yes - property

4.1 WHEN ground truth data is available THEN the system SHALL plot the actual drone trajectory as a 2D path
Thoughts: Testing that ground truth appears when available.
Testable: yes - example

4.2 WHEN displaying ground truth THEN the system SHALL use a distinct color and line style to differentiate it from GPS and model output
Thoughts: Same visual differentiation test as before.
Testable: yes - property

4.3 WHEN ground truth is not available THEN the system SHALL compute an approximation using smoothed GPS data
Thoughts: This is testing fallback behavior. We can test with data that lacks ground truth and verify smoothing is applied.
Testable: yes - example

4.4 WHEN the actual trajectory is displayed THEN the system SHALL include it in the same plot as raw GPS and model output
Thoughts: Testing that all trajectories are on one figure.
Testable: yes - example

5.1 WHEN the visualization script runs THEN the system SHALL create a single figure containing all trajectory plots
Thoughts: Testing that we create exactly one figure, not multiple.
Testable: yes - property

5.2 WHEN displaying multiple trajectories THEN the system SHALL include a legend identifying each trajectory type
Thoughts: Testing that the legend exists and contains the right labels.
Testable: yes - property

5.3 WHEN plotting trajectories THEN the system SHALL use equal aspect ratio to prevent distortion
Thoughts: Testing matplotlib axis properties. We can check that aspect='equal' is set.
Testable: yes - example

5.4 WHEN the plot is complete THEN the system SHALL include a title, axis labels, and grid lines for readability
Thoughts: Testing plot metadata exists.
Testable: yes - example

6.1 WHEN the visualization is complete THEN the system SHALL save the plot as a PNG file
Thoughts: Testing file output. We can verify the file exists and is a valid PNG.
Testable: yes - property

6.2 WHEN saving the plot THEN the system SHALL use a high resolution suitable for publication
Thoughts: Testing DPI setting. We can check that DPI >= 300.
Testable: yes - example

6.3 WHEN the file is saved THEN the system SHALL print the output file path to the console
Thoughts: Testing console output. We can capture stdout and verify the path is printed.
Testable: yes - property

6.4 WHEN the output file already exists THEN the system SHALL overwrite it with the new visualization
Thoughts: Testing file overwrite behavior. We can create a dummy file, run the script, and verify it's replaced.
Testable: yes - example

7.1 WHEN running the script THEN the system SHALL accept a command-line argument for the model path
Thoughts: Testing CLI argument parsing. We can test with various argument combinations.
Testable: yes - example

7.2 WHEN running the script THEN the system SHALL accept a command-line argument for the input CSV file
Thoughts: Same as 7.1.
Testable: yes - example

7.3 WHEN running the script THEN the system SHALL accept a command-line argument for the output plot file
Thoughts: Same as 7.1.
Testable: yes - example

7.4 WHEN command-line arguments are not provided THEN the system SHALL use sensible default values
Thoughts: Testing default behavior. We can run without arguments and verify defaults are used.
Testable: yes - example

7.5 WHEN invalid arguments are provided THEN the system SHALL display a helpful error message
Thoughts: Testing error handling. We can provide bad arguments and check for error messages.
Testable: yes - example

8.1 WHEN comparing trajectories THEN the system SHALL compute the mean absolute error between raw GPS and ground truth
Thoughts: This is testing error calculation. We can test with known trajectories and verify the MAE is correct.
Testable: yes - property

8.2 WHEN comparing trajectories THEN the system SHALL compute the mean absolute error between model output and ground truth
Thoughts: Same as 8.1.
Testable: yes - property

8.3 WHEN error metrics are computed THEN the system SHALL display them in the console output
Thoughts: Testing console output. We can capture stdout and verify metrics are printed.
Testable: yes - property

8.4 WHEN error metrics are computed THEN the system SHALL include them as text annotations on the plot
Thoughts: Testing plot annotations. We can check that text objects exist in the figure.
Testable: yes - property

### Property Reflection

After reviewing all properties, I identify the following redundancies:

- Properties 2.2, 3.2, and 4.2 all test visual differentiation - these can be combined into one property that checks all trajectories have distinct visual properties
- Properties 2.1, 3.1, and 4.1 all test that trajectories appear in the plot - these can be combined into one property
- Properties 8.1 and 8.2 both test MAE calculation - these can be combined into one property that tests error calculation for any pair of trajectories

After consolidation, we have fewer, more comprehensive properties.

### Correctness Properties

Property 1: Coordinate transformation preserves distances
*For any* pair of GPS coordinates, converting to meters and back should preserve the original coordinates within numerical precision
**Validates: Requirements 2.3**

Property 2: All trajectories use consistent coordinate system
*For any* set of trajectories (raw, corrected, ground truth), all should be transformed using the same origin point and scale factors
**Validates: Requirements 3.3**

Property 3: Trajectories have distinct visual properties
*For any* pair of trajectories in the plot, they should have different colors or line styles to enable visual differentiation
**Validates: Requirements 2.2, 3.2, 4.2**

Property 4: All available trajectory data is plotted
*For any* trajectory with N data points, the plot should contain N points (no data is dropped)
**Validates: Requirements 2.1, 3.1, 3.4**

Property 5: Error metrics are correctly calculated
*For any* pair of trajectories, the computed MAE should equal the mean of the Euclidean distances between corresponding points
**Validates: Requirements 8.1, 8.2**

Property 6: Plot contains all required metadata
*For any* generated plot, it should contain a title, axis labels with units, a legend, and grid lines
**Validates: Requirements 5.2, 5.4**

Property 7: File output succeeds and overwrites existing files
*For any* output path, saving the plot should create a valid PNG file, and running again should overwrite the previous file
**Validates: Requirements 6.1, 6.4**

Property 8: Console output includes all metrics
*For any* visualization run, the console output should contain all computed error metrics (MAE, RMSE, max error)
**Validates: Requirements 8.3**

Property 9: Plot annotations match computed metrics
*For any* visualization, the text annotations on the plot should match the error metrics printed to console
**Validates: Requirements 8.4**

## Error Handling

### File Not Found Errors
- **Model file missing**: Display clear error message with expected path
- **Input CSV missing**: Display error and suggest checking file path
- **Scaler file missing**: Attempt to load from default location, warn if not found

### Data Validation Errors
- **Missing required columns**: List missing columns and exit gracefully
- **Empty or invalid data**: Check for NaN values, zero coordinates, display warning
- **Insufficient data for windowing**: Require minimum sequence length, display error

### Model Inference Errors
- **Model architecture mismatch**: Catch load errors, suggest checking model version
- **Device errors (CUDA)**: Fall back to CPU if GPU unavailable
- **Numerical errors**: Catch NaN/Inf in predictions, log warning

### Plotting Errors
- **Empty trajectories**: Check for valid data before plotting, skip empty trajectories
- **Coordinate transformation errors**: Validate lat/lon ranges, handle edge cases
- **File write errors**: Check permissions, disk space before saving

## Testing Strategy

### Unit Testing

We will write unit tests for:

1. **CoordinateTransformer**
   - Test lat/lon to meters conversion with known coordinates
   - Test round-trip conversion (meters -> lat/lon -> meters)
   - Test edge cases (equator, poles, date line)

2. **Error Metric Calculation**
   - Test MAE calculation with known trajectories
   - Test RMSE calculation
   - Test handling of different length trajectories

3. **Data Loading**
   - Test CSV parsing with valid data
   - Test handling of missing columns
   - Test handling of NaN values

4. **CLI Argument Parsing**
   - Test with all arguments provided
   - Test with default values
   - Test with invalid arguments

### Property-Based Testing

We will use **Hypothesis** (Python's property-based testing library) for property tests.

Configuration: Each property test will run a minimum of 100 iterations.

Property tests will be tagged with comments in this format:
`# Feature: codebase-cleanup-and-visualization, Property N: <property text>`

Each correctness property will be implemented as a single property-based test:

1. **Property 1: Coordinate transformation preserves distances**
   - Generate random lat/lon coordinates
   - Convert to meters and back
   - Assert original coordinates are recovered within tolerance

2. **Property 2: All trajectories use consistent coordinate system**
   - Generate multiple random trajectories
   - Transform each to meters
   - Assert all use the same origin

3. **Property 3: Trajectories have distinct visual properties**
   - Generate random number of trajectories
   - Create plot
   - Assert each has unique color or line style

4. **Property 4: All available trajectory data is plotted**
   - Generate random trajectory with N points
   - Create plot
   - Assert plot contains N points

5. **Property 5: Error metrics are correctly calculated**
   - Generate two random trajectories
   - Compute MAE manually
   - Assert matches function output

6. **Property 6: Plot contains all required metadata**
   - Generate random trajectories
   - Create plot
   - Assert title, labels, legend, grid exist

7. **Property 7: File output succeeds and overwrites**
   - Generate random output path
   - Save plot twice
   - Assert file exists and was modified

8. **Property 8: Console output includes all metrics**
   - Capture stdout during visualization
   - Assert MAE, RMSE, max error are printed

9. **Property 9: Plot annotations match computed metrics**
   - Generate trajectories and compute metrics
   - Create plot
   - Assert annotation text matches metrics

### Integration Testing

Integration tests will verify:

1. **End-to-end visualization workflow**
   - Load real model and data
   - Run complete visualization
   - Verify output file exists and is valid

2. **CLI command preservation after cleanup**
   - Run `uav-log-processor clean` command
   - Run `uav-log-processor train` command
   - Run `uav-log-processor run` command
   - Assert all complete successfully

3. **Visualization with different data sources**
   - Test with RTK ground truth
   - Test without ground truth (smoothed fallback)
   - Test with minimal data

## Implementation Notes

### Coordinate System

The visualization uses a local tangent plane approximation:
- Origin: First GPS coordinate in the trajectory
- X-axis: East direction (longitude)
- Y-axis: North direction (latitude)
- Conversion factors depend on latitude (Earth is not a perfect sphere)

### Model Inference

The TCN model requires:
- Input shape: (batch_size, sequence_length, num_features)
- Features: [GPS_Lat, GPS_Lng, HAcc, IMU_AccX, IMU_AccY, IMU_AccZ, IMU_GyrX, IMU_GyrY, IMU_GyrZ]
- Output: Displacement in meters relative to window start
- Windowing: Sliding window with SEQ_LEN=50 timesteps

### Ground Truth Computation

When RTK data is unavailable:
1. Apply Gaussian smoothing to raw GPS (sigma=10)
2. Remove high-frequency noise
3. Use smoothed trajectory as approximate ground truth

### Performance Considerations

- Use vectorized NumPy operations for coordinate transforms
- Batch model inference when possible
- Limit plot resolution for large datasets (downsample if > 10,000 points)
