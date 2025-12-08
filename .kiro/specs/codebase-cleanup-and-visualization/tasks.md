# Implementation Plan

- [x] 1. Analyze codebase and identify unused files
  - Scan all Python files in the project
  - Build import dependency graph from entry points (cli.py, train.py, run.py, validation.py)
  - Identify files not in dependency tree
  - Generate report of files to remove
  - _Requirements: 1.1_

- [ ] 2. Remove unused files and verify functionality
  - Delete identified unused files (example_raw_usage.py, compare_parsers.py, check_scalers.py)
  - Run CLI commands to verify they still work
  - Run train.py and run.py to verify core functionality
  - Update file listings in documentation if needed
  - _Requirements: 1.2, 1.3, 1.4_

- [x] 3. Create coordinate transformation module
  - Implement CoordinateTransformer class
  - Add latlon_to_meters method using local tangent plane approximation
  - Add meters_to_latlon method for reverse transformation
  - Handle edge cases (equator, high latitudes)
  - _Requirements: 2.3_

- [ ]* 3.1 Write property test for coordinate transformation
  - **Property 1: Coordinate transformation preserves distances**
  - **Validates: Requirements 2.3**

- [x] 4. Create trajectory data structures
  - Implement TrajectoryData dataclass
  - Implement ErrorMetrics dataclass
  - Add validation methods for trajectory data
  - _Requirements: 2.1, 3.1, 4.1_

- [x] 5. Implement trajectory loading and preprocessing
  - Create function to load CSV data
  - Validate required columns exist
  - Handle missing optional columns (HAcc, RTK data)
  - Filter invalid GPS coordinates (NaN, zeros)
  - _Requirements: 2.1_

- [ ] 6. Implement model inference pipeline
  - Load trained TCN model and scalers
  - Create sliding window generator for input sequences
  - Run inference with proper coordinate transformations
  - Convert model output (relative meters) to absolute coordinates
  - Handle sequence padding at start of trajectory
  - _Requirements: 3.1, 3.3, 3.4_

- [ ]* 6.1 Write property test for consistent coordinate systems
  - **Property 2: All trajectories use consistent coordinate system**
  - **Validates: Requirements 3.3**

- [ ] 7. Implement ground truth computation
  - Check for RTK columns in data
  - If RTK available, use as ground truth
  - If RTK unavailable, apply Gaussian smoothing to GPS (sigma=10)
  - Return ground truth trajectory in same format as other trajectories
  - _Requirements: 4.1, 4.3_

- [ ] 8. Implement error metric calculation
  - Create function to compute MAE between two trajectories
  - Create function to compute RMSE
  - Create function to compute max error
  - Handle trajectories of different lengths (align by timestamp)
  - Return ErrorMetrics dataclass
  - _Requirements: 8.1, 8.2_

- [ ]* 8.1 Write property test for error metric calculation
  - **Property 5: Error metrics are correctly calculated**
  - **Validates: Requirements 8.1, 8.2**

- [ ] 9. Implement trajectory plotting
  - Create matplotlib figure with appropriate size
  - Plot raw GPS trajectory with distinct color/style
  - Plot corrected GPS trajectory with distinct color/style
  - Plot ground truth trajectory with distinct color/style
  - Set equal aspect ratio to prevent distortion
  - Add title, axis labels (in meters), legend, and grid
  - _Requirements: 2.1, 2.2, 2.4, 3.1, 3.2, 4.1, 4.2, 4.4, 5.1, 5.2, 5.3, 5.4_

- [ ]* 9.1 Write property test for distinct visual properties
  - **Property 3: Trajectories have distinct visual properties**
  - **Validates: Requirements 2.2, 3.2, 4.2**

- [ ]* 9.2 Write property test for complete data plotting
  - **Property 4: All available trajectory data is plotted**
  - **Validates: Requirements 2.1, 3.1, 3.4**

- [ ]* 9.3 Write property test for plot metadata
  - **Property 6: Plot contains all required metadata**
  - **Validates: Requirements 5.2, 5.4**

- [ ] 10. Add error metric annotations to plot
  - Compute error metrics for raw GPS vs ground truth
  - Compute error metrics for corrected GPS vs ground truth
  - Add text box to plot with formatted metrics
  - Position text box to not obscure trajectories
  - _Requirements: 8.4_

- [ ]* 10.1 Write property test for console output
  - **Property 8: Console output includes all metrics**
  - **Validates: Requirements 8.3**

- [ ]* 10.2 Write property test for plot annotations
  - **Property 9: Plot annotations match computed metrics**
  - **Validates: Requirements 8.4**

- [ ] 11. Implement file output
  - Save plot as PNG with high DPI (300+)
  - Print output file path to console
  - Handle file overwrite if output already exists
  - Handle file write errors (permissions, disk space)
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [ ]* 11.1 Write property test for file output
  - **Property 7: File output succeeds and overwrites existing files**
  - **Validates: Requirements 6.1, 6.4**

- [ ] 12. Implement command-line interface
  - Add argparse for model path, input CSV, output plot path
  - Set sensible defaults for all arguments
  - Add help text for each argument
  - Validate arguments before processing
  - Display helpful error messages for invalid arguments
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ] 13. Create main visualization script
  - Integrate all components into visualize_trajectory.py
  - Add main() function that orchestrates the workflow
  - Add error handling for each step
  - Print progress messages to console
  - Print final error metrics to console
  - _Requirements: All_

- [ ] 14. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ]* 15. Write integration tests
  - Test end-to-end visualization with real data
  - Test with RTK ground truth
  - Test without ground truth (smoothed fallback)
  - Test CLI commands still work after cleanup
  - _Requirements: All_

- [ ] 16. Update documentation
  - Add visualize_trajectory.py usage to README
  - Document command-line arguments
  - Add example usage with screenshots
  - Update file structure documentation
  - _Requirements: 1.4_
