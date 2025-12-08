# Requirements Document

## Introduction

This document specifies requirements for cleaning up the GPS-IMU fusion codebase and creating a comprehensive visualization script that displays model performance. The system processes drone flight logs to denoise GPS data using IMU measurements and a trained TCN model.

## Glossary

- **TCN**: Temporal Convolutional Network, the neural network architecture used for GPS denoising
- **IMU**: Inertial Measurement Unit, provides accelerometer and gyroscope data
- **GPS**: Global Positioning System, provides position coordinates
- **Trajectory**: The path followed by the drone during flight
- **Model Output**: The corrected GPS coordinates produced by the trained TCN model
- **Raw GPS**: The original, noisy GPS coordinates from the drone's sensors
- **Ground Truth**: The actual path of the drone (approximated from RTK GPS or smoothed data)
- **Visualization Script**: A Python script that displays multiple trajectory views for comparison

## Requirements

### Requirement 1

**User Story:** As a developer, I want to remove unused code and files from the codebase, so that the project is easier to maintain and understand.

#### Acceptance Criteria

1. WHEN analyzing the codebase THEN the system SHALL identify files that are not imported or used by the main workflows
2. WHEN removing files THEN the system SHALL preserve all files required for training, inference, and data processing
3. WHEN removing code THEN the system SHALL ensure that the CLI commands (clean, noise-bank, train, run) continue to function correctly
4. WHEN cleanup is complete THEN the system SHALL maintain all functionality described in README.md

### Requirement 2

**User Story:** As a researcher, I want to visualize the raw GPS navigation data, so that I can see the original noisy measurements.

#### Acceptance Criteria

1. WHEN the visualization script runs THEN the system SHALL plot the raw GPS latitude and longitude coordinates as a 2D trajectory
2. WHEN displaying raw GPS THEN the system SHALL use a distinct color and line style to differentiate it from other trajectories
3. WHEN plotting coordinates THEN the system SHALL convert latitude/longitude to meters for accurate distance representation
4. WHEN the raw GPS trajectory is displayed THEN the system SHALL include axis labels showing distance in meters

### Requirement 3

**User Story:** As a researcher, I want to visualize the model's corrected output, so that I can evaluate the denoising performance.

#### Acceptance Criteria

1. WHEN the visualization script runs THEN the system SHALL plot the model's corrected GPS coordinates as a 2D trajectory
2. WHEN displaying model output THEN the system SHALL use a distinct color and line style to differentiate it from raw GPS
3. WHEN plotting model output THEN the system SHALL align the corrected trajectory with the same coordinate system as raw GPS
4. WHEN the model output is displayed THEN the system SHALL show the trajectory for all timesteps where predictions are available

### Requirement 4

**User Story:** As a researcher, I want to visualize the actual drone trajectory, so that I can compare it against the raw and corrected GPS data.

#### Acceptance Criteria

1. WHEN ground truth data is available THEN the system SHALL plot the actual drone trajectory as a 2D path
2. WHEN displaying ground truth THEN the system SHALL use a distinct color and line style to differentiate it from GPS and model output
3. WHEN ground truth is not available THEN the system SHALL compute an approximation using smoothed GPS data
4. WHEN the actual trajectory is displayed THEN the system SHALL include it in the same plot as raw GPS and model output

### Requirement 5

**User Story:** As a researcher, I want to see all trajectories on a single plot, so that I can easily compare the model's performance.

#### Acceptance Criteria

1. WHEN the visualization script runs THEN the system SHALL create a single figure containing all trajectory plots
2. WHEN displaying multiple trajectories THEN the system SHALL include a legend identifying each trajectory type
3. WHEN plotting trajectories THEN the system SHALL use equal aspect ratio to prevent distortion
4. WHEN the plot is complete THEN the system SHALL include a title, axis labels, and grid lines for readability

### Requirement 6

**User Story:** As a researcher, I want to save the visualization to a file, so that I can include it in reports and presentations.

#### Acceptance Criteria

1. WHEN the visualization is complete THEN the system SHALL save the plot as a PNG file
2. WHEN saving the plot THEN the system SHALL use a high resolution suitable for publication
3. WHEN the file is saved THEN the system SHALL print the output file path to the console
4. WHEN the output file already exists THEN the system SHALL overwrite it with the new visualization

### Requirement 7

**User Story:** As a researcher, I want to configure the visualization script via command-line arguments, so that I can easily process different flight logs.

#### Acceptance Criteria

1. WHEN running the script THEN the system SHALL accept a command-line argument for the model path
2. WHEN running the script THEN the system SHALL accept a command-line argument for the input CSV file
3. WHEN running the script THEN the system SHALL accept a command-line argument for the output plot file
4. WHEN command-line arguments are not provided THEN the system SHALL use sensible default values
5. WHEN invalid arguments are provided THEN the system SHALL display a helpful error message

### Requirement 8

**User Story:** As a researcher, I want to compute error metrics between trajectories, so that I can quantify the model's performance.

#### Acceptance Criteria

1. WHEN comparing trajectories THEN the system SHALL compute the mean absolute error between raw GPS and ground truth
2. WHEN comparing trajectories THEN the system SHALL compute the mean absolute error between model output and ground truth
3. WHEN error metrics are computed THEN the system SHALL display them in the console output
4. WHEN error metrics are computed THEN the system SHALL include them as text annotations on the plot
