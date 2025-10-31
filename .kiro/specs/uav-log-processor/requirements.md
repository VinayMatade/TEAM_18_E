# Requirements Document

## Introduction

This system processes UAV flight logs from multiple formats (.tlog, .bin, .rlog, .txt) to create clean, time-aligned datasets for training Temporal Convolutional Networks (TCN) that reduce GPS position error. The system extracts sensor data, performs sensor fusion, generates ground-truth positions, and outputs structured training datasets.

## Glossary

- **UAV_Log_Processor**: The main system that processes multiple UAV log formats
- **TCN_Dataset**: Time-aligned dataset formatted for Temporal Convolutional Network training
- **GPS_Error_Vector**: The difference between raw GPS position and corrected ground-truth position
- **ENU_Coordinates**: East-North-Up coordinate system relative to home point
- **Stationary_Segment**: Time periods where drone acceleration and gyroscope readings are below movement thresholds
- **Ground_Truth_Position**: Corrected position derived from stationary anchors and IMU integration
- **Sensor_Fusion**: Process of combining GPS, IMU, and EKF data into synchronized streams
- **ArduPilot_Log**: Flight log format from ArduPilot autopilot system
- **PX4_Log**: Flight log format from PX4 autopilot system

## Requirements

### Requirement 1

**User Story:** As a machine learning researcher, I want to process multiple UAV log formats simultaneously, so that I can create comprehensive training datasets from diverse flight data sources.

#### Acceptance Criteria

1. WHEN UAV_Log_Processor receives .tlog, .bin, .rlog, .txt, and .log file paths, THE UAV_Log_Processor SHALL parse all available log formats
2. WHERE only .log files are present and no .txt files exist, THE UAV_Log_Processor SHALL rename .log files to .txt extension for processing
3. THE UAV_Log_Processor SHALL extract GPS data including latitude, longitude, altitude, HDOP, VDOP, fix type, and velocity from GPS_RAW_INT or GLOBAL_POSITION_INT messages
4. THE UAV_Log_Processor SHALL extract IMU data including accelerometer, gyroscope, and derived orientation values
5. WHERE EKF data is available, THE UAV_Log_Processor SHALL extract estimated position and velocity information
6. THE UAV_Log_Processor SHALL synchronize all data streams to a uniform timestamp axis with 10-20 Hz sampling rate

### Requirement 2

**User Story:** As a data scientist, I want synchronized and clean sensor data, so that I can train accurate position correction models without data quality issues.

#### Acceptance Criteria

1. THE UAV_Log_Processor SHALL convert all timestamps to microseconds or seconds for uniform time alignment
2. THE UAV_Log_Processor SHALL use linear interpolation for missing timestamps in synchronized streams
3. THE UAV_Log_Processor SHALL drop time periods with missing GPS or IMU data
4. THE UAV_Log_Processor SHALL convert latitude/longitude/altitude coordinates to local ENU_Coordinates relative to home point
5. THE UAV_Log_Processor SHALL preserve original GPS coordinate values throughout the processing pipeline without unintended modifications or resets
6. IF more than 10% of samples are dropped due to missing data, THEN THE UAV_Log_Processor SHALL provide warnings to the user

### Requirement 3

**User Story:** As a researcher studying GPS accuracy, I want to identify stationary vs moving flight segments, so that I can establish reliable ground-truth position anchors.

#### Acceptance Criteria

1. THE UAV_Log_Processor SHALL use IMU acceleration and gyroscope magnitude thresholds to classify time segments
2. THE UAV_Log_Processor SHALL label segments as either Stationary_Segment or moving based on motion detection
3. DURING Stationary_Segment periods, THE UAV_Log_Processor SHALL calculate ground-truth position as the mean GPS position of that period
4. DURING moving periods, THE UAV_Log_Processor SHALL estimate Ground_Truth_Position using velocity integration from IMU with drift correction
5. THE UAV_Log_Processor SHALL apply bias correction to minimize position jumps at stationary-to-moving transitions

### Requirement 4

**User Story:** As a machine learning engineer, I want accurate GPS error targets, so that I can train TCN models to predict and correct GPS positioning errors.

#### Acceptance Criteria

1. THE UAV_Log_Processor SHALL compute GPS_Error_Vector for each timestamp as the difference between raw GPS position and Ground_Truth_Position
2. THE UAV_Log_Processor SHALL calculate error components for x, y, and z coordinates separately
3. THE UAV_Log_Processor SHALL compute error magnitude as the Euclidean norm of the error vector
4. THE UAV_Log_Processor SHALL output error vectors as prediction targets for TCN training
5. THE UAV_Log_Processor SHALL ensure error calculations maintain temporal consistency across flight segments

### Requirement 5

**User Story:** As a model developer, I want properly formatted and split datasets, so that I can train, validate, and test TCN models effectively.

#### Acceptance Criteria

1. THE UAV_Log_Processor SHALL combine all features into a single time-aligned TCN_Dataset
2. THE UAV_Log_Processor SHALL include timestamp, GPS coordinates, IMU data, velocity, HDOP, VDOP, fix type, and GPS error vectors in the dataset
3. THE UAV_Log_Processor SHALL normalize IMU and GPS features using mean/standard deviation normalization
4. THE UAV_Log_Processor SHALL split data sequentially into 70% training, 15% validation, and 15% test sets
5. THE UAV_Log_Processor SHALL output train.csv, valid.csv, and test.csv files with consistent formatting
6. THE UAV_Log_Processor SHALL store all numeric data in full precision format without scientific notation to ensure data usability

### Requirement 6

**User Story:** As a data analyst, I want comprehensive metadata and visualization, so that I can verify data quality and understand the processing results.

#### Acceptance Criteria

1. THE UAV_Log_Processor SHALL generate metadata.json describing each feature column, source, sampling rate, and normalization statistics
2. THE UAV_Log_Processor SHALL create visualization.png showing raw GPS trajectory versus corrected Ground_Truth_Position trajectory
3. THE UAV_Log_Processor SHALL save aligned_full.csv containing all synchronized raw data for reproducibility
4. WHERE multiple GPS units are present, THE UAV_Log_Processor SHALL use the most reliable GPS source with fix_type ≥ 3 and lowest HDOP
5. THE UAV_Log_Processor SHALL ensure no temporal overlap between train, validation, and test sets when multiple flights exist

### Requirement 7

**User Story:** As a data quality engineer, I want robust coordinate validation and file format handling, so that I can ensure data integrity and prevent processing errors.

#### Acceptance Criteria

1. THE UAV_Log_Processor SHALL validate GPS coordinates to ensure latitude values are between -90 and 90 degrees
2. THE UAV_Log_Processor SHALL validate GPS coordinates to ensure longitude values are between -180 and 180 degrees
3. THE UAV_Log_Processor SHALL ensure coordinate processing logic never incorrectly calculates or resets latitude and longitude values to zero during data transformation
4. THE UAV_Log_Processor SHALL automatically detect and rename .log files to .txt extension when no .txt files are present in the input directory
5. THE UAV_Log_Processor SHALL log all coordinate validation failures and file format conversions for audit purposes