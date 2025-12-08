"""
UAV Log Processor - CLI for GPS-IMU fusion model training and inference.

This package provides a command-line interface for:
- Cleaning and converting raw UAV log files to CSV
- Building noise banks for training
- Training TCN models for GPS correction
- Running inference to correct GPS drift
"""

__version__ = "1.0.0"
__author__ = "UAV Log Processor Team"

# Import key classes for easy access
from uav_log_processor.coordinate_transformer import CoordinateTransformer
from uav_log_processor.data_structures import TrajectoryData, ErrorMetrics

__all__ = ["CoordinateTransformer", "TrajectoryData", "ErrorMetrics"]