"""
UAV Log Processor - A system for processing multiple UAV log formats into ML-ready datasets.

This package provides tools for parsing various UAV log formats (.tlog, .bin, .rlog, .txt),
synchronizing sensor data, generating ground truth positions, and creating training datasets
for Temporal Convolutional Networks (TCN) to reduce GPS position error.
"""

__version__ = "1.0.0"
__author__ = "UAV Log Processor Team"

from .config import ProcessingConfig
from .pipeline import UAVLogProcessor

__all__ = ["ProcessingConfig", "UAVLogProcessor"]