"""Data structures for trajectory visualization and analysis."""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class TrajectoryData:
    """Container for trajectory information.
    
    Attributes:
        timestamps: Time in seconds (1D array)
        x: X position in meters (1D array)
        y: Y position in meters (1D array)
        label: Trajectory label for legend
        color: Plot color (matplotlib color string)
        style: Line style ('-', '--', ':', '-.')
    """
    timestamps: np.ndarray
    x: np.ndarray
    y: np.ndarray
    label: str
    color: str
    style: str = '-'
    
    def __post_init__(self):
        """Validate trajectory data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate trajectory data consistency.
        
        Raises:
            ValueError: If data is invalid or inconsistent
        """
        # Check that all arrays are numpy arrays
        if not isinstance(self.timestamps, np.ndarray):
            raise ValueError("timestamps must be a numpy array")
        if not isinstance(self.x, np.ndarray):
            raise ValueError("x must be a numpy array")
        if not isinstance(self.y, np.ndarray):
            raise ValueError("y must be a numpy array")
        
        # Check that all arrays are 1D
        if self.timestamps.ndim != 1:
            raise ValueError(f"timestamps must be 1D, got shape {self.timestamps.shape}")
        if self.x.ndim != 1:
            raise ValueError(f"x must be 1D, got shape {self.x.shape}")
        if self.y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {self.y.shape}")
        
        # Check that all arrays have the same length
        n_timestamps = len(self.timestamps)
        n_x = len(self.x)
        n_y = len(self.y)
        
        if not (n_timestamps == n_x == n_y):
            raise ValueError(
                f"All arrays must have the same length. "
                f"Got timestamps: {n_timestamps}, x: {n_x}, y: {n_y}"
            )
        
        # Check that we have at least one data point
        if n_timestamps == 0:
            raise ValueError("Trajectory must contain at least one data point")
        
        # Check for NaN or Inf values
        if np.any(np.isnan(self.timestamps)) or np.any(np.isinf(self.timestamps)):
            raise ValueError("timestamps contains NaN or Inf values")
        if np.any(np.isnan(self.x)) or np.any(np.isinf(self.x)):
            raise ValueError("x contains NaN or Inf values")
        if np.any(np.isnan(self.y)) or np.any(np.isinf(self.y)):
            raise ValueError("y contains NaN or Inf values")
        
        # Check that timestamps are monotonically increasing
        if len(self.timestamps) > 1:
            if not np.all(np.diff(self.timestamps) >= 0):
                raise ValueError("timestamps must be monotonically increasing")
        
        # Validate label is non-empty string
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        
        # Validate color is non-empty string
        if not isinstance(self.color, str) or not self.color.strip():
            raise ValueError("color must be a non-empty string")
        
        # Validate style is one of the valid matplotlib line styles
        valid_styles = ['-', '--', ':', '-.']
        if self.style not in valid_styles:
            raise ValueError(
                f"style must be one of {valid_styles}, got '{self.style}'"
            )
    
    @property
    def length(self) -> int:
        """Return the number of points in the trajectory."""
        return len(self.timestamps)
    
    def get_bounds(self) -> tuple[float, float, float, float]:
        """Get the bounding box of the trajectory.
        
        Returns:
            Tuple of (min_x, max_x, min_y, max_y)
        """
        return (
            float(np.min(self.x)),
            float(np.max(self.x)),
            float(np.min(self.y)),
            float(np.max(self.y))
        )


@dataclass
class ErrorMetrics:
    """GPS error statistics.
    
    Attributes:
        mae: Mean Absolute Error in meters
        rmse: Root Mean Square Error in meters
        max_error: Maximum error in meters
        median_error: Median error in meters (optional)
        std_error: Standard deviation in meters (optional)
    """
    mae: float
    rmse: float
    max_error: float
    median_error: Optional[float] = None
    std_error: Optional[float] = None
    
    def __post_init__(self):
        """Validate error metrics after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate error metrics.
        
        Raises:
            ValueError: If metrics are invalid
        """
        # Check that required metrics are non-negative
        if self.mae < 0:
            raise ValueError(f"mae must be non-negative, got {self.mae}")
        if self.rmse < 0:
            raise ValueError(f"rmse must be non-negative, got {self.rmse}")
        if self.max_error < 0:
            raise ValueError(f"max_error must be non-negative, got {self.max_error}")
        
        # Check that required metrics are not NaN or Inf
        if np.isnan(self.mae) or np.isinf(self.mae):
            raise ValueError("mae cannot be NaN or Inf")
        if np.isnan(self.rmse) or np.isinf(self.rmse):
            raise ValueError("rmse cannot be NaN or Inf")
        if np.isnan(self.max_error) or np.isinf(self.max_error):
            raise ValueError("max_error cannot be NaN or Inf")
        
        # Check optional metrics if provided
        if self.median_error is not None:
            if self.median_error < 0:
                raise ValueError(f"median_error must be non-negative, got {self.median_error}")
            if np.isnan(self.median_error) or np.isinf(self.median_error):
                raise ValueError("median_error cannot be NaN or Inf")
        
        if self.std_error is not None:
            if self.std_error < 0:
                raise ValueError(f"std_error must be non-negative, got {self.std_error}")
            if np.isnan(self.std_error) or np.isinf(self.std_error):
                raise ValueError("std_error cannot be NaN or Inf")
        
        # Validate mathematical relationships
        # RMSE should be >= MAE (by Cauchy-Schwarz inequality)
        if self.rmse < self.mae - 1e-6:  # Small tolerance for floating point
            raise ValueError(
                f"RMSE ({self.rmse}) should be >= MAE ({self.mae})"
            )
        
        # Max error should be >= MAE
        if self.max_error < self.mae - 1e-6:
            raise ValueError(
                f"max_error ({self.max_error}) should be >= MAE ({self.mae})"
            )
    
    def __str__(self) -> str:
        """Return a formatted string representation of the metrics."""
        lines = [
            f"MAE: {self.mae:.2f} m",
            f"RMSE: {self.rmse:.2f} m",
            f"Max Error: {self.max_error:.2f} m"
        ]
        
        if self.median_error is not None:
            lines.append(f"Median Error: {self.median_error:.2f} m")
        
        if self.std_error is not None:
            lines.append(f"Std Dev: {self.std_error:.2f} m")
        
        return "\n".join(lines)
