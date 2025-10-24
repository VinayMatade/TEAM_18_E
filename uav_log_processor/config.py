"""
Configuration management for UAV log processing.

Provides centralized configuration handling with validation and defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import json
from pathlib import Path


@dataclass
class ProcessingConfig:
    """Configuration class for UAV log processing pipeline."""
    
    # Data synchronization settings
    target_frequency: float = 15.0  # Hz - target sampling rate for synchronized data
    interpolation_method: str = "linear"  # Method for filling missing timestamps
    max_gap_seconds: float = 1.0  # Maximum gap to interpolate (larger gaps are dropped)
    min_data_coverage: float = 0.5  # Minimum fraction of data required to keep a time period
    
    # Motion classification settings
    accel_threshold: float = 0.5  # m/s² - threshold for stationary detection
    gyro_threshold: float = 0.1  # rad/s - threshold for stationary detection
    min_stationary_duration: float = 3.0  # seconds - minimum duration for stationary segment
    motion_window_size: float = 5.0  # seconds - sliding window for motion smoothing
    
    # Coordinate system settings
    coordinate_system: str = "ENU"  # East-North-Up coordinate system
    home_point_method: str = "first_fix"  # Method to determine home point
    
    # Ground truth generation settings
    fusion_method: str = "ekf"  # Sensor fusion method: "ekf", "complementary", or "simple"
    drift_correction: bool = True  # Enable IMU drift correction at stationary points
    smoothing_method: str = "cubic_spline"  # Smoothing for transitions
    
    # Data quality settings
    min_gps_fix_type: int = 3  # Minimum GPS fix type to accept
    max_hdop: float = 5.0  # Maximum HDOP value to accept
    max_data_loss_warning: float = 0.1  # Warn if more than 10% data is lost
    
    # Dataset formatting settings
    normalization_method: str = "zscore"  # Normalization method for features
    train_ratio: float = 0.7  # Fraction of data for training
    val_ratio: float = 0.15  # Fraction of data for validation
    test_ratio: float = 0.15  # Fraction of data for testing
    
    # Output settings
    output_dir: str = "output"  # Directory for output files
    save_intermediate: bool = True  # Save intermediate processing results
    create_visualizations: bool = True  # Generate trajectory visualizations
    
    # Processing settings
    chunk_size: Optional[int] = None  # Process large files in chunks (None = auto)
    n_jobs: int = 1  # Number of parallel jobs for processing
    verbose: bool = False  # Enable verbose logging
    
    # Parser-specific settings
    parser_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if self.target_frequency <= 0:
            raise ValueError("target_frequency must be positive")
        
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between 0 and 1")
        
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1")
        
        if not 0 < self.test_ratio < 1:
            raise ValueError("test_ratio must be between 0 and 1")
        
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")
        
        if self.accel_threshold < 0:
            raise ValueError("accel_threshold must be non-negative")
        
        if self.gyro_threshold < 0:
            raise ValueError("gyro_threshold must be non-negative")
        
        if self.min_stationary_duration <= 0:
            raise ValueError("min_stationary_duration must be positive")
        
        if self.coordinate_system not in ["ENU", "NED"]:
            raise ValueError("coordinate_system must be 'ENU' or 'NED'")
        
        if self.interpolation_method not in ["linear", "cubic", "nearest"]:
            raise ValueError("interpolation_method must be 'linear', 'cubic', or 'nearest'")
        
        if self.fusion_method not in ["ekf", "complementary", "simple"]:
            raise ValueError("fusion_method must be 'ekf', 'complementary', or 'simple'")
        
        if self.normalization_method not in ["zscore", "minmax", "robust"]:
            raise ValueError("normalization_method must be 'zscore', 'minmax', or 'robust'")
    
    @classmethod
    def from_file(cls, config_path: str) -> 'ProcessingConfig':
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            ProcessingConfig instance
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def to_file(self, config_path: str):
        """
        Save configuration to JSON file.
        
        Args:
            config_path: Path where to save the configuration
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary, handling dataclass fields
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                config_dict[key] = value
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def get_parser_config(self, parser_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific parser.
        
        Args:
            parser_name: Name of the parser (e.g., 'tlog', 'bin')
            
        Returns:
            Configuration dictionary for the parser
        """
        return self.parser_configs.get(parser_name, {})
    
    def set_parser_config(self, parser_name: str, config: Dict[str, Any]):
        """
        Set configuration for a specific parser.
        
        Args:
            parser_name: Name of the parser
            config: Configuration dictionary
        """
        self.parser_configs[parser_name] = config
    
    def copy(self) -> 'ProcessingConfig':
        """Create a copy of the configuration."""
        return ProcessingConfig(
            target_frequency=self.target_frequency,
            interpolation_method=self.interpolation_method,
            max_gap_seconds=self.max_gap_seconds,
            min_data_coverage=self.min_data_coverage,
            accel_threshold=self.accel_threshold,
            gyro_threshold=self.gyro_threshold,
            min_stationary_duration=self.min_stationary_duration,
            motion_window_size=self.motion_window_size,
            coordinate_system=self.coordinate_system,
            home_point_method=self.home_point_method,
            fusion_method=self.fusion_method,
            drift_correction=self.drift_correction,
            smoothing_method=self.smoothing_method,
            min_gps_fix_type=self.min_gps_fix_type,
            max_hdop=self.max_hdop,
            max_data_loss_warning=self.max_data_loss_warning,
            normalization_method=self.normalization_method,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            test_ratio=self.test_ratio,
            output_dir=self.output_dir,
            save_intermediate=self.save_intermediate,
            create_visualizations=self.create_visualizations,
            chunk_size=self.chunk_size,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            parser_configs=self.parser_configs.copy()
        )