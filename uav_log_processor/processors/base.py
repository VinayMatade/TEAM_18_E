"""
Base classes for data processors.

Defines common interfaces for all processing components in the pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd


class BaseProcessor(ABC):
    """Abstract base class for data processors."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the processor with optional configuration.
        
        Args:
            config: Optional configuration dictionary for processor settings
        """
        self.config = config or {}
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process input data and return processed result.
        
        Args:
            data: Input DataFrame to process
            
        Returns:
            Processed DataFrame
        """
        pass
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate that input data has required columns and format.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        return isinstance(data, pd.DataFrame) and not data.empty


class BaseSynchronizer(BaseProcessor):
    """Base class for data synchronization processors."""
    
    @abstractmethod
    def synchronize_streams(self, data_streams: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Synchronize multiple data streams to uniform timestamps.
        
        Args:
            data_streams: Dictionary of stream name to DataFrame
            
        Returns:
            Synchronized DataFrame with aligned timestamps
        """
        pass


class BaseClassifier(BaseProcessor):
    """Base class for classification processors."""
    
    @abstractmethod
    def classify(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify data points into categories.
        
        Args:
            data: Input DataFrame to classify
            
        Returns:
            Series with classification labels
        """
        pass


class BaseGenerator(BaseProcessor):
    """Base class for data generation processors."""
    
    @abstractmethod
    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate new data based on input.
        
        Args:
            data: Input DataFrame
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame with generated data
        """
        pass


class BaseCalculator(BaseProcessor):
    """Base class for calculation processors."""
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate metrics by comparing data with reference.
        
        Args:
            data: Primary data DataFrame
            reference: Reference data DataFrame
            
        Returns:
            DataFrame with calculated metrics
        """
        pass


class BaseFormatter(BaseProcessor):
    """Base class for data formatting processors."""
    
    @abstractmethod
    def format_dataset(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Format data for machine learning consumption.
        
        Args:
            data: Input DataFrame to format
            
        Returns:
            Tuple of (formatted DataFrame, metadata dictionary)
        """
        pass
    
    @abstractmethod
    def split_dataset(self, data: pd.DataFrame, 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            data: Input DataFrame to split
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        pass