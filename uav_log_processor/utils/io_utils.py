"""
File I/O utilities.

This module provides common file handling operations.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional


class FileHandler:
    """Handles file I/O operations for the UAV log processor."""
    
    def __init__(self, config=None):
        """Initialize file handler."""
        self.config = config or {}
    
    def save_json(self, data: Dict[str, Any], file_path: str):
        """
        Save dictionary to JSON file.
        
        Args:
            data: Dictionary to save
            file_path: Path where to save the file
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_json(self, file_path: str) -> Dict[str, Any]:
        """
        Load dictionary from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded dictionary
        """
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_csv(self, data: pd.DataFrame, file_path: str, **kwargs):
        """
        Save DataFrame to CSV file.
        
        Args:
            data: DataFrame to save
            file_path: Path where to save the file
            **kwargs: Additional arguments for pandas.to_csv
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(path, index=False, **kwargs)
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load DataFrame from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pandas.read_csv
            
        Returns:
            Loaded DataFrame
        """
        return pd.read_csv(file_path, **kwargs)
    
    def find_log_files(self, directory: str, extensions: Optional[List[str]] = None) -> List[str]:
        """
        Find log files in directory with specified extensions.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions to look for
            
        Returns:
            List of found file paths
        """
        if extensions is None:
            extensions = ['.tlog', '.bin', '.rlog', '.txt', '.log']
        
        directory_path = Path(directory)
        found_files = []
        
        for ext in extensions:
            found_files.extend(directory_path.glob(f"*{ext}"))
        
        return [str(f) for f in sorted(found_files)]
    
    def validate_output_directory(self, output_dir: str) -> bool:
        """
        Validate that output directory can be created/written to.
        
        Args:
            output_dir: Output directory path
            
        Returns:
            True if directory is valid, False otherwise
        """
        try:
            path = Path(output_dir)
            path.mkdir(parents=True, exist_ok=True)
            
            # Test write access
            test_file = path / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            
            return True
        except Exception:
            return False