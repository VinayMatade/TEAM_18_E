"""
RLOG parser for custom format logs.

Parses custom RLOG files containing firmware metadata, parameter sets, 
calibration data, and sensor readings.
"""

from .base import BaseLogParser
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import struct


class RLogParser(BaseLogParser):
    """Parser for RLOG (custom format) files."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self._supported_extensions = {'.rlog'}
        self.logger = logging.getLogger(__name__)
        
        # Parameter patterns we're interested in for sensor fusion
        self.sensor_params = {
            'COMPASS_': ['OFS_X', 'OFS_Y', 'OFS_Z', 'DIA_X', 'DIA_Y', 'DIA_Z'],
            'INS_': ['GYROFFS_X', 'GYROFFS_Y', 'GYROFFS_Z', 'ACCOFFS_X', 'ACCOFFS_Y', 'ACCOFFS_Z'],
            'ATC_': ['ANG_RLL_P', 'ANG_PIT_P', 'ANG_YAW_P'],
            'WPNAV_': ['SPEED', 'ACCEL', 'RADIUS']
        }
        
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parse RLOG file and extract sensor data and parameters.
        
        Args:
            file_path: Path to the RLOG file
            
        Returns:
            DataFrame with standardized columns
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid or missing RLOG file: {file_path}")
        
        self.logger.info(f"Parsing RLOG file: {file_path}")
        
        try:
            # Read file content
            with open(file_path, 'rb') as f:
                content = f.read()
            
            # Extract firmware metadata
            metadata = self._extract_metadata(content)
            
            # Extract parameter sets
            parameters = self._extract_parameters(content)
            
            # Extract sensor data (if any embedded)
            sensor_data = self._extract_sensor_data(content)
            
            # Create DataFrame from extracted data
            df = self._create_dataframe(metadata, parameters, sensor_data)
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            self.logger.info(f"Successfully parsed {len(df)} records from RLOG file")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing RLOG file {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse RLOG file: {str(e)}")
    
    def _extract_metadata(self, content: bytes) -> Dict[str, Any]:
        """Extract firmware and vehicle metadata from RLOG content."""
        metadata = {}
        
        try:
            # Convert to string for text parsing (handle encoding issues)
            text_content = content.decode('utf-8', errors='ignore')
            
            # Extract ArduCopter version
            version_match = re.search(r'ArduCopter V([\d\.]+)', text_content)
            if version_match:
                metadata['firmware_version'] = version_match.group(1)
            
            # Extract ChibiOS version
            chibios_match = re.search(r'ChibiOS: ([a-f0-9]+)', text_content)
            if chibios_match:
                metadata['chibios_version'] = chibios_match.group(1)
            
            # Extract hardware info
            hw_match = re.search(r'(Pixhawk\w+)\s+([A-F0-9\s]+)', text_content)
            if hw_match:
                metadata['hardware'] = hw_match.group(1)
                metadata['hardware_id'] = hw_match.group(2).replace(' ', '')
            
            # Extract frame type
            frame_match = re.search(r'Frame: (\w+/\w+)', text_content)
            if frame_match:
                metadata['frame_type'] = frame_match.group(1)
                
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {str(e)}")
        
        return metadata
    
    def _extract_parameters(self, content: bytes) -> Dict[str, float]:
        """Extract parameter sets from RLOG content."""
        parameters = {}
        
        try:
            # Convert to string for parameter parsing
            text_content = content.decode('utf-8', errors='ignore')
            
            # Extract parameters using regex patterns
            # Parameters appear to be in format: PARAM_NAME followed by value
            param_patterns = [
                r'([A-Z_]+[A-Z0-9_]*)\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)',
                r'([A-Z_]+[A-Z0-9_]*)\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            ]
            
            for pattern in param_patterns:
                matches = re.findall(pattern, text_content)
                for param_name, value_str in matches:
                    # Filter for sensor-related parameters
                    if self._is_sensor_parameter(param_name):
                        try:
                            parameters[param_name] = float(value_str)
                        except ValueError:
                            continue
                            
        except Exception as e:
            self.logger.warning(f"Error extracting parameters: {str(e)}")
        
        return parameters
    
    def _is_sensor_parameter(self, param_name: str) -> bool:
        """Check if parameter is sensor-related."""
        for prefix, suffixes in self.sensor_params.items():
            if param_name.startswith(prefix):
                if any(param_name.endswith(suffix) for suffix in suffixes):
                    return True
                # Also include some general sensor parameters
                if any(keyword in param_name for keyword in ['GPS', 'IMU', 'BARO', 'MAG']):
                    return True
        return False
    
    def _extract_sensor_data(self, content: bytes) -> List[Dict[str, Any]]:
        """Extract embedded sensor data from RLOG content."""
        sensor_data = []
        
        try:
            # Look for binary sensor data patterns
            # RLOG files may contain embedded sensor readings
            offset = 0
            while offset < len(content) - 32:  # Ensure we have enough bytes
                try:
                    # Look for potential timestamp markers (common in binary logs)
                    if self._is_potential_sensor_record(content[offset:offset+32]):
                        record = self._parse_sensor_record(content[offset:offset+32])
                        if record:
                            sensor_data.append(record)
                    offset += 1
                except:
                    offset += 1
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error extracting sensor data: {str(e)}")
        
        return sensor_data
    
    def _is_potential_sensor_record(self, data: bytes) -> bool:
        """Check if byte sequence could be a sensor record."""
        if len(data) < 32:
            return False
        
        try:
            # Look for reasonable timestamp values (microseconds since boot)
            timestamp = struct.unpack('<Q', data[0:8])[0]  # Little-endian uint64
            
            # Reasonable timestamp range (not too small, not too large)
            if 1000 < timestamp < 1e15:  # Between 1ms and ~31 years in microseconds
                return True
                
            # Try different timestamp formats
            timestamp_ms = struct.unpack('<I', data[0:4])[0]  # 32-bit milliseconds
            if 1000 < timestamp_ms < 1e10:  # Reasonable millisecond range
                return True
                
        except struct.error:
            pass
        
        return False
    
    def _parse_sensor_record(self, data: bytes) -> Optional[Dict[str, Any]]:
        """Parse a potential sensor record from binary data."""
        try:
            # Try to extract timestamp and sensor values
            # This is speculative parsing based on common binary log formats
            
            # Try 64-bit timestamp first
            try:
                timestamp_us = struct.unpack('<Q', data[0:8])[0]
                timestamp = timestamp_us / 1e6  # Convert to seconds
                data_offset = 8
            except:
                # Try 32-bit timestamp
                timestamp_ms = struct.unpack('<I', data[0:4])[0]
                timestamp = timestamp_ms / 1000.0  # Convert to seconds
                data_offset = 4
            
            # Extract potential sensor values (assuming float32)
            values = []
            for i in range(data_offset, min(len(data), data_offset + 24), 4):
                try:
                    value = struct.unpack('<f', data[i:i+4])[0]
                    # Filter out obviously invalid values
                    if not (np.isnan(value) or np.isinf(value) or abs(value) > 1e6):
                        values.append(value)
                except:
                    break
            
            # Only return if we have reasonable data
            if len(values) >= 6 and 0 < timestamp < 1e10:
                return {
                    'timestamp': timestamp,
                    'gps_lat': values[0] if len(values) > 0 else np.nan,
                    'gps_lon': values[1] if len(values) > 1 else np.nan,
                    'gps_alt': values[2] if len(values) > 2 else np.nan,
                    'imu_ax': values[3] if len(values) > 3 else np.nan,
                    'imu_ay': values[4] if len(values) > 4 else np.nan,
                    'imu_az': values[5] if len(values) > 5 else np.nan,
                    'imu_gx': values[6] if len(values) > 6 else np.nan,
                    'imu_gy': values[7] if len(values) > 7 else np.nan,
                    'imu_gz': values[8] if len(values) > 8 else np.nan,
                    'velocity_x': np.nan,
                    'velocity_y': np.nan,
                    'velocity_z': np.nan,
                    'hdop': np.nan,
                    'vdop': np.nan,
                    'fix_type': np.nan
                }
                
        except Exception as e:
            pass
        
        return None
    
    def _create_dataframe(self, metadata: Dict[str, Any], 
                         parameters: Dict[str, float], 
                         sensor_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame from extracted RLOG data."""
        
        # If we have sensor data, use it
        if sensor_data:
            df = pd.DataFrame(sensor_data)
        else:
            # Create a minimal DataFrame with metadata/parameters as context
            # Since RLOG files may not contain time-series sensor data,
            # we create a single record with available information
            base_record = {
                'timestamp': 0.0,  # Placeholder timestamp
                'gps_lat': np.nan,
                'gps_lon': np.nan,
                'gps_alt': np.nan,
                'imu_ax': np.nan,
                'imu_ay': np.nan,
                'imu_az': np.nan,
                'imu_gx': np.nan,
                'imu_gy': np.nan,
                'imu_gz': np.nan,
                'velocity_x': np.nan,
                'velocity_y': np.nan,
                'velocity_z': np.nan,
                'hdop': np.nan,
                'vdop': np.nan,
                'fix_type': np.nan
            }
            
            # Add calibration data as sensor offsets if available
            if 'INS_ACCOFFS_X' in parameters:
                base_record['imu_ax'] = parameters['INS_ACCOFFS_X']
            if 'INS_ACCOFFS_Y' in parameters:
                base_record['imu_ay'] = parameters['INS_ACCOFFS_Y']
            if 'INS_ACCOFFS_Z' in parameters:
                base_record['imu_az'] = parameters['INS_ACCOFFS_Z']
                
            if 'INS_GYROFFS_X' in parameters:
                base_record['imu_gx'] = parameters['INS_GYROFFS_X']
            if 'INS_GYROFFS_Y' in parameters:
                base_record['imu_gy'] = parameters['INS_GYROFFS_Y']
            if 'INS_GYROFFS_Z' in parameters:
                base_record['imu_gz'] = parameters['INS_GYROFFS_Z']
            
            df = pd.DataFrame([base_record])
        
        # Store metadata and parameters as attributes for later use
        df.attrs['metadata'] = metadata
        df.attrs['parameters'] = parameters
        
        return df