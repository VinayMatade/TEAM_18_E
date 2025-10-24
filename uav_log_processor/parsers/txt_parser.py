"""
TXT parser for text-based logs.

Parses text-based ArduPilot logs and other structured text log formats
to extract GPS, IMU, and sensor data.
"""

from .base import BaseLogParser
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import csv
from io import StringIO


class TxtParser(BaseLogParser):
    """Parser for TXT (text-based) files."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self._supported_extensions = {'.txt', '.log'}  # Support both .txt and .log extensions
        self.logger = logging.getLogger(__name__)
        
        # Message types we want to extract from ArduPilot text logs
        self.target_messages = {
            'GPS', 'GPA', 'IMU', 'IMU2', 'IMU3', 'ATT', 'AHR2', 'AHRS',
            'BARO', 'MAG', 'MAG2', 'MAG3', 'EKF1', 'EKF2', 'EKF3', 'EKF4', 'EKF5',
            'NKF1', 'NKF2', 'NKF3', 'NKF4', 'NKF5', 'XKF1', 'XKF2', 'XKF3', 'XKF4', 'XKF5'
        }
        
        # Format definitions cache
        self.format_definitions = {}
        
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parse text-based log file and extract sensor data.
        
        Args:
            file_path: Path to the text log file
            
        Returns:
            DataFrame with standardized columns
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid or missing text log file: {file_path}")
        
        self.logger.info(f"Parsing text log file: {file_path}")
        
        try:
            # Detect log format
            log_format = self._detect_format(file_path)
            
            if log_format == 'ardupilot':
                df = self._parse_ardupilot_log(file_path)
            elif log_format == 'csv':
                df = self._parse_csv_log(file_path)
            else:
                df = self._parse_generic_text_log(file_path)
            
            if df.empty:
                raise ValueError("No valid sensor data found in file")
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            self.logger.info(f"Successfully parsed {len(df)} records from text log file")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing text log file {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse text log file: {str(e)}")
    
    def _detect_format(self, file_path: str) -> str:
        """Detect the format of the text log file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline().strip() for _ in range(10)]
            
            # Check for ArduPilot format (FMT messages)
            if any('FMT,' in line for line in first_lines):
                return 'ardupilot'
            
            # Check for CSV format
            if any(',' in line and not line.startswith('#') for line in first_lines):
                # Try to detect if it's a proper CSV with headers
                first_data_line = next((line for line in first_lines if line and not line.startswith('#')), '')
                if first_data_line.count(',') > 3:  # Reasonable number of columns
                    return 'csv'
            
            return 'generic'
            
        except Exception as e:
            self.logger.warning(f"Error detecting format: {str(e)}")
            return 'generic'
    
    def _parse_ardupilot_log(self, file_path: str) -> pd.DataFrame:
        """Parse ArduPilot text log format."""
        messages = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # Parse FMT messages to understand data structure
                        if line.startswith('FMT,'):
                            self._parse_fmt_message(line)
                        else:
                            # Parse data messages
                            msg_data = self._parse_ardupilot_message(line)
                            if msg_data:
                                messages.append(msg_data)
                                
                    except Exception as e:
                        self.logger.warning(f"Error parsing line {line_num}: {str(e)}")
                        continue
            
            if messages:
                return self._messages_to_dataframe(messages)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error reading ArduPilot log: {str(e)}")
            return pd.DataFrame()
    
    def _parse_fmt_message(self, line: str):
        """Parse FMT message to understand data structure."""
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                msg_type = parts[3]  # Message type name (index 3, not 4)
                format_str = parts[4]  # Format string (index 4, not 5)
                columns = parts[5:] if len(parts) > 5 else []  # Column names (index 5+)
                
                self.format_definitions[msg_type] = {
                    'format': format_str,
                    'columns': columns
                }
        except Exception as e:
            self.logger.warning(f"Error parsing FMT message: {str(e)}")
    
    def _parse_ardupilot_message(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse individual ArduPilot message line."""
        try:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                return None
            
            msg_type = parts[0]
            
            # Only process messages we're interested in
            if msg_type not in self.target_messages:
                return None
            
            # Get format definition if available
            fmt_def = self.format_definitions.get(msg_type, {})
            columns = fmt_def.get('columns', [])
            values = parts[1:]

            column_map = {}
            if columns and len(columns) == len(values):
                column_map = {columns[i]: values[i] for i in range(len(columns))}

            def _to_float(value: Any) -> float:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return np.nan

            def _get_numeric(
                names: Optional[List[str]] = None,
                fallback_idx: Optional[int] = None,
                *,
                is_latlon: bool = False
            ) -> float:
                if names:
                    for name in names:
                        if name in column_map:
                            val = _to_float(column_map[name])
                            if is_latlon and not np.isnan(val) and abs(val) > 1e6:
                                val /= 1e7
                            return val
                if fallback_idx is not None and fallback_idx < len(parts):
                    val = _to_float(parts[fallback_idx])
                    if is_latlon and not np.isnan(val) and abs(val) > 1e6:
                        val /= 1e7
                    return val
                return np.nan
            
            # Create base message data
            msg_data = {
                'timestamp': np.nan,
                'message_type': msg_type,
                'gps_lat': np.nan, 'gps_lon': np.nan, 'gps_alt': np.nan,
                'imu_ax': np.nan, 'imu_ay': np.nan, 'imu_az': np.nan,
                'imu_gx': np.nan, 'imu_gy': np.nan, 'imu_gz': np.nan,
                'velocity_x': np.nan, 'velocity_y': np.nan, 'velocity_z': np.nan,
                'hdop': np.nan, 'vdop': np.nan, 'fix_type': np.nan
            }
            
            # Extract timestamp (usually first data field)
            timestamp_us = _get_numeric(['TimeUS', 'time_us', 'TIME'], fallback_idx=1)
            if not np.isnan(timestamp_us):
                if timestamp_us > 1e6:
                    msg_data['timestamp'] = timestamp_us / 1e6
                else:
                    msg_data['timestamp'] = timestamp_us
            
            # Parse message-specific data
            if msg_type == 'GPS' and len(parts) >= 4:
                try:
                    msg_data.update({
                        'fix_type': _get_numeric(['Status', 'FixType'], fallback_idx=3),
                        'hdop': _get_numeric(['HDop', 'HAcc'], fallback_idx=6),
                        'gps_lat': _get_numeric(['Lat', 'Latitude'], fallback_idx=8, is_latlon=True),
                        'gps_lon': _get_numeric(['Lng', 'Lon', 'Longitude'], fallback_idx=7, is_latlon=True),
                        'gps_alt': _get_numeric(['Alt', 'Altitude'], fallback_idx=9),
                        'velocity_x': _get_numeric(['Spd', 'Speed'], fallback_idx=10)
                    })
                except (ValueError, IndexError):
                    pass
                    
            elif msg_type == 'GPA' and len(parts) >= 8:
                try:
                    msg_data.update({
                        'vdop': float(parts[3]) if parts[3] else np.nan,
                        'velocity_x': float(parts[4]) if parts[4] else np.nan,
                        'velocity_y': float(parts[5]) if parts[5] else np.nan,
                        'velocity_z': float(parts[6]) if parts[6] else np.nan
                    })
                except (ValueError, IndexError):
                    pass
                    
            elif msg_type in ['IMU', 'IMU2', 'IMU3'] and len(parts) >= 8:
                try:
                    msg_data.update({
                        'imu_ax': float(parts[3]) if parts[3] else np.nan,
                        'imu_ay': float(parts[4]) if parts[4] else np.nan,
                        'imu_az': float(parts[5]) if parts[5] else np.nan,
                        'imu_gx': float(parts[6]) if parts[6] else np.nan,
                        'imu_gy': float(parts[7]) if parts[7] else np.nan,
                        'imu_gz': float(parts[8]) if parts[8] else np.nan
                    })
                except (ValueError, IndexError):
                    pass
                    
            elif msg_type in ['AHR2', 'AHRS'] and len(parts) >= 6:
                try:
                    msg_data.update({
                        'gps_lat': float(parts[3]) if parts[3] else np.nan,
                        'gps_lon': float(parts[4]) if parts[4] else np.nan,
                        'gps_alt': float(parts[5]) if parts[5] else np.nan
                    })
                except (ValueError, IndexError):
                    pass
            
            return msg_data
            
        except Exception as e:
            self.logger.warning(f"Error parsing message: {str(e)}")
            return None
    
    def _parse_csv_log(self, file_path: str) -> pd.DataFrame:
        """Parse CSV format log file."""
        try:
            # Try to read as CSV
            df = pd.read_csv(file_path, comment='#')
            
            # Map common column names to our standard format
            column_mapping = {
                'time': 'timestamp', 'timestamp': 'timestamp', 'time_us': 'timestamp',
                'lat': 'gps_lat', 'latitude': 'gps_lat', 'gps_lat': 'gps_lat',
                'lon': 'gps_lon', 'lng': 'gps_lon', 'longitude': 'gps_lon', 'gps_lon': 'gps_lon',
                'alt': 'gps_alt', 'altitude': 'gps_alt', 'gps_alt': 'gps_alt',
                'acc_x': 'imu_ax', 'accel_x': 'imu_ax', 'ax': 'imu_ax',
                'acc_y': 'imu_ay', 'accel_y': 'imu_ay', 'ay': 'imu_ay',
                'acc_z': 'imu_az', 'accel_z': 'imu_az', 'az': 'imu_az',
                'gyro_x': 'imu_gx', 'gyr_x': 'imu_gx', 'gx': 'imu_gx',
                'gyro_y': 'imu_gy', 'gyr_y': 'imu_gy', 'gy': 'imu_gy',
                'gyro_z': 'imu_gz', 'gyr_z': 'imu_gz', 'gz': 'imu_gz',
                'vel_x': 'velocity_x', 'vx': 'velocity_x',
                'vel_y': 'velocity_y', 'vy': 'velocity_y',
                'vel_z': 'velocity_z', 'vz': 'velocity_z',
                'hdop': 'hdop', 'vdop': 'vdop', 'fix_type': 'fix_type'
            }
            
            # Rename columns
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp if needed
            if 'timestamp' in df.columns:
                # Handle different timestamp formats
                if df['timestamp'].dtype == 'object':
                    # Try to parse as datetime
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) / 1e9
                    except:
                        pass
                elif df['timestamp'].max() > 1e9:
                    # Likely microseconds, convert to seconds
                    df['timestamp'] = df['timestamp'] / 1e6
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV log: {str(e)}")
            return pd.DataFrame()
    
    def _parse_generic_text_log(self, file_path: str) -> pd.DataFrame:
        """Parse generic text log format using regex patterns."""
        messages = []
        
        # Common regex patterns for sensor data
        patterns = {
            'gps': re.compile(r'GPS.*?lat[:\s=]+([+-]?\d*\.?\d+).*?lon[:\s=]+([+-]?\d*\.?\d+).*?alt[:\s=]+([+-]?\d*\.?\d+)', re.IGNORECASE),
            'imu_accel': re.compile(r'(?:ACC|ACCEL).*?X[:\s=]+([+-]?\d*\.?\d+).*?Y[:\s=]+([+-]?\d*\.?\d+).*?Z[:\s=]+([+-]?\d*\.?\d+)', re.IGNORECASE),
            'imu_gyro': re.compile(r'(?:GYR|GYRO).*?X[:\s=]+([+-]?\d*\.?\d+).*?Y[:\s=]+([+-]?\d*\.?\d+).*?Z[:\s=]+([+-]?\d*\.?\d+)', re.IGNORECASE),
            'timestamp': re.compile(r'(?:TIME|TIMESTAMP)[:\s=]+([+-]?\d*\.?\d+)', re.IGNORECASE)
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                timestamp = 0.0
                
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    msg_data = {
                        'timestamp': timestamp,
                        'gps_lat': np.nan, 'gps_lon': np.nan, 'gps_alt': np.nan,
                        'imu_ax': np.nan, 'imu_ay': np.nan, 'imu_az': np.nan,
                        'imu_gx': np.nan, 'imu_gy': np.nan, 'imu_gz': np.nan,
                        'velocity_x': np.nan, 'velocity_y': np.nan, 'velocity_z': np.nan,
                        'hdop': np.nan, 'vdop': np.nan, 'fix_type': np.nan
                    }
                    
                    # Extract timestamp
                    ts_match = patterns['timestamp'].search(line)
                    if ts_match:
                        try:
                            timestamp = float(ts_match.group(1))
                            if timestamp > 1e9:  # Likely microseconds
                                timestamp /= 1e6
                            msg_data['timestamp'] = timestamp
                        except ValueError:
                            pass
                    
                    # Extract GPS data
                    gps_match = patterns['gps'].search(line)
                    if gps_match:
                        try:
                            msg_data.update({
                                'gps_lat': float(gps_match.group(1)),
                                'gps_lon': float(gps_match.group(2)),
                                'gps_alt': float(gps_match.group(3))
                            })
                        except ValueError:
                            pass
                    
                    # Extract IMU accelerometer data
                    accel_match = patterns['imu_accel'].search(line)
                    if accel_match:
                        try:
                            msg_data.update({
                                'imu_ax': float(accel_match.group(1)),
                                'imu_ay': float(accel_match.group(2)),
                                'imu_az': float(accel_match.group(3))
                            })
                        except ValueError:
                            pass
                    
                    # Extract IMU gyroscope data
                    gyro_match = patterns['imu_gyro'].search(line)
                    if gyro_match:
                        try:
                            msg_data.update({
                                'imu_gx': float(gyro_match.group(1)),
                                'imu_gy': float(gyro_match.group(2)),
                                'imu_gz': float(gyro_match.group(3))
                            })
                        except ValueError:
                            pass
                    
                    # Only add message if we extracted some useful data
                    if any(not np.isnan(v) for k, v in msg_data.items() 
                           if k != 'timestamp' and isinstance(v, (int, float))):
                        messages.append(msg_data)
                        timestamp += 0.1  # Increment timestamp for next message
            
            if messages:
                return pd.DataFrame(messages)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error parsing generic text log: {str(e)}")
            return pd.DataFrame()
    
    def _messages_to_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of message dictionaries to DataFrame."""
        if not messages:
            return pd.DataFrame()
        
        df = pd.DataFrame(messages)
        
        # Group by timestamp and combine data from different message types
        df_combined = df.groupby('timestamp').agg({
            'gps_lat': 'first',
            'gps_lon': 'first', 
            'gps_alt': 'first',
            'imu_ax': 'first',
            'imu_ay': 'first',
            'imu_az': 'first',
            'imu_gx': 'first',
            'imu_gy': 'first',
            'imu_gz': 'first',
            'velocity_x': 'first',
            'velocity_y': 'first',
            'velocity_z': 'first',
            'hdop': 'first',
            'vdop': 'first',
            'fix_type': 'first'
        }).reset_index()
        
        return df_combined