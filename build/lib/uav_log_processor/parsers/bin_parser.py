"""
BIN parser for ArduPilot binary logs.

Parses ArduPilot .bin files using pymavlink to extract GPS, IMU, EKF, and other sensor data.
"""

from .base import BaseLogParser
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import ardupilotmega as mavlink
except ImportError:
    raise ImportError("pymavlink is required for BIN parsing. Install with: pip install pymavlink")


class BinParser(BaseLogParser):
    """Parser for BIN (ArduPilot binary) files."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self._supported_extensions = {'.bin'}
        self.logger = logging.getLogger(__name__)
        
        # ArduPilot log message types we want to extract
        self.target_messages = {
            # GPS data
            'GPS', 'GPA', 'GPS2', 'GPS_RAW_INT', 'GLOBAL_POSITION_INT',
            # IMU data
            'IMU', 'IMU2', 'IMU3', 'RAW_IMU', 'SCALED_IMU2', 'IMT', 'IMT2', 'IMT3',
            # EKF data
            'XKF1', 'XKF2', 'XKF3', 'XKF4', 'XKF5', 'XKQ1', 'XKQ2', 'XKV1', 'XKV2',
            'EKF1', 'EKF2', 'EKF3', 'EKF4', 'EKF5', 'NKF1', 'NKF2', 'NKF3', 'NKF4', 'NKF5',
            # PID data
            'PIDR', 'PIDP', 'PIDY', 'PID',
            # RC data
            'RCIN', 'RCOU', 'SERVO_OUTPUT_RAW',
            # Sensor data
            'BARO', 'BARD', 'ARSP', 'VIBE',
            # GNSS quality
            'UBX1', 'UBX2', 'SBR1', 'SBR2', 'GRAW', 'GRX1', 'GRX2',
            # Power/health
            'RSSI', 'POWR', 'BAT', 'BAT2', 'BCL', 'BCL2',
            # Attitude
            'ATT', 'AHR2', 'AHRS'
        }
    
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parse ArduPilot BIN file and extract sensor data.
        
        Args:
            file_path: Path to the BIN file
            
        Returns:
            DataFrame with standardized columns
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid or missing BIN file: {file_path}")
        
        self.logger.info(f"Parsing BIN file: {file_path}")
        
        try:
            # Open ArduPilot log connection
            mav_conn = mavutil.mavlink_connection(file_path)
            
            # Extract messages
            messages = self._extract_messages(mav_conn)
            
            if not messages:
                raise ValueError("No valid ArduPilot messages found in file")
            
            # Convert to DataFrame
            df = self._messages_to_dataframe(messages)
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            self.logger.info(f"Successfully parsed {len(df)} records from BIN file")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing BIN file {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse BIN file: {str(e)}")
    
    def _extract_messages(self, mav_conn) -> List[Dict[str, Any]]:
        """Extract relevant ArduPilot messages from connection."""
        messages = []
        
        while True:
            try:
                msg = mav_conn.recv_match(blocking=False)
                if msg is None:
                    break
                
                # Process both MAVLink messages and ArduPilot log messages
                msg_type = msg.get_type()
                if msg_type in self.target_messages:
                    msg_data = self._parse_message(msg)
                    if msg_data:
                        messages.append(msg_data)
                        
            except Exception as e:
                self.logger.warning(f"Error reading message: {str(e)}")
                continue
        
        return messages
    
    def _parse_message(self, msg) -> Optional[Dict[str, Any]]:
        """Parse individual ArduPilot message into standardized format."""
        msg_type = msg.get_type()
        
        # Extract timestamp - ArduPilot uses TimeUS (microseconds)
        timestamp = self._extract_timestamp(msg)
        if np.isnan(timestamp):
            return None
        
        base_data = {
            'timestamp': timestamp,
            'message_type': msg_type,
            'gps_lat': np.nan, 'gps_lon': np.nan, 'gps_alt': np.nan,
            'imu_ax': np.nan, 'imu_ay': np.nan, 'imu_az': np.nan,
            'imu_gx': np.nan, 'imu_gy': np.nan, 'imu_gz': np.nan,
            'velocity_x': np.nan, 'velocity_y': np.nan, 'velocity_z': np.nan,
            'hdop': np.nan, 'vdop': np.nan, 'fix_type': np.nan
        }
        
        try:
            # GPS messages
            if msg_type in ['GPS', 'GPS2']:
                base_data.update({
                    'gps_lat': msg.Lat if hasattr(msg, 'Lat') else np.nan,
                    'gps_lon': msg.Lng if hasattr(msg, 'Lng') else np.nan,
                    'gps_alt': msg.Alt if hasattr(msg, 'Alt') else np.nan,
                    'velocity_x': msg.Spd if hasattr(msg, 'Spd') else np.nan,
                    'hdop': msg.HDop if hasattr(msg, 'HDop') else np.nan,
                    'vdop': msg.VDop if hasattr(msg, 'VDop') else np.nan,
                    'fix_type': msg.Status if hasattr(msg, 'Status') else np.nan
                })
                
            elif msg_type == 'GPA':
                base_data.update({
                    'velocity_x': msg.VN if hasattr(msg, 'VN') else np.nan,
                    'velocity_y': msg.VE if hasattr(msg, 'VE') else np.nan,
                    'velocity_z': msg.VD if hasattr(msg, 'VD') else np.nan
                })
                
            # IMU messages
            elif msg_type in ['IMU', 'IMU2', 'IMU3']:
                base_data.update({
                    'imu_ax': msg.AccX if hasattr(msg, 'AccX') else np.nan,
                    'imu_ay': msg.AccY if hasattr(msg, 'AccY') else np.nan,
                    'imu_az': msg.AccZ if hasattr(msg, 'AccZ') else np.nan,
                    'imu_gx': msg.GyrX if hasattr(msg, 'GyrX') else np.nan,
                    'imu_gy': msg.GyrY if hasattr(msg, 'GyrY') else np.nan,
                    'imu_gz': msg.GyrZ if hasattr(msg, 'GyrZ') else np.nan
                })
                
            elif msg_type in ['IMT', 'IMT2', 'IMT3']:
                # IMT messages have temperature data
                base_data.update({
                    'imu_ax': msg.AccX if hasattr(msg, 'AccX') else np.nan,
                    'imu_ay': msg.AccY if hasattr(msg, 'AccY') else np.nan,
                    'imu_az': msg.AccZ if hasattr(msg, 'AccZ') else np.nan,
                    'imu_gx': msg.GyrX if hasattr(msg, 'GyrX') else np.nan,
                    'imu_gy': msg.GyrY if hasattr(msg, 'GyrY') else np.nan,
                    'imu_gz': msg.GyrZ if hasattr(msg, 'GyrZ') else np.nan
                })
                
            # EKF messages - extract position and velocity estimates
            elif msg_type in ['XKF1', 'EKF1', 'NKF1']:
                # Position estimates
                base_data.update({
                    'velocity_x': msg.VN if hasattr(msg, 'VN') else np.nan,
                    'velocity_y': msg.VE if hasattr(msg, 'VE') else np.nan,
                    'velocity_z': msg.VD if hasattr(msg, 'VD') else np.nan
                })
                
            elif msg_type in ['XKF2', 'EKF2', 'NKF2']:
                # More EKF state data
                pass
                
            # Attitude messages
            elif msg_type == 'ATT':
                # Attitude data - could extract orientation info
                pass
                
            elif msg_type in ['AHR2', 'AHRS']:
                # AHRS data
                base_data.update({
                    'gps_lat': msg.Lat if hasattr(msg, 'Lat') else np.nan,
                    'gps_lon': msg.Lng if hasattr(msg, 'Lng') else np.nan,
                    'gps_alt': msg.Alt if hasattr(msg, 'Alt') else np.nan
                })
                
            # Handle MAVLink messages that might be embedded
            elif msg_type == 'GPS_RAW_INT':
                base_data.update({
                    'gps_lat': msg.lat / 1e7 if hasattr(msg, 'lat') else np.nan,
                    'gps_lon': msg.lon / 1e7 if hasattr(msg, 'lon') else np.nan,
                    'gps_alt': msg.alt / 1000.0 if hasattr(msg, 'alt') else np.nan,
                    'velocity_x': msg.vel / 100.0 if hasattr(msg, 'vel') else np.nan,
                    'hdop': msg.eph / 100.0 if hasattr(msg, 'eph') else np.nan,
                    'vdop': msg.epv / 100.0 if hasattr(msg, 'epv') else np.nan,
                    'fix_type': msg.fix_type if hasattr(msg, 'fix_type') else np.nan
                })
                
            elif msg_type == 'GLOBAL_POSITION_INT':
                base_data.update({
                    'gps_lat': msg.lat / 1e7 if hasattr(msg, 'lat') else np.nan,
                    'gps_lon': msg.lon / 1e7 if hasattr(msg, 'lon') else np.nan,
                    'gps_alt': msg.alt / 1000.0 if hasattr(msg, 'alt') else np.nan,
                    'velocity_x': msg.vx / 100.0 if hasattr(msg, 'vx') else np.nan,
                    'velocity_y': msg.vy / 100.0 if hasattr(msg, 'vy') else np.nan,
                    'velocity_z': msg.vz / 100.0 if hasattr(msg, 'vz') else np.nan
                })
                
            # Other message types (PID, RC, sensors, etc.) can be added as needed
            # For now, we focus on GPS and IMU data for the core functionality
            
            return base_data
            
        except Exception as e:
            self.logger.warning(f"Error parsing {msg_type} message: {str(e)}")
            return None
    
    def _messages_to_dataframe(self, messages: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert list of message dictionaries to DataFrame."""
        if not messages:
            return pd.DataFrame()
        
        df = pd.DataFrame(messages)
        
        # Group by timestamp and combine data from different message types
        # Use forward fill to propagate GPS data and backward fill for IMU data
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
    
    def _extract_timestamp(self, msg) -> float:
        """
        Extract timestamp from ArduPilot message.
        
        Args:
            msg: ArduPilot message object
            
        Returns:
            Timestamp in seconds since epoch
        """
        try:
            # ArduPilot logs typically use TimeUS (microseconds since boot)
            if hasattr(msg, 'TimeUS'):
                # Convert microseconds to seconds
                return float(msg.TimeUS) / 1e6
            elif hasattr(msg, '_timestamp'):
                return float(msg._timestamp)
            elif hasattr(msg, 'time_boot_ms'):
                # Convert milliseconds to seconds
                return float(msg.time_boot_ms) / 1000.0
            else:
                return np.nan
                
        except (AttributeError, ValueError, TypeError):
            return np.nan