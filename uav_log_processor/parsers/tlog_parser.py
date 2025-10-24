"""
TLOG parser for MAVLink telemetry logs.

Parses TLOG files using pymavlink to extract GPS, IMU, and other sensor data.
"""

from .base import BaseLogParser
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    from pymavlink import mavutil
    from pymavlink.dialects.v20 import common as mavlink
except ImportError:
    raise ImportError("pymavlink is required for TLOG parsing. Install with: pip install pymavlink")


class TLogParser(BaseLogParser):
    """Parser for TLOG (MAVLink telemetry) files."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self._supported_extensions = {'.tlog'}
        self.logger = logging.getLogger(__name__)
        
        # Message types we want to extract
        self.target_messages = {
            'GPS_RAW_INT', 'GLOBAL_POSITION_INT', 'IMU_RAW', 'ATTITUDE',
            'EKF_STATUS_REPORT', 'AHRS', 'AHRS2', 'NAV_CONTROLLER_OUTPUT', 
            'VFR_HUD', 'RAW_IMU', 'SCALED_IMU2'
        }
    
    def parse(self, file_path: str) -> pd.DataFrame:
        """
        Parse TLOG file and extract sensor data.
        
        Args:
            file_path: Path to the TLOG file
            
        Returns:
            DataFrame with standardized columns
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        if not self.validate_file(file_path):
            raise FileNotFoundError(f"Invalid or missing TLOG file: {file_path}")
        
        self.logger.info(f"Parsing TLOG file: {file_path}")
        
        try:
            # Open MAVLink connection
            mav_conn = mavutil.mavlink_connection(file_path)
            
            # Extract messages
            messages = self._extract_messages(mav_conn)
            
            if not messages:
                raise ValueError("No valid MAVLink messages found in file")
            
            # Convert to DataFrame
            df = self._messages_to_dataframe(messages)
            
            # Standardize columns
            df = self._standardize_columns(df)
            
            self.logger.info(f"Successfully parsed {len(df)} records from TLOG file")
            return df
            
        except Exception as e:
            self.logger.error(f"Error parsing TLOG file {file_path}: {str(e)}")
            raise ValueError(f"Failed to parse TLOG file: {str(e)}")
    
    def _extract_messages(self, mav_conn) -> List[Dict[str, Any]]:
        """Extract relevant MAVLink messages from connection."""
        messages = []
        
        while True:
            try:
                msg = mav_conn.recv_match(blocking=False)
                if msg is None:
                    break
                
                # Only process messages we're interested in
                if msg.get_type() in self.target_messages:
                    msg_data = self._parse_message(msg)
                    if msg_data:
                        messages.append(msg_data)
                        
            except Exception as e:
                self.logger.warning(f"Error reading message: {str(e)}")
                continue
        
        return messages
    
    def _parse_message(self, msg) -> Optional[Dict[str, Any]]:
        """Parse individual MAVLink message into standardized format."""
        msg_type = msg.get_type()
        timestamp = self._extract_timestamp(msg._timestamp)
        
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
            if msg_type == 'GPS_RAW_INT':
                base_data.update({
                    'gps_lat': msg.lat / 1e7,  # Convert from 1e7 degrees to degrees
                    'gps_lon': msg.lon / 1e7,
                    'gps_alt': msg.alt / 1000.0,  # Convert from mm to m
                    'velocity_x': msg.vel / 100.0 if hasattr(msg, 'vel') else np.nan,
                    'hdop': msg.eph / 100.0 if hasattr(msg, 'eph') else np.nan,
                    'vdop': msg.epv / 100.0 if hasattr(msg, 'epv') else np.nan,
                    'fix_type': msg.fix_type if hasattr(msg, 'fix_type') else np.nan
                })
                
            elif msg_type == 'GLOBAL_POSITION_INT':
                base_data.update({
                    'gps_lat': msg.lat / 1e7,
                    'gps_lon': msg.lon / 1e7,
                    'gps_alt': msg.alt / 1000.0,
                    'velocity_x': msg.vx / 100.0 if hasattr(msg, 'vx') else np.nan,
                    'velocity_y': msg.vy / 100.0 if hasattr(msg, 'vy') else np.nan,
                    'velocity_z': msg.vz / 100.0 if hasattr(msg, 'vz') else np.nan
                })
                
            elif msg_type in ['IMU_RAW', 'RAW_IMU']:
                # Convert from milli-g and milli-rad/s to m/s² and rad/s
                base_data.update({
                    'imu_ax': msg.xacc / 1000.0 * 9.81 if hasattr(msg, 'xacc') else np.nan,
                    'imu_ay': msg.yacc / 1000.0 * 9.81 if hasattr(msg, 'yacc') else np.nan,
                    'imu_az': msg.zacc / 1000.0 * 9.81 if hasattr(msg, 'zacc') else np.nan,
                    'imu_gx': msg.xgyro / 1000.0 if hasattr(msg, 'xgyro') else np.nan,
                    'imu_gy': msg.ygyro / 1000.0 if hasattr(msg, 'ygyro') else np.nan,
                    'imu_gz': msg.zgyro / 1000.0 if hasattr(msg, 'zgyro') else np.nan
                })
                
            elif msg_type == 'SCALED_IMU2':
                # Already in correct units (m/s² and rad/s)
                base_data.update({
                    'imu_ax': msg.xacc / 1000.0 * 9.81 if hasattr(msg, 'xacc') else np.nan,
                    'imu_ay': msg.yacc / 1000.0 * 9.81 if hasattr(msg, 'yacc') else np.nan,
                    'imu_az': msg.zacc / 1000.0 * 9.81 if hasattr(msg, 'zacc') else np.nan,
                    'imu_gx': msg.xgyro / 1000.0 if hasattr(msg, 'xgyro') else np.nan,
                    'imu_gy': msg.ygyro / 1000.0 if hasattr(msg, 'ygyro') else np.nan,
                    'imu_gz': msg.zgyro / 1000.0 if hasattr(msg, 'zgyro') else np.nan
                })
                
            elif msg_type == 'ATTITUDE':
                # Extract attitude data - could be used for orientation
                # For now, we don't have specific columns for attitude
                pass
                
            elif msg_type == 'VFR_HUD':
                # Extract velocity information
                if hasattr(msg, 'groundspeed'):
                    base_data['velocity_x'] = msg.groundspeed  # m/s
                    
            # Add other message types as needed (EKF_STATUS_REPORT, AHRS, etc.)
            # These would require additional columns in the base schema
            
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
        # This handles cases where GPS and IMU data come in separate messages
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
    
    def _extract_timestamp(self, raw_timestamp: Any) -> float:
        """
        Convert MAVLink timestamp to standardized format.
        
        Args:
            raw_timestamp: Raw timestamp from MAVLink message
            
        Returns:
            Timestamp in seconds since epoch
        """
        if raw_timestamp is None:
            return np.nan
            
        # MAVLink timestamps are typically in seconds since epoch
        if isinstance(raw_timestamp, (int, float)):
            return float(raw_timestamp)
        
        # Handle other timestamp formats if needed
        return super()._extract_timestamp(raw_timestamp)