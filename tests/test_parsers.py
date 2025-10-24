"""
Unit tests for UAV log parsers.

Tests all parser implementations with sample data and edge cases.
"""

import unittest
import tempfile
import os
import pandas as pd
import numpy as np
from pathlib import Path
import struct

from uav_log_processor.parsers.tlog_parser import TLogParser
from uav_log_processor.parsers.bin_parser import BinParser
from uav_log_processor.parsers.rlog_parser import RLogParser
from uav_log_processor.parsers.txt_parser import TxtParser


class TestTLogParser(unittest.TestCase):
    """Test TLOG parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = TLogParser()
        
    def test_supported_extensions(self):
        """Test that parser supports correct extensions."""
        self.assertIn('.tlog', self.parser.supported_extensions)
        
    def test_validate_file_nonexistent(self):
        """Test validation of non-existent file."""
        self.assertFalse(self.parser.validate_file('nonexistent.tlog'))
        
    def test_validate_file_wrong_extension(self):
        """Test validation of file with wrong extension."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b'test data')
            temp_path = f.name
        
        try:
            self.assertFalse(self.parser.validate_file(temp_path))
        finally:
            os.unlink(temp_path)
    
    def test_parse_empty_file(self):
        """Test parsing empty TLOG file."""
        with tempfile.NamedTemporaryFile(suffix='.tlog', delete=False) as f:
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.parser.parse(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_parse_invalid_file(self):
        """Test parsing invalid TLOG file."""
        with tempfile.NamedTemporaryFile(suffix='.tlog', delete=False) as f:
            f.write(b'invalid tlog data')
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.parser.parse(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_extract_timestamp(self):
        """Test timestamp extraction."""
        # Test valid timestamp
        timestamp = self.parser._extract_timestamp(1234567890.5)
        self.assertEqual(timestamp, 1234567890.5)
        
        # Test None timestamp
        timestamp = self.parser._extract_timestamp(None)
        self.assertTrue(np.isnan(timestamp))


class TestBinParser(unittest.TestCase):
    """Test BIN parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = BinParser()
        
    def test_supported_extensions(self):
        """Test that parser supports correct extensions."""
        self.assertIn('.bin', self.parser.supported_extensions)
        
    def test_validate_file_nonexistent(self):
        """Test validation of non-existent file."""
        self.assertFalse(self.parser.validate_file('nonexistent.bin'))
    
    def test_parse_empty_file(self):
        """Test parsing empty BIN file."""
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.parser.parse(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_extract_timestamp_with_timeus(self):
        """Test timestamp extraction from TimeUS field."""
        # Mock message with TimeUS
        class MockMsg:
            def __init__(self):
                self.TimeUS = 1234567890
        
        msg = MockMsg()
        timestamp = self.parser._extract_timestamp(msg)
        self.assertAlmostEqual(timestamp, 1234.56789, places=5)
    
    def test_extract_timestamp_invalid(self):
        """Test timestamp extraction with invalid data."""
        class MockMsg:
            pass
        
        msg = MockMsg()
        timestamp = self.parser._extract_timestamp(msg)
        self.assertTrue(np.isnan(timestamp))


class TestRLogParser(unittest.TestCase):
    """Test RLOG parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = RLogParser()
        
    def test_supported_extensions(self):
        """Test that parser supports correct extensions."""
        self.assertIn('.rlog', self.parser.supported_extensions)
    
    def test_extract_metadata(self):
        """Test metadata extraction from RLOG content."""
        content = b"""
        ArduCopter V4.5.1 (71a2f169)
        ChibiOS: 6a85082c
        Pixhawk6C 003D0048 3133510F 31343630
        Frame: QUAD/X
        """
        
        metadata = self.parser._extract_metadata(content)
        
        self.assertEqual(metadata.get('firmware_version'), '4.5.1')
        self.assertEqual(metadata.get('chibios_version'), '6a85082c')
        self.assertEqual(metadata.get('hardware'), 'Pixhawk6C')
        self.assertEqual(metadata.get('frame_type'), 'QUAD/X')
    
    def test_extract_parameters(self):
        """Test parameter extraction from RLOG content."""
        content = b"""
        COMPASS_OFS_X 123.45
        INS_GYROFFS_Y -0.0123
        ATC_ANG_RLL_P 4.5
        WPNAV_SPEED 500.0
        UNRELATED_PARAM 999.0
        """
        
        parameters = self.parser._extract_parameters(content)
        
        # Should extract sensor-related parameters
        self.assertIn('COMPASS_OFS_X', parameters)
        self.assertIn('INS_GYROFFS_Y', parameters)
        self.assertIn('ATC_ANG_RLL_P', parameters)
        self.assertIn('WPNAV_SPEED', parameters)
        
        # Should not extract unrelated parameters
        self.assertNotIn('UNRELATED_PARAM', parameters)
        
        # Check values
        self.assertAlmostEqual(parameters['COMPASS_OFS_X'], 123.45)
        self.assertAlmostEqual(parameters['INS_GYROFFS_Y'], -0.0123)
    
    def test_is_sensor_parameter(self):
        """Test sensor parameter identification."""
        # Should identify sensor parameters
        self.assertTrue(self.parser._is_sensor_parameter('COMPASS_OFS_X'))
        self.assertTrue(self.parser._is_sensor_parameter('INS_ACCOFFS_Y'))
        self.assertTrue(self.parser._is_sensor_parameter('ATC_ANG_PIT_P'))
        self.assertTrue(self.parser._is_sensor_parameter('WPNAV_ACCEL'))
        
        # Should not identify non-sensor parameters
        self.assertFalse(self.parser._is_sensor_parameter('SERIAL0_BAUD'))
        self.assertFalse(self.parser._is_sensor_parameter('LOG_BITMASK'))
    
    def test_parse_empty_file(self):
        """Test parsing empty RLOG file."""
        with tempfile.NamedTemporaryFile(suffix='.rlog', delete=False) as f:
            temp_path = f.name
        
        try:
            df = self.parser.parse(temp_path)
            self.assertIsInstance(df, pd.DataFrame)
            # Should have at least one row with NaN values
            self.assertGreaterEqual(len(df), 1)
        finally:
            os.unlink(temp_path)


class TestTxtParser(unittest.TestCase):
    """Test TXT parser functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = TxtParser()
        
    def test_supported_extensions(self):
        """Test that parser supports correct extensions."""
        self.assertIn('.txt', self.parser.supported_extensions)
        self.assertIn('.log', self.parser.supported_extensions)
    
    def test_detect_format_ardupilot(self):
        """Test ArduPilot format detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write('FMT, 128, 89, FMT, BBnNZ, Type,Length,Name,Format,Columns\n')
            f.write('GPS, 123456789, 1, 3, 1234, 5, 1.5, -122.123456, 37.123456, 100.5, 2.3, 45.0, -0.1, 90.0, 1\n')
            temp_path = f.name
        
        try:
            format_type = self.parser._detect_format(temp_path)
            self.assertEqual(format_type, 'ardupilot')
        finally:
            os.unlink(temp_path)
    
    def test_detect_format_csv(self):
        """Test CSV format detection."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('timestamp,lat,lon,alt,acc_x,acc_y,acc_z\n')
            f.write('1234567890,37.123456,-122.123456,100.5,0.1,0.2,9.8\n')
            temp_path = f.name
        
        try:
            format_type = self.parser._detect_format(temp_path)
            self.assertEqual(format_type, 'csv')
        finally:
            os.unlink(temp_path)
    
    def test_parse_fmt_message(self):
        """Test FMT message parsing."""
        fmt_line = 'FMT, 94, 51, GPS, QBBIHBcLLeffffB, TimeUS,I,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd,GCrs,VZ,Yaw,U'
        
        self.parser._parse_fmt_message(fmt_line)
        
        self.assertIn('GPS', self.parser.format_definitions)
        fmt_def = self.parser.format_definitions['GPS']
        self.assertEqual(fmt_def['format'], 'QBBIHBcLLeffffB')
        self.assertIn('TimeUS', fmt_def['columns'])
        self.assertIn('Lat', fmt_def['columns'])
        self.assertIn('Lng', fmt_def['columns'])
    
    def test_parse_ardupilot_gps_message(self):
        """Test ArduPilot GPS message parsing."""
        # Set up format definition
        self.parser.format_definitions['GPS'] = {
            'format': 'QBBIHBcLLeffffB',
            'columns': ['TimeUS', 'I', 'Status', 'GMS', 'GWk', 'NSats', 'HDop', 'Lat', 'Lng', 'Alt', 'Spd', 'GCrs', 'VZ', 'Yaw', 'U']
        }
        
        # Use realistic microsecond timestamp (123456789 microseconds = 123.456789 seconds)
        gps_line = 'GPS, 123456789, 1, 3, 1234, 5, 1.5, -122.123456, 37.123456, 100.5, 2.3, 45.0, -0.1, 90.0, 1'
        
        msg_data = self.parser._parse_ardupilot_message(gps_line)
        
        self.assertIsNotNone(msg_data)
        self.assertEqual(msg_data['message_type'], 'GPS')
        self.assertAlmostEqual(msg_data['timestamp'], 123.456789)  # Converted from microseconds
        self.assertAlmostEqual(msg_data['gps_lat'], 37.123456)
        self.assertAlmostEqual(msg_data['gps_lon'], -122.123456)
        self.assertAlmostEqual(msg_data['gps_alt'], 100.5)
        self.assertAlmostEqual(msg_data['velocity_x'], 2.3)
        self.assertAlmostEqual(msg_data['hdop'], 1.5)
        self.assertEqual(msg_data['fix_type'], 3.0)
    
    def test_parse_csv_log(self):
        """Test CSV log parsing."""
        csv_content = """timestamp,lat,lon,alt,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
1234567890.5,37.123456,-122.123456,100.5,0.1,0.2,9.8,0.01,0.02,0.03
1234567891.0,37.123457,-122.123457,100.6,0.11,0.21,9.81,0.011,0.021,0.031
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            df = self.parser._parse_csv_log(temp_path)
            
            self.assertIsInstance(df, pd.DataFrame)
            self.assertEqual(len(df), 2)
            
            # Check column mapping
            self.assertIn('timestamp', df.columns)
            self.assertIn('gps_lat', df.columns)
            self.assertIn('gps_lon', df.columns)
            self.assertIn('gps_alt', df.columns)
            self.assertIn('imu_ax', df.columns)
            self.assertIn('imu_gx', df.columns)
            
            # Check values
            self.assertAlmostEqual(df.iloc[0]['gps_lat'], 37.123456)
            self.assertAlmostEqual(df.iloc[0]['gps_lon'], -122.123456)
            self.assertAlmostEqual(df.iloc[0]['imu_ax'], 0.1)
            
        finally:
            os.unlink(temp_path)
    
    def test_parse_empty_file(self):
        """Test parsing empty text file."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            with self.assertRaises(ValueError):
                self.parser.parse(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_messages_to_dataframe(self):
        """Test message list to DataFrame conversion."""
        messages = [
            {
                'timestamp': 1.0,
                'gps_lat': 37.123,
                'gps_lon': -122.123,
                'gps_alt': 100.0,
                'imu_ax': np.nan,
                'imu_ay': np.nan,
                'imu_az': np.nan,
                'imu_gx': np.nan,
                'imu_gy': np.nan,
                'imu_gz': np.nan,
                'velocity_x': 2.0,
                'velocity_y': np.nan,
                'velocity_z': np.nan,
                'hdop': 1.5,
                'vdop': np.nan,
                'fix_type': 3.0
            },
            {
                'timestamp': 1.0,  # Same timestamp
                'gps_lat': np.nan,
                'gps_lon': np.nan,
                'gps_alt': np.nan,
                'imu_ax': 0.1,
                'imu_ay': 0.2,
                'imu_az': 9.8,
                'imu_gx': 0.01,
                'imu_gy': 0.02,
                'imu_gz': 0.03,
                'velocity_x': np.nan,
                'velocity_y': np.nan,
                'velocity_z': np.nan,
                'hdop': np.nan,
                'vdop': np.nan,
                'fix_type': np.nan
            }
        ]
        
        df = self.parser._messages_to_dataframe(messages)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 1)  # Should combine messages with same timestamp
        
        # Check that GPS data from first message is preserved
        self.assertAlmostEqual(df.iloc[0]['gps_lat'], 37.123)
        self.assertAlmostEqual(df.iloc[0]['gps_lon'], -122.123)
        
        # Check that IMU data from second message is preserved
        self.assertAlmostEqual(df.iloc[0]['imu_ax'], 0.1)
        self.assertAlmostEqual(df.iloc[0]['imu_ay'], 0.2)


class TestParserIntegration(unittest.TestCase):
    """Integration tests for all parsers."""
    
    def test_all_parsers_standardize_columns(self):
        """Test that all parsers produce standardized column format."""
        expected_columns = [
            'timestamp', 'gps_lat', 'gps_lon', 'gps_alt',
            'imu_ax', 'imu_ay', 'imu_az', 'imu_gx', 'imu_gy', 'imu_gz',
            'velocity_x', 'velocity_y', 'velocity_z', 'hdop', 'vdop', 'fix_type'
        ]
        
        # Create minimal test data for each parser
        test_data = {
            'timestamp': [1.0, 2.0],
            'gps_lat': [37.123, 37.124],
            'gps_lon': [-122.123, -122.124],
            'gps_alt': [100.0, 101.0],
            'imu_ax': [0.1, 0.11],
            'imu_ay': [0.2, 0.21],
            'imu_az': [9.8, 9.81],
            'imu_gx': [0.01, 0.011],
            'imu_gy': [0.02, 0.021],
            'imu_gz': [0.03, 0.031],
            'velocity_x': [2.0, 2.1],
            'velocity_y': [1.0, 1.1],
            'velocity_z': [0.0, 0.1],
            'hdop': [1.5, 1.6],
            'vdop': [2.0, 2.1],
            'fix_type': [3.0, 3.0]
        }
        
        df = pd.DataFrame(test_data)
        
        # Test each parser's standardization
        parsers = [TLogParser(), BinParser(), RLogParser(), TxtParser()]
        
        for parser in parsers:
            standardized_df = parser._standardize_columns(df.copy())
            
            # Check that all expected columns are present
            for col in expected_columns:
                self.assertIn(col, standardized_df.columns, 
                             f"Column {col} missing in {parser.__class__.__name__}")
            
            # Check that columns are in correct order
            self.assertEqual(list(standardized_df.columns), expected_columns,
                           f"Column order incorrect in {parser.__class__.__name__}")
            
            # Check that timestamp is sorted
            self.assertTrue(standardized_df['timestamp'].is_monotonic_increasing,
                          f"Timestamp not sorted in {parser.__class__.__name__}")


if __name__ == '__main__':
    unittest.main()