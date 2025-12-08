"""
Ground truth position generator.

Generates corrected position estimates using sensor fusion techniques.
"""

from .base import BaseGenerator
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple, Any
import logging
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.spatial.distance import euclidean


class GroundTruthGenerator(BaseGenerator):
    """Generates ground truth positions using sensor fusion."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ground truth generator.
        
        Args:
            config: Configuration dictionary with generation parameters
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.rtk_fix_types = self.config.get('rtk_fix_types', [4, 5, 6])  # RTK fix types
        self.high_confidence_hdop_threshold = self.config.get('high_confidence_hdop_threshold', 1.0)
        self.anchor_validation_threshold = self.config.get('anchor_validation_threshold', 2.0)  # meters
        self.min_anchor_samples = self.config.get('min_anchor_samples', 10)
        self.max_velocity_threshold = self.config.get('max_velocity_threshold', 50.0)  # m/s
        self.integration_method = self.config.get('integration_method', 'trapezoidal')
        self.drift_correction_method = self.config.get('drift_correction_method', 'linear')
        self.smoothing_window = self.config.get('smoothing_window', 5)
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Process data by generating ground truth positions.
        
        Args:
            data: Input DataFrame with GPS, IMU, and motion classification data
            
        Returns:
            DataFrame with ground truth positions added
        """
        if not self.validate_input(data):
            raise ValueError("Invalid input data for ground truth generation")
        
        # Generate ground truth positions
        ground_truth_data = self.generate(data)
        
        # Merge with original data
        result = data.copy()
        for col in ground_truth_data.columns:
            if col not in ['timestamp']:  # Don't overwrite timestamp
                result[col] = ground_truth_data[col]
        
        return result
    
    def generate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate ground truth positions using sensor fusion.
        
        Args:
            data: Input DataFrame with GPS, IMU, and motion data
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with ground truth positions
        """
        self.logger.info("Starting ground truth generation")
        
        # Step 1: Calculate stationary anchor points
        anchor_points = self.calculate_stationary_anchors(data)
        self.logger.info(f"Found {len(anchor_points)} anchor points")
        
        # Step 2: Perform IMU velocity integration between anchors
        integrated_positions = self.integrate_imu_velocity(data, anchor_points)
        
        # Step 3: Apply sensor fusion and smoothing
        fused_positions = self.apply_sensor_fusion(data, integrated_positions, anchor_points)
        
        # Create result DataFrame
        result = pd.DataFrame(index=data.index)
        result['timestamp'] = data['timestamp']
        result['ground_truth_x'] = fused_positions['x']
        result['ground_truth_y'] = fused_positions['y'] 
        result['ground_truth_z'] = fused_positions['z']
        
        self.logger.info("Ground truth generation completed")
        return result
    
    def calculate_stationary_anchors(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate mean GPS position for stationary segments with validation.
        
        Args:
            data: DataFrame with GPS data and motion classification
            
        Returns:
            List of anchor point dictionaries
        """
        if 'motion_label' not in data.columns:
            raise ValueError("Motion classification required for anchor calculation")
        
        # Find GPS position columns
        gps_cols = self._find_gps_columns(data)
        if len(gps_cols) < 3:
            raise ValueError(f"Insufficient GPS columns found: {gps_cols}")
        
        x_col, y_col, z_col = gps_cols[:3]
        
        # Get stationary segments
        stationary_mask = data['motion_label'] == 'stationary'
        if not stationary_mask.any():
            self.logger.warning("No stationary segments found")
            return []
        
        # Find continuous stationary segments
        segments = self._find_continuous_segments(stationary_mask, data)
        
        anchor_points = []
        for segment in segments:
            start_idx, end_idx = segment['start_index'], segment['end_index']
            segment_data = data.iloc[start_idx:end_idx + 1]
            
            # Skip segments that are too short
            if len(segment_data) < self.min_anchor_samples:
                continue
            
            # Calculate anchor point
            anchor = self._calculate_segment_anchor(segment_data, x_col, y_col, z_col, segment)
            
            if anchor is not None:
                anchor_points.append(anchor)
        
        # Validate and filter anchor points
        validated_anchors = self._validate_anchor_points(anchor_points)
        
        self.logger.info(f"Generated {len(validated_anchors)} validated anchor points from {len(segments)} stationary segments")
        return validated_anchors
    
    def _find_gps_columns(self, data: pd.DataFrame) -> List[str]:
        """Find GPS position columns in the data."""
        # Look for ENU coordinates first (preferred)
        enu_cols = []
        for suffix in ['x', 'y', 'z']:
            candidates = [col for col in data.columns if col.endswith(f'gps_{suffix}') or col == f'gps_{suffix}']
            if candidates:
                enu_cols.append(candidates[0])
        
        if len(enu_cols) == 3:
            return enu_cols
        
        # Fallback to any GPS position columns
        gps_cols = [col for col in data.columns if 'gps' in col.lower() and any(coord in col.lower() for coord in ['x', 'y', 'z', 'lat', 'lon', 'alt'])]
        return gps_cols[:3] if len(gps_cols) >= 3 else gps_cols
    
    def _find_continuous_segments(self, mask: pd.Series, data: pd.DataFrame) -> List[Dict]:
        """Find continuous segments where mask is True."""
        segments = []
        start_idx = None
        
        for i, is_segment in enumerate(mask):
            if is_segment and start_idx is None:
                start_idx = i
            elif not is_segment and start_idx is not None:
                segments.append({
                    'start_index': start_idx,
                    'end_index': i - 1,
                    'start_timestamp': data.iloc[start_idx]['timestamp'],
                    'end_timestamp': data.iloc[i - 1]['timestamp'],
                    'duration': data.iloc[i - 1]['timestamp'] - data.iloc[start_idx]['timestamp']
                })
                start_idx = None
        
        # Handle case where data ends with a segment
        if start_idx is not None:
            segments.append({
                'start_index': start_idx,
                'end_index': len(mask) - 1,
                'start_timestamp': data.iloc[start_idx]['timestamp'],
                'end_timestamp': data.iloc[-1]['timestamp'],
                'duration': data.iloc[-1]['timestamp'] - data.iloc[start_idx]['timestamp']
            })
        
        return segments
    
    def _calculate_segment_anchor(self, segment_data: pd.DataFrame, x_col: str, y_col: str, z_col: str, segment_info: Dict) -> Optional[Dict]:
        """Calculate anchor point for a stationary segment."""
        # Check for RTK or high-confidence GPS data
        is_high_confidence = self._is_high_confidence_segment(segment_data)
        
        # Extract position data
        x_data = segment_data[x_col].dropna()
        y_data = segment_data[y_col].dropna()
        z_data = segment_data[z_col].dropna()
        
        if len(x_data) < self.min_anchor_samples or len(y_data) < self.min_anchor_samples or len(z_data) < self.min_anchor_samples:
            return None
        
        # Calculate statistics
        if is_high_confidence:
            # For high-confidence data, use simple mean
            anchor_x = x_data.mean()
            anchor_y = y_data.mean()
            anchor_z = z_data.mean()
            std_x = x_data.std()
            std_y = y_data.std()
            std_z = z_data.std()
        else:
            # For regular GPS, use robust statistics
            anchor_x = x_data.median()
            anchor_y = y_data.median()
            anchor_z = z_data.median()
            # Calculate median absolute deviation manually
            std_x = (x_data - x_data.median()).abs().median()
            std_y = (y_data - y_data.median()).abs().median()
            std_z = (z_data - z_data.median()).abs().median()
        
        # Calculate confidence metrics
        position_std = np.sqrt(std_x**2 + std_y**2 + std_z**2)
        
        anchor = {
            'timestamp': segment_info['start_timestamp'] + segment_info['duration'] / 2,  # Middle of segment
            'index': int((segment_info['start_index'] + segment_info['end_index']) / 2),
            'x': anchor_x,
            'y': anchor_y,
            'z': anchor_z,
            'std_x': std_x,
            'std_y': std_y,
            'std_z': std_z,
            'position_std': position_std,
            'confidence': 'high' if is_high_confidence else 'normal',
            'sample_count': len(x_data),
            'duration': segment_info['duration'],
            'segment_start': segment_info['start_index'],
            'segment_end': segment_info['end_index']
        }
        
        return anchor
    
    def _is_high_confidence_segment(self, segment_data: pd.DataFrame) -> bool:
        """Check if segment has high-confidence GPS data (RTK or low HDOP)."""
        # Check for RTK fix types
        if 'gps_fix_type' in segment_data.columns:
            fix_types = segment_data['gps_fix_type'].dropna()
            if not fix_types.empty and any(fix_type in self.rtk_fix_types for fix_type in fix_types):
                return True
        
        # Check for low HDOP values
        if 'gps_hdop' in segment_data.columns:
            hdop_values = segment_data['gps_hdop'].dropna()
            if not hdop_values.empty and hdop_values.median() <= self.high_confidence_hdop_threshold:
                return True
        
        return False
    
    def _validate_anchor_points(self, anchor_points: List[Dict]) -> List[Dict]:
        """Validate and filter anchor points based on consistency."""
        if len(anchor_points) <= 1:
            return anchor_points
        
        validated = []
        
        for i, anchor in enumerate(anchor_points):
            is_valid = True
            
            # Check position standard deviation
            if anchor['position_std'] > self.anchor_validation_threshold:
                self.logger.debug(f"Anchor {i} rejected: high position std ({anchor['position_std']:.2f}m)")
                is_valid = False
            
            # Check consistency with nearby anchors
            if is_valid and len(validated) > 0:
                # Find closest validated anchor
                distances = []
                for val_anchor in validated:
                    dist = euclidean(
                        [anchor['x'], anchor['y'], anchor['z']],
                        [val_anchor['x'], val_anchor['y'], val_anchor['z']]
                    )
                    distances.append(dist)
                
                min_distance = min(distances)
                # Allow larger distances for high-confidence anchors
                threshold = self.anchor_validation_threshold * (2 if anchor['confidence'] == 'high' else 1)
                
                if min_distance > threshold:
                    self.logger.debug(f"Anchor {i} rejected: inconsistent with nearby anchors (dist={min_distance:.2f}m)")
                    is_valid = False
            
            if is_valid:
                validated.append(anchor)
        
        return validated   
 
    def integrate_imu_velocity(self, data: pd.DataFrame, anchor_points: List[Dict]) -> pd.DataFrame:
        """
        Integrate IMU velocity between anchor points with drift correction.
        
        Args:
            data: DataFrame with IMU data and timestamps
            anchor_points: List of anchor point dictionaries
            
        Returns:
            DataFrame with integrated positions
        """
        if not anchor_points:
            self.logger.warning("No anchor points available for IMU integration")
            # Return GPS positions as fallback
            gps_cols = self._find_gps_columns(data)
            if len(gps_cols) >= 3:
                return pd.DataFrame({
                    'x': data[gps_cols[0]],
                    'y': data[gps_cols[1]], 
                    'z': data[gps_cols[2]]
                }, index=data.index)
            else:
                return pd.DataFrame({
                    'x': np.zeros(len(data)),
                    'y': np.zeros(len(data)),
                    'z': np.zeros(len(data))
                }, index=data.index)
        
        # Find IMU columns
        imu_cols = self._find_imu_columns(data)
        if len(imu_cols) < 3:
            raise ValueError(f"Insufficient IMU acceleration columns: {imu_cols}")
        
        ax_col, ay_col, az_col = imu_cols[:3]
        
        # Initialize position arrays
        positions = pd.DataFrame({
            'x': np.zeros(len(data)),
            'y': np.zeros(len(data)),
            'z': np.zeros(len(data))
        }, index=data.index)
        
        # Sort anchor points by timestamp
        sorted_anchors = sorted(anchor_points, key=lambda a: a['timestamp'])
        
        # Integrate between each pair of anchors
        for i in range(len(sorted_anchors)):
            if i == 0:
                # From start to first anchor
                start_idx = 0
                end_idx = sorted_anchors[i]['index']
                start_pos = [sorted_anchors[i]['x'], sorted_anchors[i]['y'], sorted_anchors[i]['z']]
                end_pos = start_pos
            else:
                # Between anchors
                start_idx = sorted_anchors[i-1]['index']
                end_idx = sorted_anchors[i]['index']
                start_pos = [sorted_anchors[i-1]['x'], sorted_anchors[i-1]['y'], sorted_anchors[i-1]['z']]
                end_pos = [sorted_anchors[i]['x'], sorted_anchors[i]['y'], sorted_anchors[i]['z']]
            
            # Integrate this segment
            segment_positions = self._integrate_segment(
                data.iloc[start_idx:end_idx+1],
                ax_col, ay_col, az_col,
                start_pos, end_pos
            )
            
            # Store results
            positions.iloc[start_idx:end_idx+1] = segment_positions
        
        # Handle segment after last anchor
        if len(sorted_anchors) > 0:
            last_anchor = sorted_anchors[-1]
            if last_anchor['index'] < len(data) - 1:
                start_idx = last_anchor['index']
                end_pos = [last_anchor['x'], last_anchor['y'], last_anchor['z']]
                
                segment_positions = self._integrate_segment(
                    data.iloc[start_idx:],
                    ax_col, ay_col, az_col,
                    end_pos, end_pos  # No drift correction for final segment
                )
                
                positions.iloc[start_idx:] = segment_positions
        
        return positions
    
    def _find_imu_columns(self, data: pd.DataFrame) -> List[str]:
        """Find IMU acceleration columns."""
        imu_cols = []
        for suffix in ['ax', 'ay', 'az']:
            candidates = [col for col in data.columns if col.endswith(f'imu_{suffix}') or col == f'imu_{suffix}']
            if candidates:
                imu_cols.append(candidates[0])
        
        return imu_cols
    
    def _integrate_segment(self, segment_data: pd.DataFrame, ax_col: str, ay_col: str, az_col: str, 
                          start_pos: List[float], end_pos: List[float]) -> pd.DataFrame:
        """
        Integrate IMU acceleration for a single segment with drift correction.
        
        Args:
            segment_data: DataFrame with IMU data for this segment
            ax_col, ay_col, az_col: Column names for acceleration data
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z] for drift correction
            
        Returns:
            DataFrame with integrated positions
        """
        if len(segment_data) < 2:
            # Not enough data for integration
            return pd.DataFrame({
                'x': [start_pos[0]] * len(segment_data),
                'y': [start_pos[1]] * len(segment_data),
                'z': [start_pos[2]] * len(segment_data)
            }, index=segment_data.index)
        
        # Extract acceleration data
        ax = segment_data[ax_col].fillna(0).values
        ay = segment_data[ay_col].fillna(0).values
        az = segment_data[az_col].fillna(0).values
        
        # Calculate time intervals
        timestamps = segment_data['timestamp'].values
        dt = np.diff(timestamps)
        dt = np.append(dt, dt[-1] if len(dt) > 0 else 0.1)  # Assume same interval for last sample
        
        # Remove gravity from vertical acceleration (approximate)
        az_corrected = az - 9.81
        
        # Apply velocity integration
        if self.integration_method == 'trapezoidal':
            vx = self._integrate_trapezoidal(ax, dt)
            vy = self._integrate_trapezoidal(ay, dt)
            vz = self._integrate_trapezoidal(az_corrected, dt)
            
            x = self._integrate_trapezoidal(vx, dt) + start_pos[0]
            y = self._integrate_trapezoidal(vy, dt) + start_pos[1]
            z = self._integrate_trapezoidal(vz, dt) + start_pos[2]
        else:
            # Simple cumulative integration
            vx = np.cumsum(ax * dt)
            vy = np.cumsum(ay * dt)
            vz = np.cumsum(az_corrected * dt)
            
            x = np.cumsum(vx * dt) + start_pos[0]
            y = np.cumsum(vy * dt) + start_pos[1]
            z = np.cumsum(vz * dt) + start_pos[2]
        
        # Apply drift correction if we have different start and end positions
        if not np.allclose(start_pos, end_pos, atol=0.1):
            x, y, z = self._apply_drift_correction(x, y, z, start_pos, end_pos, timestamps)
        
        # Limit velocities to reasonable values
        x, y, z = self._limit_velocities(x, y, z, dt)
        
        return pd.DataFrame({
            'x': x,
            'y': y,
            'z': z
        }, index=segment_data.index)
    
    def _integrate_trapezoidal(self, acceleration: np.ndarray, dt: np.ndarray) -> np.ndarray:
        """Integrate acceleration using trapezoidal rule."""
        if len(acceleration) < 2:
            return np.zeros_like(acceleration)
        
        # Trapezoidal integration: v[i] = v[i-1] + (a[i-1] + a[i]) * dt[i] / 2
        velocity = np.zeros_like(acceleration)
        for i in range(1, len(acceleration)):
            velocity[i] = velocity[i-1] + (acceleration[i-1] + acceleration[i]) * dt[i-1] / 2
        
        return velocity
    
    def _apply_drift_correction(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                               start_pos: List[float], end_pos: List[float], 
                               timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply drift correction to integrated positions.
        
        Args:
            x, y, z: Integrated position arrays
            start_pos: Expected starting position
            end_pos: Expected ending position
            timestamps: Timestamp array
            
        Returns:
            Drift-corrected position arrays
        """
        if len(x) < 2:
            return x, y, z
        
        # Calculate drift (difference between integrated and expected end position)
        drift_x = x[-1] - end_pos[0]
        drift_y = y[-1] - end_pos[1]
        drift_z = z[-1] - end_pos[2]
        
        if self.drift_correction_method == 'linear':
            # Linear drift correction
            time_normalized = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
            
            x_corrected = x - drift_x * time_normalized
            y_corrected = y - drift_y * time_normalized
            z_corrected = z - drift_z * time_normalized
        else:
            # Proportional drift correction (default)
            total_time = timestamps[-1] - timestamps[0]
            if total_time > 0:
                time_ratios = (timestamps - timestamps[0]) / total_time
                
                x_corrected = x - drift_x * time_ratios
                y_corrected = y - drift_y * time_ratios
                z_corrected = z - drift_z * time_ratios
            else:
                x_corrected, y_corrected, z_corrected = x, y, z
        
        return x_corrected, y_corrected, z_corrected
    
    def _limit_velocities(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                         dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Limit unrealistic velocities in integrated positions."""
        if len(x) < 2:
            return x, y, z
        
        # Calculate velocities
        # np.diff(x) has length n-1, dt should have same length
        dt_for_velocity = dt[:len(x)-1] if len(dt) >= len(x)-1 else dt
        vx = np.diff(x) / dt_for_velocity
        vy = np.diff(y) / dt_for_velocity
        vz = np.diff(z) / dt_for_velocity
        
        # Find points with excessive velocity
        velocity_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        excessive_velocity = velocity_magnitude > self.max_velocity_threshold
        
        if np.any(excessive_velocity):
            self.logger.debug(f"Limiting {np.sum(excessive_velocity)} points with excessive velocity")
            
            # Apply smoothing to reduce velocity spikes
            x_smooth = savgol_filter(x, min(self.smoothing_window, len(x)//2*2+1), 2)
            y_smooth = savgol_filter(y, min(self.smoothing_window, len(y)//2*2+1), 2)
            z_smooth = savgol_filter(z, min(self.smoothing_window, len(z)//2*2+1), 2)
            
            return x_smooth, y_smooth, z_smooth
        
        return x, y, z
    
    def apply_sensor_fusion(self, data: pd.DataFrame, integrated_positions: pd.DataFrame, 
                           anchor_points: List[Dict]) -> pd.DataFrame:
        """
        Apply sensor fusion algorithms to combine GPS, IMU, and anchor data.
        
        Args:
            data: Original data with GPS and IMU
            integrated_positions: IMU-integrated positions
            anchor_points: Stationary anchor points
            
        Returns:
            DataFrame with fused positions
        """
        fusion_method = self.config.get('fusion_method', 'complementary')
        
        if fusion_method == 'ekf':
            return self._apply_extended_kalman_filter(data, integrated_positions, anchor_points)
        else:
            return self._apply_complementary_filter(data, integrated_positions, anchor_points)
    
    def _apply_extended_kalman_filter(self, data: pd.DataFrame, integrated_positions: pd.DataFrame,
                                    anchor_points: List[Dict]) -> pd.DataFrame:
        """
        Apply Extended Kalman Filter for position estimation.
        
        Args:
            data: Original sensor data
            integrated_positions: IMU-integrated positions
            anchor_points: Anchor points for reference
            
        Returns:
            DataFrame with EKF-filtered positions
        """
        # EKF state: [x, y, z, vx, vy, vz]
        state_dim = 6
        measurement_dim = 3
        
        # Initialize state and covariance
        state = np.zeros(state_dim)
        P = np.eye(state_dim) * 10.0  # Initial covariance
        
        # Process noise covariance
        Q = np.eye(state_dim)
        Q[:3, :3] *= 0.1  # Position process noise
        Q[3:, 3:] *= 1.0  # Velocity process noise
        
        # Measurement noise covariance
        R = np.eye(measurement_dim) * 2.0  # GPS measurement noise
        
        # Find GPS columns
        gps_cols = self._find_gps_columns(data)
        if len(gps_cols) < 3:
            self.logger.warning("Insufficient GPS columns for EKF, falling back to complementary filter")
            return self._apply_complementary_filter(data, integrated_positions, anchor_points)
        
        # Initialize results
        filtered_positions = pd.DataFrame({
            'x': np.zeros(len(data)),
            'y': np.zeros(len(data)),
            'z': np.zeros(len(data))
        }, index=data.index)
        
        # Initialize state with first GPS measurement
        if not data[gps_cols[0]].isna().iloc[0]:
            state[:3] = [data[gps_cols[0]].iloc[0], data[gps_cols[1]].iloc[0], data[gps_cols[2]].iloc[0]]
        
        timestamps = data['timestamp'].values
        
        for i in range(len(data)):
            dt = timestamps[i] - timestamps[i-1] if i > 0 else 0.1
            
            # Prediction step
            F = self._get_state_transition_matrix(dt)
            state = F @ state
            P = F @ P @ F.T + Q
            
            # Update step with GPS measurement
            if not any(data[col].isna().iloc[i] for col in gps_cols):
                # GPS measurement available
                z = np.array([data[gps_cols[0]].iloc[i], data[gps_cols[1]].iloc[i], data[gps_cols[2]].iloc[i]])
                H = np.zeros((measurement_dim, state_dim))
                H[:3, :3] = np.eye(3)  # Measure position directly
                
                # Adjust measurement noise based on GPS quality
                R_current = R.copy()
                if 'gps_hdop' in data.columns and not data['gps_hdop'].isna().iloc[i]:
                    hdop = data['gps_hdop'].iloc[i]
                    R_current *= max(1.0, hdop)  # Increase noise for poor GPS
                
                # Kalman update
                y = z - H @ state  # Innovation
                S = H @ P @ H.T + R_current  # Innovation covariance
                K = P @ H.T @ np.linalg.inv(S)  # Kalman gain
                
                state = state + K @ y
                P = (np.eye(state_dim) - K @ H) @ P
            
            # Store filtered position
            filtered_positions.iloc[i] = state[:3]
        
        # Apply cubic spline smoothing for transitions
        return self._apply_cubic_spline_smoothing(filtered_positions, anchor_points)
    
    def _apply_complementary_filter(self, data: pd.DataFrame, integrated_positions: pd.DataFrame,
                                  anchor_points: List[Dict]) -> pd.DataFrame:
        """
        Apply complementary filter as alternative fusion method.
        
        Args:
            data: Original sensor data
            integrated_positions: IMU-integrated positions
            anchor_points: Anchor points for reference
            
        Returns:
            DataFrame with complementary-filtered positions
        """
        # Find GPS columns
        gps_cols = self._find_gps_columns(data)
        if len(gps_cols) < 3:
            self.logger.warning("Insufficient GPS columns, using integrated positions only")
            return integrated_positions
        
        # Complementary filter parameters
        alpha = self.config.get('complementary_alpha', 0.98)  # Trust IMU more for short term
        
        # Initialize with GPS or first anchor
        fused_positions = pd.DataFrame({
            'x': np.zeros(len(data)),
            'y': np.zeros(len(data)),
            'z': np.zeros(len(data))
        }, index=data.index)
        
        # Start with GPS position
        fused_positions.iloc[0] = [
            data[gps_cols[0]].iloc[0] if not data[gps_cols[0]].isna().iloc[0] else 0,
            data[gps_cols[1]].iloc[0] if not data[gps_cols[1]].isna().iloc[0] else 0,
            data[gps_cols[2]].iloc[0] if not data[gps_cols[2]].isna().iloc[0] else 0
        ]
        
        for i in range(1, len(data)):
            # Get IMU-integrated position change
            imu_dx = integrated_positions.iloc[i]['x'] - integrated_positions.iloc[i-1]['x']
            imu_dy = integrated_positions.iloc[i]['y'] - integrated_positions.iloc[i-1]['y']
            imu_dz = integrated_positions.iloc[i]['z'] - integrated_positions.iloc[i-1]['z']
            
            # Predict position using IMU
            predicted_x = fused_positions.iloc[i-1]['x'] + imu_dx
            predicted_y = fused_positions.iloc[i-1]['y'] + imu_dy
            predicted_z = fused_positions.iloc[i-1]['z'] + imu_dz
            
            # Check if GPS measurement is available
            if not any(data[col].isna().iloc[i] for col in gps_cols):
                gps_x = data[gps_cols[0]].iloc[i]
                gps_y = data[gps_cols[1]].iloc[i]
                gps_z = data[gps_cols[2]].iloc[i]
                
                # Adjust alpha based on GPS quality
                current_alpha = alpha
                if 'gps_hdop' in data.columns and not data['gps_hdop'].isna().iloc[i]:
                    hdop = data['gps_hdop'].iloc[i]
                    # Reduce trust in GPS for high HDOP
                    current_alpha = min(0.99, alpha + (hdop - 1.0) * 0.1)
                
                # Complementary filter: fused = alpha * predicted + (1-alpha) * gps
                fused_positions.iloc[i] = [
                    current_alpha * predicted_x + (1 - current_alpha) * gps_x,
                    current_alpha * predicted_y + (1 - current_alpha) * gps_y,
                    current_alpha * predicted_z + (1 - current_alpha) * gps_z
                ]
            else:
                # No GPS, use IMU prediction
                fused_positions.iloc[i] = [predicted_x, predicted_y, predicted_z]
        
        # Apply anchor point corrections
        fused_positions = self._apply_anchor_corrections(fused_positions, anchor_points, data)
        
        # Apply cubic spline smoothing for transitions
        return self._apply_cubic_spline_smoothing(fused_positions, anchor_points)
    
    def _get_state_transition_matrix(self, dt: float) -> np.ndarray:
        """Get state transition matrix for EKF."""
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt  # Position = position + velocity * dt
        return F
    
    def _apply_anchor_corrections(self, positions: pd.DataFrame, anchor_points: List[Dict], 
                                 data: pd.DataFrame) -> pd.DataFrame:
        """Apply corrections based on anchor points."""
        if not anchor_points:
            return positions
        
        corrected_positions = positions.copy()
        
        for anchor in anchor_points:
            # Find data points near this anchor
            anchor_idx = anchor['index']
            window_size = 10  # Points around anchor to correct
            
            start_idx = max(0, anchor_idx - window_size)
            end_idx = min(len(positions), anchor_idx + window_size + 1)
            
            # Calculate correction needed
            current_pos = positions.iloc[anchor_idx]
            target_pos = [anchor['x'], anchor['y'], anchor['z']]
            
            correction = [
                target_pos[0] - current_pos['x'],
                target_pos[1] - current_pos['y'],
                target_pos[2] - current_pos['z']
            ]
            
            # Apply correction with tapering
            for i in range(start_idx, end_idx):
                distance_from_anchor = abs(i - anchor_idx)
                weight = max(0, 1 - distance_from_anchor / window_size)
                
                corrected_positions.iloc[i] = [
                    positions.iloc[i]['x'] + correction[0] * weight,
                    positions.iloc[i]['y'] + correction[1] * weight,
                    positions.iloc[i]['z'] + correction[2] * weight
                ]
        
        return corrected_positions
    
    def _apply_cubic_spline_smoothing(self, positions: pd.DataFrame, anchor_points: List[Dict]) -> pd.DataFrame:
        """
        Apply cubic spline smoothing for transitions between segments.
        
        Args:
            positions: Position data to smooth
            anchor_points: Anchor points for reference
            
        Returns:
            Smoothed position data
        """
        if len(positions) < 4:  # Need at least 4 points for cubic spline
            return positions
        
        smoothed_positions = positions.copy()
        
        # Apply smoothing around anchor points to reduce discontinuities
        for anchor in anchor_points:
            anchor_idx = anchor['index']
            window_size = self.config.get('smoothing_window_size', 20)
            
            start_idx = max(0, anchor_idx - window_size)
            end_idx = min(len(positions), anchor_idx + window_size + 1)
            
            if end_idx - start_idx < 4:
                continue
            
            # Extract segment for smoothing
            segment_indices = np.arange(start_idx, end_idx)
            segment_positions = positions.iloc[start_idx:end_idx]
            
            try:
                # Create cubic splines for each coordinate
                spline_x = interpolate.CubicSpline(segment_indices, segment_positions['x'])
                spline_y = interpolate.CubicSpline(segment_indices, segment_positions['y'])
                spline_z = interpolate.CubicSpline(segment_indices, segment_positions['z'])
                
                # Apply smoothing with reduced weight near anchor
                for i, idx in enumerate(segment_indices):
                    distance_from_anchor = abs(idx - anchor_idx)
                    # Reduce smoothing weight near anchor to preserve anchor accuracy
                    smooth_weight = min(0.5, distance_from_anchor / (window_size / 2))
                    
                    original = positions.iloc[idx]
                    smoothed = [spline_x(idx), spline_y(idx), spline_z(idx)]
                    
                    smoothed_positions.iloc[idx] = [
                        original['x'] * (1 - smooth_weight) + smoothed[0] * smooth_weight,
                        original['y'] * (1 - smooth_weight) + smoothed[1] * smooth_weight,
                        original['z'] * (1 - smooth_weight) + smoothed[2] * smooth_weight
                    ]
                    
            except Exception as e:
                self.logger.debug(f"Spline smoothing failed for anchor {anchor_idx}: {e}")
                continue
        
        return smoothed_positions
    
    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data for ground truth generation.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if not super().validate_input(data):
            return False
        
        # Check for required columns
        required_cols = ['timestamp']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for GPS columns
        gps_cols = self._find_gps_columns(data)
        if len(gps_cols) < 3:
            self.logger.error(f"Insufficient GPS columns found: {gps_cols}")
            return False
        
        # Check for IMU columns
        imu_cols = self._find_imu_columns(data)
        if len(imu_cols) < 3:
            self.logger.error(f"Insufficient IMU columns found: {imu_cols}")
            return False
        
        return True