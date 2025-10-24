"""
Visualization utilities for trajectory plotting.

Creates comprehensive visualizations for UAV trajectories including:
- Raw GPS vs corrected trajectory plots
- Error distribution visualizations  
- Flight path and motion segment visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class TrajectoryVisualizer:
    """Creates comprehensive visualizations for UAV trajectories and analysis."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize trajectory visualizer.
        
        Args:
            config: Configuration dictionary with visualization parameters
        """
        self.config = config or {}
        
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Default figure parameters
        self.default_figsize = (15, 10)
        self.default_dpi = 300
        
    def plot_trajectory(self, gps_data: pd.DataFrame, 
                       ground_truth: Optional[pd.DataFrame] = None,
                       output_path: Optional[str] = None) -> str:
        """
        Create comprehensive trajectory visualization.
        
        Args:
            gps_data: DataFrame with GPS trajectory data
            ground_truth: Optional DataFrame with ground truth positions
            output_path: Path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        logger.info("Creating trajectory visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main trajectory plot (spans 2x2)
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        
        # Error distribution plots
        ax_error_hist = fig.add_subplot(gs[0, 2])
        ax_error_time = fig.add_subplot(gs[1, 2])
        
        # Motion segments and statistics
        ax_motion = fig.add_subplot(gs[2, 0])
        ax_stats = fig.add_subplot(gs[2, 1])
        ax_3d = fig.add_subplot(gs[2, 2], projection='3d')
        
        # Plot main trajectory comparison
        self._plot_trajectory_comparison(ax_main, gps_data, ground_truth)
        
        # Plot error analysis
        if ground_truth is not None and 'gps_error_norm' in gps_data.columns:
            self._plot_error_distribution(ax_error_hist, gps_data)
            self._plot_error_timeline(ax_error_time, gps_data)
        
        # Plot motion segments
        self._plot_motion_segments(ax_motion, gps_data)
        
        # Plot statistics summary
        self._plot_statistics_summary(ax_stats, gps_data, ground_truth)
        
        # Plot 3D trajectory
        self._plot_3d_trajectory(ax_3d, gps_data, ground_truth)
        
        # Add overall title
        fig.suptitle('UAV Trajectory Analysis', fontsize=16, fontweight='bold')
        
        # Save the plot
        if output_path is None:
            output_path = "trajectory_visualization.png"
        
        plt.savefig(output_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved trajectory visualization to {output_path}")
        return output_path
    
    def _plot_trajectory_comparison(self, ax, gps_data: pd.DataFrame, 
                                  ground_truth: Optional[pd.DataFrame]):
        """Plot GPS vs ground truth trajectory comparison."""
        has_gps_data = 'gps_x' in gps_data.columns and 'gps_y' in gps_data.columns
        
        # Plot GPS trajectory
        if has_gps_data:
            ax.plot(gps_data['gps_x'], gps_data['gps_y'], 
                   'b-', alpha=0.7, linewidth=1, label='Raw GPS')
            
            # Mark start and end points
            ax.plot(gps_data['gps_x'].iloc[0], gps_data['gps_y'].iloc[0], 
                   'go', markersize=8, label='Start')
            ax.plot(gps_data['gps_x'].iloc[-1], gps_data['gps_y'].iloc[-1], 
                   'ro', markersize=8, label='End')
        
        # Plot ground truth trajectory if available
        if ground_truth is not None:
            if 'ground_truth_x' in ground_truth.columns and 'ground_truth_y' in ground_truth.columns:
                ax.plot(ground_truth['ground_truth_x'], ground_truth['ground_truth_y'], 
                       'r--', alpha=0.8, linewidth=2, label='Ground Truth')
            elif 'ground_truth_x' in gps_data.columns and 'ground_truth_y' in gps_data.columns:
                ax.plot(gps_data['ground_truth_x'], gps_data['ground_truth_y'], 
                       'r--', alpha=0.8, linewidth=2, label='Ground Truth')
        
        # Color code by GPS error if available
        if ('gps_error_norm' in gps_data.columns and has_gps_data):
            scatter = ax.scatter(gps_data['gps_x'], gps_data['gps_y'], 
                               c=gps_data['gps_error_norm'], 
                               cmap='viridis', alpha=0.6, s=10)
            plt.colorbar(scatter, ax=ax, label='GPS Error (m)')
        
        # If no GPS data available, show message
        if not has_gps_data:
            ax.text(0.5, 0.5, 'GPS trajectory data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_title('Flight Trajectory')
        if has_gps_data or ground_truth is not None:
            ax.legend()
        ax.grid(True, alpha=0.3)
        if has_gps_data:
            ax.axis('equal')
    
    def _plot_error_distribution(self, ax, gps_data: pd.DataFrame):
        """Plot GPS error distribution histogram."""
        if 'gps_error_norm' in gps_data.columns:
            errors = gps_data['gps_error_norm'].dropna()
            
            ax.hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(errors.mean(), color='red', linestyle='--', 
                      label=f'Mean: {errors.mean():.2f}m')
            ax.axvline(errors.median(), color='orange', linestyle='--', 
                      label=f'Median: {errors.median():.2f}m')
            
            ax.set_xlabel('GPS Error (m)')
            ax.set_ylabel('Frequency')
            ax.set_title('Error Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_error_timeline(self, ax, gps_data: pd.DataFrame):
        """Plot GPS error over time."""
        if 'gps_error_norm' in gps_data.columns and 'timestamp' in gps_data.columns:
            # Convert timestamp to relative time in minutes
            time_rel = (gps_data['timestamp'] - gps_data['timestamp'].iloc[0]) / 60.0
            
            ax.plot(time_rel, gps_data['gps_error_norm'], 'b-', alpha=0.7, linewidth=1)
            ax.fill_between(time_rel, gps_data['gps_error_norm'], alpha=0.3)
            
            ax.set_xlabel('Time (minutes)')
            ax.set_ylabel('GPS Error (m)')
            ax.set_title('Error Timeline')
            ax.grid(True, alpha=0.3)
    
    def _plot_motion_segments(self, ax, gps_data: pd.DataFrame):
        """Plot motion classification segments."""
        # Try to identify motion segments from acceleration data
        if all(col in gps_data.columns for col in ['imu_ax', 'imu_ay', 'imu_az']):
            # Calculate acceleration magnitude
            accel_mag = np.sqrt(gps_data['imu_ax']**2 + 
                              gps_data['imu_ay']**2 + 
                              gps_data['imu_az']**2)
            
            # Simple motion classification
            accel_threshold = self.config.get('accel_threshold', 0.5)
            motion_labels = accel_mag > accel_threshold
            
            # Plot acceleration magnitude
            if 'timestamp' in gps_data.columns:
                time_rel = (gps_data['timestamp'] - gps_data['timestamp'].iloc[0]) / 60.0
                ax.plot(time_rel, accel_mag, 'b-', alpha=0.7, linewidth=1)
                ax.axhline(accel_threshold, color='red', linestyle='--', 
                          label=f'Threshold: {accel_threshold} m/s²')
                
                # Color background by motion state
                stationary_mask = ~motion_labels
                if stationary_mask.any():
                    ax.fill_between(time_rel, 0, accel_mag.max(), 
                                  where=stationary_mask, alpha=0.2, 
                                  color='green', label='Stationary')
                
                ax.set_xlabel('Time (minutes)')
                ax.set_ylabel('Acceleration (m/s²)')
                ax.set_title('Motion Classification')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                # Plot without time axis
                ax.plot(accel_mag, 'b-', alpha=0.7, linewidth=1)
                ax.axhline(accel_threshold, color='red', linestyle='--')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Acceleration (m/s²)')
                ax.set_title('Motion Classification')
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Motion data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Motion Classification')
    
    def _plot_statistics_summary(self, ax, gps_data: pd.DataFrame, 
                               ground_truth: Optional[pd.DataFrame]):
        """Plot key statistics summary."""
        ax.axis('off')
        
        # Collect statistics
        stats_text = []
        stats_text.append(f"Dataset Statistics")
        stats_text.append(f"{'='*20}")
        stats_text.append(f"Total Samples: {len(gps_data):,}")
        
        if 'timestamp' in gps_data.columns:
            duration = (gps_data['timestamp'].max() - gps_data['timestamp'].min()) / 60.0
            stats_text.append(f"Duration: {duration:.1f} min")
        
        # GPS quality statistics
        if 'fix_type' in gps_data.columns:
            high_quality = (gps_data['fix_type'] >= 3).sum()
            stats_text.append(f"High Quality GPS: {high_quality:,} ({high_quality/len(gps_data)*100:.1f}%)")
        
        if 'hdop' in gps_data.columns:
            mean_hdop = gps_data['hdop'].mean()
            stats_text.append(f"Mean HDOP: {mean_hdop:.2f}")
        
        # Error statistics
        if 'gps_error_norm' in gps_data.columns:
            errors = gps_data['gps_error_norm'].dropna()
            stats_text.append(f"")
            stats_text.append(f"Error Statistics")
            stats_text.append(f"{'='*20}")
            stats_text.append(f"Mean Error: {errors.mean():.2f} m")
            stats_text.append(f"Std Error: {errors.std():.2f} m")
            stats_text.append(f"Max Error: {errors.max():.2f} m")
            stats_text.append(f"95th Percentile: {errors.quantile(0.95):.2f} m")
        
        # Display text
        ax.text(0.05, 0.95, '\n'.join(stats_text), 
               transform=ax.transAxes, fontsize=10, 
               verticalalignment='top', fontfamily='monospace')
    
    def _plot_3d_trajectory(self, ax, gps_data: pd.DataFrame, 
                          ground_truth: Optional[pd.DataFrame]):
        """Plot 3D trajectory visualization."""
        # Plot GPS trajectory
        if all(col in gps_data.columns for col in ['gps_x', 'gps_y', 'gps_z']):
            ax.plot(gps_data['gps_x'], gps_data['gps_y'], gps_data['gps_z'], 
                   'b-', alpha=0.7, linewidth=1, label='GPS')
        
        # Plot ground truth if available
        if ground_truth is not None:
            if all(col in ground_truth.columns for col in ['ground_truth_x', 'ground_truth_y', 'ground_truth_z']):
                ax.plot(ground_truth['ground_truth_x'], ground_truth['ground_truth_y'], 
                       ground_truth['ground_truth_z'], 'r--', alpha=0.8, linewidth=2, label='Ground Truth')
            elif all(col in gps_data.columns for col in ['ground_truth_x', 'ground_truth_y', 'ground_truth_z']):
                ax.plot(gps_data['ground_truth_x'], gps_data['ground_truth_y'], 
                       gps_data['ground_truth_z'], 'r--', alpha=0.8, linewidth=2, label='Ground Truth')
        
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Up (m)')
        ax.set_title('3D Trajectory')
        if ax.legend_:
            ax.legend()
    
    def create_error_analysis_plot(self, gps_data: pd.DataFrame, 
                                 output_path: Optional[str] = None) -> str:
        """
        Create detailed error analysis visualization.
        
        Args:
            gps_data: DataFrame with GPS error data
            output_path: Path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        if 'gps_error_norm' not in gps_data.columns:
            raise ValueError("GPS error data not available for error analysis")
        
        logger.info("Creating error analysis visualization")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('GPS Error Analysis', fontsize=16, fontweight='bold')
        
        errors = gps_data['gps_error_norm'].dropna()
        
        # Error histogram
        axes[0, 0].hist(errors, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(errors.mean(), color='red', linestyle='--', label=f'Mean: {errors.mean():.2f}m')
        axes[0, 0].set_xlabel('GPS Error (m)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error box plot
        axes[0, 1].boxplot(errors, orientation='vertical')
        axes[0, 1].set_ylabel('GPS Error (m)')
        axes[0, 1].set_title('Error Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error CDF
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[0, 2].plot(sorted_errors, cdf, 'b-', linewidth=2)
        axes[0, 2].set_xlabel('GPS Error (m)')
        axes[0, 2].set_ylabel('Cumulative Probability')
        axes[0, 2].set_title('Error CDF')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Error vs time
        if 'timestamp' in gps_data.columns:
            time_rel = (gps_data['timestamp'] - gps_data['timestamp'].iloc[0]) / 60.0
            axes[1, 0].plot(time_rel, gps_data['gps_error_norm'], 'b-', alpha=0.7)
            axes[1, 0].set_xlabel('Time (minutes)')
            axes[1, 0].set_ylabel('GPS Error (m)')
            axes[1, 0].set_title('Error vs Time')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Error vs HDOP (if available)
        if 'hdop' in gps_data.columns:
            axes[1, 1].scatter(gps_data['hdop'], gps_data['gps_error_norm'], alpha=0.5)
            axes[1, 1].set_xlabel('HDOP')
            axes[1, 1].set_ylabel('GPS Error (m)')
            axes[1, 1].set_title('Error vs HDOP')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Error statistics table
        axes[1, 2].axis('off')
        stats_text = [
            f"Error Statistics",
            f"{'='*20}",
            f"Count: {len(errors):,}",
            f"Mean: {errors.mean():.3f} m",
            f"Std: {errors.std():.3f} m",
            f"Min: {errors.min():.3f} m",
            f"Max: {errors.max():.3f} m",
            f"Median: {errors.median():.3f} m",
            f"Q25: {errors.quantile(0.25):.3f} m",
            f"Q75: {errors.quantile(0.75):.3f} m",
            f"Q95: {errors.quantile(0.95):.3f} m",
            f"Q99: {errors.quantile(0.99):.3f} m"
        ]
        axes[1, 2].text(0.05, 0.95, '\n'.join(stats_text), 
                       transform=axes[1, 2].transAxes, fontsize=11, 
                       verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = "error_analysis.png"
        
        plt.savefig(output_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved error analysis to {output_path}")
        return output_path
    
    def create_motion_visualization(self, gps_data: pd.DataFrame, 
                                  motion_labels: Optional[pd.Series] = None,
                                  output_path: Optional[str] = None) -> str:
        """
        Create motion segment visualization.
        
        Args:
            gps_data: DataFrame with IMU and GPS data
            motion_labels: Optional Series with motion classification labels
            output_path: Path to save the visualization
            
        Returns:
            Path to saved visualization file
        """
        logger.info("Creating motion visualization")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Motion Analysis', fontsize=16, fontweight='bold')
        
        # Calculate acceleration magnitude if IMU data available
        if all(col in gps_data.columns for col in ['imu_ax', 'imu_ay', 'imu_az']):
            accel_mag = np.sqrt(gps_data['imu_ax']**2 + 
                              gps_data['imu_ay']**2 + 
                              gps_data['imu_az']**2)
            
            # Plot acceleration magnitude
            if 'timestamp' in gps_data.columns:
                time_rel = (gps_data['timestamp'] - gps_data['timestamp'].iloc[0]) / 60.0
                axes[0, 0].plot(time_rel, accel_mag, 'b-', alpha=0.7)
                axes[0, 0].set_xlabel('Time (minutes)')
            else:
                axes[0, 0].plot(accel_mag, 'b-', alpha=0.7)
                axes[0, 0].set_xlabel('Sample Index')
            
            axes[0, 0].set_ylabel('Acceleration Magnitude (m/s²)')
            axes[0, 0].set_title('Acceleration Magnitude')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Calculate gyroscope magnitude if available
        if all(col in gps_data.columns for col in ['imu_gx', 'imu_gy', 'imu_gz']):
            gyro_mag = np.sqrt(gps_data['imu_gx']**2 + 
                             gps_data['imu_gy']**2 + 
                             gps_data['imu_gz']**2)
            
            # Plot gyroscope magnitude
            if 'timestamp' in gps_data.columns:
                time_rel = (gps_data['timestamp'] - gps_data['timestamp'].iloc[0]) / 60.0
                axes[0, 1].plot(time_rel, gyro_mag, 'g-', alpha=0.7)
                axes[0, 1].set_xlabel('Time (minutes)')
            else:
                axes[0, 1].plot(gyro_mag, 'g-', alpha=0.7)
                axes[0, 1].set_xlabel('Sample Index')
            
            axes[0, 1].set_ylabel('Angular Velocity Magnitude (rad/s)')
            axes[0, 1].set_title('Gyroscope Magnitude')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot trajectory colored by motion if available
        if 'gps_x' in gps_data.columns and 'gps_y' in gps_data.columns:
            if motion_labels is not None:
                # Color by motion state
                stationary_mask = motion_labels == 'stationary' if motion_labels.dtype == 'object' else ~motion_labels
                moving_mask = ~stationary_mask
                
                if stationary_mask.any():
                    axes[1, 0].scatter(gps_data.loc[stationary_mask, 'gps_x'], 
                                     gps_data.loc[stationary_mask, 'gps_y'], 
                                     c='red', alpha=0.6, s=10, label='Stationary')
                if moving_mask.any():
                    axes[1, 0].scatter(gps_data.loc[moving_mask, 'gps_x'], 
                                     gps_data.loc[moving_mask, 'gps_y'], 
                                     c='blue', alpha=0.6, s=10, label='Moving')
                axes[1, 0].legend()
            else:
                axes[1, 0].plot(gps_data['gps_x'], gps_data['gps_y'], 'b-', alpha=0.7)
            
            axes[1, 0].set_xlabel('East (m)')
            axes[1, 0].set_ylabel('North (m)')
            axes[1, 0].set_title('Trajectory by Motion State')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axis('equal')
        
        # Motion statistics
        axes[1, 1].axis('off')
        stats_text = ["Motion Statistics", "=" * 20]
        
        if motion_labels is not None:
            if motion_labels.dtype == 'object':
                stationary_count = (motion_labels == 'stationary').sum()
                moving_count = (motion_labels == 'moving').sum()
            else:
                stationary_count = (~motion_labels).sum()
                moving_count = motion_labels.sum()
            
            total = len(motion_labels)
            stats_text.extend([
                f"Total Samples: {total:,}",
                f"Stationary: {stationary_count:,} ({stationary_count/total*100:.1f}%)",
                f"Moving: {moving_count:,} ({moving_count/total*100:.1f}%)"
            ])
        
        # Add IMU statistics if available
        if all(col in gps_data.columns for col in ['imu_ax', 'imu_ay', 'imu_az']):
            accel_mag = np.sqrt(gps_data['imu_ax']**2 + gps_data['imu_ay']**2 + gps_data['imu_az']**2)
            stats_text.extend([
                "",
                "Acceleration Stats",
                "=" * 20,
                f"Mean: {accel_mag.mean():.3f} m/s²",
                f"Std: {accel_mag.std():.3f} m/s²",
                f"Max: {accel_mag.max():.3f} m/s²"
            ])
        
        axes[1, 1].text(0.05, 0.95, '\n'.join(stats_text), 
                       transform=axes[1, 1].transAxes, fontsize=11, 
                       verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_path is None:
            output_path = "motion_analysis.png"
        
        plt.savefig(output_path, dpi=self.default_dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved motion analysis to {output_path}")
        return output_path