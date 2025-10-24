"""
Main pipeline orchestrator for UAV log processing.

Coordinates all processing components to transform raw log files into ML-ready datasets.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

from .config import ProcessingConfig
from .parsers import BaseLogParser, TLogParser, BinParser, RLogParser, TxtParser
from .processors import (
    DataSynchronizer, MotionClassifier, GroundTruthGenerator,
    ErrorCalculator, DatasetFormatter, MetadataGenerator, ReproducibilityManager
)
from .utils import FileHandler, DataValidator, TrajectoryVisualizer


class UAVLogProcessor:
    """Main pipeline for processing UAV log files into ML datasets."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the UAV log processor.
        
        Args:
            config: Processing configuration. If None, uses default config.
        """
        self.config = config or ProcessingConfig()
        self.logger = self._setup_logging()
        
        # Initialize parsers
        self.parsers = {
            '.tlog': TLogParser(self.config.get_parser_config('tlog')),
            '.bin': BinParser(self.config.get_parser_config('bin')),
            '.rlog': RLogParser(self.config.get_parser_config('rlog')),
            '.txt': TxtParser(self.config.get_parser_config('txt'))
        }
        
        # Initialize processors
        self.synchronizer = DataSynchronizer(self.config.__dict__)
        self.motion_classifier = MotionClassifier(self.config.__dict__)
        self.ground_truth_generator = GroundTruthGenerator(self.config.__dict__)
        self.error_calculator = ErrorCalculator(self.config.__dict__)
        self.dataset_formatter = DatasetFormatter(self.config.__dict__)
        self.metadata_generator = MetadataGenerator(self.config.__dict__)
        self.reproducibility_manager = ReproducibilityManager(self.config.__dict__)
        
        # Initialize utilities
        self.file_handler = FileHandler(self.config.__dict__)
        self.validator = DataValidator(self.config.__dict__)
        self.visualizer = TrajectoryVisualizer(self.config.__dict__)
        
        # Processing state
        self.processing_stats = {}
        self.intermediate_data = {}
    
    def process_logs(self, log_files: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple UAV log files into ML-ready datasets.
        
        Args:
            log_files: List of paths to log files to process
            output_dir: Output directory (overrides config if provided)
            
        Returns:
            Dictionary with processing results and statistics
        """
        if output_dir:
            self.config.output_dir = output_dir
        
        # Create output directory
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting UAV log processing with {len(log_files)} files")
        self.logger.info(f"Output directory: {output_path.absolute()}")
        
        try:
            # Step 1: Parse all log files
            parsed_data = self._parse_logs(log_files)
            
            # Step 2: Synchronize data streams
            synchronized_data = self._synchronize_data(parsed_data)
            
            # Step 3: Classify motion segments
            data_with_motion = self._classify_motion(synchronized_data)
            
            # Step 4: Standardize column names for downstream processing
            standardized_data = self._standardize_columns(data_with_motion)
            
            # Step 5: Generate ground truth positions
            ground_truth = self._generate_ground_truth(standardized_data, standardized_data.get('motion_label'))
            
            # Step 6: Calculate GPS errors
            error_data = self._calculate_errors(standardized_data, ground_truth)
            
            # Step 7: Format and split dataset
            datasets, metadata = self._format_datasets(error_data)
            
            # Step 7: Generate outputs
            results = self._generate_outputs(datasets, metadata, output_path)
            
            self.logger.info("UAV log processing completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
    
    def _parse_logs(self, log_files: List[str]) -> Dict[str, pd.DataFrame]:
        """Parse all log files using appropriate parsers."""
        self.logger.info("Parsing log files...")
        parsed_data = {}
        
        for file_path in log_files:
            path = Path(file_path)
            extension = path.suffix.lower()
            
            if extension not in self.parsers:
                self.logger.warning(f"Unsupported file format: {extension} for {file_path}")
                continue
            
            try:
                parser = self.parsers[extension]
                if not parser.validate_file(file_path):
                    self.logger.warning(f"Invalid file: {file_path}")
                    continue
                
                self.logger.info(f"Parsing {path.name} ({extension})")
                data = parser.parse(file_path)
                parsed_data[file_path] = data
                
                self.logger.info(f"Parsed {len(data)} records from {path.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to parse {file_path}: {str(e)}")
                continue
        
        if not parsed_data:
            raise ValueError("No valid log files could be parsed")
        
        self.intermediate_data['parsed'] = parsed_data
        return parsed_data
    
    def _synchronize_data(self, parsed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Synchronize all parsed data streams."""
        self.logger.info("Synchronizing data streams...")
        
        synchronized = self.synchronizer.synchronize_streams(parsed_data)
        
        self.logger.info(f"Synchronized to {len(synchronized)} records at {self.config.target_frequency} Hz")
        
        if self.config.save_intermediate:
            output_path = Path(self.config.output_dir) / "aligned_full.csv"
            synchronized.to_csv(output_path, index=False)
            self.logger.info(f"Saved synchronized data to {output_path}")
        
        self.intermediate_data['synchronized'] = synchronized
        return synchronized
    
    def _classify_motion(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify motion segments in the data."""
        self.logger.info("Classifying motion segments...")
        
        # Use process method which adds motion_label column to the data
        data_with_motion = self.motion_classifier.process(data)
        
        if 'motion_label' in data_with_motion.columns:
            motion_labels = data_with_motion['motion_label']
            stationary_count = (motion_labels == 'stationary').sum()
            moving_count = (motion_labels == 'moving').sum()
            
            self.logger.info(f"Classified {stationary_count} stationary and {moving_count} moving samples")
            
            self.intermediate_data['motion_labels'] = motion_labels
        else:
            self.logger.warning("Motion classification did not add motion_label column")
        
        return data_with_motion
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for downstream processing."""
        self.logger.info("Standardizing column names...")
        
        # Use dataset formatter to standardize column names
        standardized_data = self.dataset_formatter.standardize_features(data)
        
        self.logger.info(f"Standardized {len(standardized_data.columns)} columns")
        
        self.intermediate_data['standardized'] = standardized_data
        return standardized_data
    
    def _generate_ground_truth(self, data: pd.DataFrame, motion_labels: pd.Series) -> pd.DataFrame:
        """Generate ground truth positions using sensor fusion."""
        self.logger.info("Generating ground truth positions...")
        
        ground_truth = self.ground_truth_generator.generate(data, motion_labels=motion_labels)
        
        self.logger.info(f"Generated ground truth for {len(ground_truth)} samples")
        
        self.intermediate_data['ground_truth'] = ground_truth
        return ground_truth
    
    def _calculate_errors(self, data: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
        """Calculate GPS error vectors."""
        self.logger.info("Calculating GPS errors...")
        
        combined_data = data.copy()
        for col in ground_truth.columns:
            if col == 'timestamp' and col in combined_data.columns:
                continue
            combined_data[col] = ground_truth[col]

        error_enriched_data = self.error_calculator.process(combined_data)
        error_data = error_enriched_data[[
            'gps_error_x',
            'gps_error_y',
            'gps_error_z',
            'gps_error_norm'
        ]]
        
        mean_error = error_data['gps_error_norm'].mean()
        self.logger.info(f"Mean GPS error: {mean_error:.3f} meters")
        
        self.intermediate_data['errors'] = error_data
        return error_enriched_data
    
    def _format_datasets(self, data: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """Format and split datasets for ML training."""
        self.logger.info("Formatting datasets...")
        
        formatted_data, metadata = self.dataset_formatter.format_dataset(data)

        schema_metadata = self.metadata_generator.process(formatted_data)

        combined_metadata = schema_metadata.copy()
        combined_metadata['normalization_stats'] = metadata.get('normalization_stats', {})
        combined_metadata['feature_list'] = metadata.get('features', [])
        combined_metadata['categorical_features'] = metadata.get('categorical_features', [])
        combined_metadata['continuous_features'] = metadata.get('continuous_features', [])

        combined_metadata['processing_config'] = {
            key: value for key, value in self.config.__dict__.items() if not key.startswith('_')
        }
        
        train_df, val_df, test_df = self.dataset_formatter.split_dataset(
            formatted_data, 
            self.config.train_ratio,
            self.config.val_ratio
        )
        
        datasets = {
            'train': train_df,
            'validation': val_df,
            'test': test_df
        }
        
        self.logger.info(f"Split into train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
        
        return datasets, combined_metadata
    
    def _generate_outputs(self, datasets: Dict[str, pd.DataFrame], 
                         metadata: Dict[str, Any], 
                         output_path: Path) -> Dict[str, Any]:
        """Generate all output files and visualizations."""
        self.logger.info("Generating outputs...")
        
        results = {
            'output_files': [],
            'statistics': {},
            'metadata': metadata
        }
        
        # Save dataset files
        for split_name, df in datasets.items():
            file_path = output_path / f"{split_name}.csv"
            df.to_csv(file_path, index=False)
            results['output_files'].append(str(file_path))
            self.logger.info(f"Saved {split_name} dataset: {file_path}")
        
        # Save metadata
        metadata_path = output_path / "metadata.json"
        self.file_handler.save_json(metadata, str(metadata_path))
        results['output_files'].append(str(metadata_path))
        
        if self.config.save_intermediate:
            aligned_path = output_path / "aligned_full.csv"
            if aligned_path.exists() and str(aligned_path) not in results['output_files']:
                results['output_files'].append(str(aligned_path))

        # Generate visualizations if enabled
        if self.config.create_visualizations and 'synchronized' in self.intermediate_data:
            viz_path = output_path / "visualization.png"
            self.visualizer.plot_trajectory(
                self.intermediate_data['synchronized'],
                self.intermediate_data.get('ground_truth'),
                str(viz_path)
            )
            results['output_files'].append(str(viz_path))
            self.logger.info(f"Generated visualization: {viz_path}")
        
        # Calculate processing statistics
        results['statistics'] = self._calculate_statistics(datasets)
        
        return results
    
    def _calculate_statistics(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calculate processing statistics."""
        stats = {
            'total_samples': sum(len(df) for df in datasets.values()),
            'dataset_sizes': {name: len(df) for name, df in datasets.items()},
            'processing_time': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
        
        if 'errors' in self.intermediate_data:
            error_data = self.intermediate_data['errors']
            stats['error_statistics'] = {
                'mean_error_m': float(error_data['gps_error_norm'].mean()),
                'std_error_m': float(error_data['gps_error_norm'].std()),
                'max_error_m': float(error_data['gps_error_norm'].max()),
                'median_error_m': float(error_data['gps_error_norm'].median())
            }
        
        return stats
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('uav_log_processor')
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        return logger