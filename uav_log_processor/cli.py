"""
Command-line interface for UAV Log Processor.

Provides a complete CLI for processing UAV log files into ML-ready datasets.
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Optional

from .config import ProcessingConfig
from .pipeline import UAVLogProcessor
from .utils import FileHandler


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Process UAV log files into ML-ready datasets for TCN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single log file
  uav-log-processor flight.tlog

  # Process multiple files with custom output directory
  uav-log-processor flight1.tlog flight2.bin -o /path/to/output

  # Use configuration file
  uav-log-processor *.tlog --config config.json

  # Enable verbose output and save intermediate files
  uav-log-processor flight.tlog -v --save-intermediate

  # Auto-discover log files in directory
  uav-log-processor --input-dir /path/to/logs

Supported formats: .tlog, .bin, .rlog, .txt
        """
    )

    # Input arguments
    parser.add_argument(
        'files',
        nargs='*',
        help='Log files to process'
    )
    parser.add_argument(
        '--input-dir', '-i',
        type=str,
        help='Directory to search for log files (alternative to specifying files)'
    )

    # Output arguments
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )

    # Configuration arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path (JSON format)'
    )

    parser.add_argument(
        '--save-config',
        type=str,
        help='Save current configuration to file'
    )

    # Processing parameters
    parser.add_argument(
        '--frequency', '-f',
        type=float,
        help='Target sampling frequency in Hz (default: 15.0)'
    )

    parser.add_argument(
        '--accel-threshold',
        type=float,
        help='Acceleration threshold for motion detection (m/s²)'
    )

    parser.add_argument(
        '--gyro-threshold',
        type=float,
        help='Gyroscope threshold for motion detection (rad/s)'
    )

    parser.add_argument(
        '--fusion-method',
        choices=['ekf', 'complementary', 'simple'],
        help='Sensor fusion method for ground truth generation'
    )

    # Data split arguments
    parser.add_argument(
        '--train-ratio',
        type=float,
        help='Training data ratio (default: 0.7)'
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        help='Validation data ratio (default: 0.15)'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        help='Test data ratio (default: 0.15)'
    )

    # Output control
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='Save intermediate processing results'
    )

    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip generating visualizations'
    )

    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Skip generating metadata files'
    )

    # Processing control
    parser.add_argument(
        '--jobs', '-j',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        help='Chunk size for processing large files'
    )

    # Logging and debugging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )

    # Validation and info
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate input files without processing'
    )

    parser.add_argument(
        '--list-files',
        action='store_true',
        help='List discovered log files and exit'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    return parser


def setup_logging(verbose: bool = False, debug: bool = False, quiet: bool = False):
    """Configure logging based on command line arguments."""
    if quiet:
        level = logging.ERROR
    elif debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def discover_log_files(input_dir: str) -> List[str]:
    """Discover log files in the specified directory."""
    file_handler = FileHandler()
    return file_handler.find_log_files(input_dir)


def validate_files(files: List[str]) -> List[str]:
    """Validate that input files exist and are readable."""
    valid_files = []

    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            print(f"Warning: File not found: {file_path}", file=sys.stderr)
            continue

        if not path.is_file():
            print(f"Warning: Not a file: {file_path}", file=sys.stderr)
            continue

        if not path.suffix.lower() in ['.tlog', '.bin', '.rlog', '.txt', '.log']:
            print(
                f"Warning: Unsupported file format: {file_path}", file=sys.stderr)
            continue

        valid_files.append(str(path.absolute()))

    return valid_files


def create_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """Create ProcessingConfig from command line arguments."""
    # Start with config file if provided
    if args.config:
        try:
            config = ProcessingConfig.from_file(args.config)
        except Exception as e:
            print(f"Error loading config file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        config = ProcessingConfig()

    # Override with command line arguments
    if args.frequency is not None:
        config.target_frequency = args.frequency

    if args.accel_threshold is not None:
        config.accel_threshold = args.accel_threshold

    if args.gyro_threshold is not None:
        config.gyro_threshold = args.gyro_threshold

    if args.fusion_method is not None:
        config.fusion_method = args.fusion_method

    if args.train_ratio is not None:
        config.train_ratio = args.train_ratio

    if args.val_ratio is not None:
        config.val_ratio = args.val_ratio

    if args.test_ratio is not None:
        config.test_ratio = args.test_ratio

    if args.save_intermediate:
        config.save_intermediate = True

    if args.no_visualizations:
        config.create_visualizations = False

    if args.jobs is not None:
        config.n_jobs = args.jobs

    if args.chunk_size is not None:
        config.chunk_size = args.chunk_size

    if args.verbose or args.debug:
        config.verbose = True

    # Set output directory
    config.output_dir = args.output

    return config


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle special cases first
    if not args.files and not args.input_dir:
        parser.print_help()
        return 1

    # Ensure only one input method is used
    if args.files and args.input_dir:
        print("Error: Cannot specify both files and --input-dir", file=sys.stderr)
        return 1

    # Setup logging
    setup_logging(args.verbose, args.debug, args.quiet)
    logger = logging.getLogger('uav_log_processor.cli')

    try:
        # Discover input files
        if args.input_dir:
            logger.info(f"Discovering log files in: {args.input_dir}")
            files = discover_log_files(args.input_dir)
            if not files:
                print(
                    f"No log files found in directory: {args.input_dir}", file=sys.stderr)
                return 1
        else:
            files = args.files

        # Validate files
        valid_files = validate_files(files)
        if not valid_files:
            print("No valid log files to process", file=sys.stderr)
            return 1

        # Handle list files option
        if args.list_files:
            print("Discovered log files:")
            for file_path in valid_files:
                print(f"  {file_path}")
            return 0

        # Handle validate only option
        if args.validate_only:
            print(
                f"Validation complete. Found {len(valid_files)} valid log files.")
            return 0

        # Create configuration
        config = create_config_from_args(args)

        # Save configuration if requested
        if args.save_config:
            config.to_file(args.save_config)
            logger.info(f"Configuration saved to: {args.save_config}")

        # Validate output directory
        file_handler = FileHandler()
        if not file_handler.validate_output_directory(config.output_dir):
            print(
                f"Error: Cannot write to output directory: {config.output_dir}", file=sys.stderr)
            return 1

        # Initialize processor and run
        logger.info(
            f"Starting UAV log processing with {len(valid_files)} files")
        logger.info(f"Output directory: {Path(config.output_dir).absolute()}")

        processor = UAVLogProcessor(config)
        results = processor.process_logs(valid_files)

        # Print results summary
        if not args.quiet:
            print("\nProcessing completed successfully!")
            print(f"Output files generated:")
            for file_path in results['output_files']:
                print(f"  {file_path}")

            if 'statistics' in results:
                stats = results['statistics']
                print(f"\nDataset statistics:")
                print(f"  Total samples: {stats['total_samples']}")
                for split, size in stats['dataset_sizes'].items():
                    print(f"  {split.capitalize()}: {size} samples")

                if 'error_statistics' in stats:
                    error_stats = stats['error_statistics']
                    print(f"\nGPS error statistics:")
                    print(f"  Mean error: {error_stats['mean_error_m']:.3f} m")
                    print(f"  Std error: {error_stats['std_error_m']:.3f} m")
                    print(f"  Max error: {error_stats['max_error_m']:.3f} m")

        return 0

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user", file=sys.stderr)
        return 130

    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
