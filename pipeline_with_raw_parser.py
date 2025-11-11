#!/usr/bin/env python3
"""
Example of using the raw parser in the UAV log processing pipeline.

This shows how to modify the main pipeline to use TxtParserRaw instead
of the filtered TxtParser for complete data preservation.
"""

from pathlib import Path
from uav_log_processor.pipeline import UAVLogProcessor
from uav_log_processor.config import ProcessingConfig
from uav_log_processor.parsers import TxtParserRaw

class UAVLogProcessorRaw(UAVLogProcessor):
    """UAV Log Processor that uses raw parsers for unfiltered data import."""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Replace the TXT parser with the raw version
        self.parsers['.txt'] = TxtParserRaw(self.config.get_parser_config('txt'))
        
        self.logger.info("Initialized UAV Log Processor with RAW parsers")

def process_with_raw_parser(log_files, output_dir="output_raw"):
    """Process UAV logs using the raw parser."""
    
    print("=" * 60)
    print("UAV LOG PROCESSING WITH RAW PARSER")
    print("=" * 60)
    
    # Create configuration
    config = ProcessingConfig()
    config.output_dir = output_dir
    config.save_intermediate = True
    config.create_visualizations = True
    
    # Initialize processor with raw parser
    processor = UAVLogProcessorRaw(config)
    
    # Process the logs
    try:
        results = processor.process_logs(log_files, output_dir)
        
        print("\nProcessing completed successfully!")
        print(f"Output directory: {Path(output_dir).absolute()}")
        print("\nGenerated files:")
        for file_path in results['output_files']:
            print(f"  - {file_path}")
        
        # Display statistics
        stats = results['statistics']
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        for split, size in stats['dataset_sizes'].items():
            print(f"  {split.capitalize()}: {size} samples")
        
        if 'error_statistics' in stats:
            error_stats = stats['error_statistics']
            print(f"\nGPS Error Statistics:")
            print(f"  Mean error: {error_stats['mean_error_m']:.3f} m")
            print(f"  Std error: {error_stats['std_error_m']:.3f} m")
            print(f"  Max error: {error_stats['max_error_m']:.3f} m")
        
        return results
        
    except Exception as e:
        print(f"Processing failed: {str(e)}")
        raise

def compare_filtered_vs_raw(log_files):
    """Compare results from filtered vs raw processing."""
    
    print("=" * 60)
    print("COMPARING FILTERED VS RAW PROCESSING")
    print("=" * 60)
    
    # Process with filtered parser (default)
    print("\n1. Processing with FILTERED parser...")
    config_filtered = ProcessingConfig()
    config_filtered.output_dir = "output_filtered"
    processor_filtered = UAVLogProcessor(config_filtered)
    results_filtered = processor_filtered.process_logs(log_files, "output_filtered")
    
    # Process with raw parser
    print("\n2. Processing with RAW parser...")
    config_raw = ProcessingConfig()
    config_raw.output_dir = "output_raw"
    processor_raw = UAVLogProcessorRaw(config_raw)
    results_raw = processor_raw.process_logs(log_files, "output_raw")
    
    # Compare results
    print("\n3. COMPARISON RESULTS")
    print("-" * 40)
    
    stats_filtered = results_filtered['statistics']
    stats_raw = results_raw['statistics']
    
    print(f"Filtered processing:")
    print(f"  Total samples: {stats_filtered['total_samples']}")
    if 'error_statistics' in stats_filtered:
        print(f"  Mean GPS error: {stats_filtered['error_statistics']['mean_error_m']:.3f} m")
    
    print(f"\nRaw processing:")
    print(f"  Total samples: {stats_raw['total_samples']}")
    if 'error_statistics' in stats_raw:
        print(f"  Mean GPS error: {stats_raw['error_statistics']['mean_error_m']:.3f} m")
    
    print(f"\nData retention:")
    if stats_raw['total_samples'] > 0:
        retention = stats_filtered['total_samples'] / stats_raw['total_samples'] * 100
        print(f"  Filtered keeps {retention:.1f}% of raw data")
    
    print(f"\nOutput directories:")
    print(f"  Filtered: output_filtered/")
    print(f"  Raw: output_raw/")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline_with_raw_parser.py <log_file> [mode]")
        print("Modes:")
        print("  raw     - Process with raw parser only")
        print("  compare - Compare filtered vs raw processing")
        print("\nExamples:")
        print("  python pipeline_with_raw_parser.py 'files/logs/2025-08-04 16-17-25.txt' raw")
        print("  python pipeline_with_raw_parser.py 'files/logs/2025-08-04 16-17-25.txt' compare")
        sys.exit(1)
    
    log_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "raw"
    
    log_files = [log_file]
    
    if mode == "compare":
        compare_filtered_vs_raw(log_files)
    else:
        process_with_raw_parser(log_files)
    
    print("\nDone!")