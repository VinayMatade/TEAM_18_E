"""
Command-line interface for UAV Log Processor.

This module will be implemented in task 10.
"""

import argparse
import sys
from pathlib import Path
from .config import ProcessingConfig
from .pipeline import UAVLogProcessor


def main():
    """Main CLI entry point - implementation pending."""
    print("UAV Log Processor CLI")
    print("Implementation will be completed in task 10.2")
    print("For now, use the Python API directly:")
    print()
    print("from uav_log_processor import UAVLogProcessor, ProcessingConfig")
    print("processor = UAVLogProcessor()")
    print("results = processor.process_logs(['path/to/logfile.tlog'])")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())