"""
Main entry point for UAV Log Processor when run as a module.

This allows running the processor with: python -m uav_log_processor
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())