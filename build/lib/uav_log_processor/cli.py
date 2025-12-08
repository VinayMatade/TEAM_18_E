#!/usr/bin/env python3
"""
Command-line interface for UAV Log Processor.

Provides commands for cleaning logs, building noise banks, training models, and running inference.
"""

import argparse
import sys
import os
import glob
import subprocess
from pathlib import Path


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog='uav-log-processor',
        description="UAV Log Processor - Clean logs, train models, and correct GPS drift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  clean         Clean and convert raw log files to CSV format
  noise-bank    Build noise bank from cleaned logs for training
  train         Train the GPS correction model
  run           Run inference on a flight log to correct GPS drift

Examples:
  # Clean raw logs
  uav-log-processor clean -i files/logs -o files/cleaned

  # Build noise bank from cleaned logs
  uav-log-processor noise-bank -i files/cleaned -o noise_bank.npy

  # Train model
  uav-log-processor train -i files/cleaned/train -n noise_bank.npy -e 60

  # Run inference
  uav-log-processor run -m best_model.pth -i input.csv -o corrected.csv
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean and convert raw log files')
    clean_parser.add_argument('-i', '--input', required=True, help='Input folder with raw .log files')
    clean_parser.add_argument('-o', '--output', required=True, help='Output folder for cleaned CSVs')

    # Noise bank command
    noise_parser = subparsers.add_parser('noise-bank', help='Build noise bank from cleaned logs')
    noise_parser.add_argument('-i', '--input', required=True, help='Input folder with cleaned CSVs')
    noise_parser.add_argument('-o', '--output', default='noise_bank.npy', help='Output noise bank file (default: noise_bank.npy)')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train GPS correction model')
    train_parser.add_argument('-i', '--input', required=True, help='Input folder with cleaned training CSVs')
    train_parser.add_argument('-n', '--noise-bank', default='noise_bank.npy', help='Noise bank file (default: noise_bank.npy)')
    train_parser.add_argument('-e', '--epochs', type=int, default=60, help='Number of training epochs (default: 60)')
    train_parser.add_argument('-b', '--batch-size', type=int, default=128, help='Batch size (default: 128)')
    train_parser.add_argument('-s', '--seq-len', type=int, default=125, help='Sequence length (default: 125)')
    train_parser.add_argument('-lr', '--learning-rate', type=float, default=0.0005, help='Learning rate (default: 0.0005)')
    train_parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    train_parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto', help='Device to use (default: auto)')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run inference to correct GPS drift')
    run_parser.add_argument('-m', '--model', required=True, help='Path to trained model (.pth file)')
    run_parser.add_argument('-i', '--input', required=True, help='Input CSV file to correct')
    run_parser.add_argument('-o', '--output', default='corrected_flight.csv', help='Output corrected CSV (default: corrected_flight.csv)')
    run_parser.add_argument('-s', '--seq-len', type=int, default=50, help='Sequence length (default: 50)')
    run_parser.add_argument('--plot', action='store_true', help='Generate comparison plot')
    run_parser.add_argument('--device', choices=['cuda', 'cpu', 'auto'], default='auto', help='Device to use (default: auto)')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    return parser


def cmd_clean(args):
    """Execute clean command."""
    print(f"🧹 Cleaning logs from {args.input} -> {args.output}")
    
    # Import and run the cleaning logic
    try:
        # Run clean_and_convert_logs.py with modified paths
        script_path = Path(__file__).parent.parent / 'clean_and_convert_logs.py'
        
        # Set environment variables for the script
        env = os.environ.copy()
        env['INPUT_FOLDER'] = args.input
        env['OUTPUT_FOLDER'] = args.output
        
        # Execute the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=False
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error during cleaning: {e}", file=sys.stderr)
        return 1


def cmd_noise_bank(args):
    """Execute noise-bank command."""
    print(f"🔊 Building noise bank from {args.input} -> {args.output}")
    
    try:
        script_path = Path(__file__).parent.parent / 'build_noise_bank.py'
        
        env = os.environ.copy()
        env['INPUT_FOLDER'] = args.input
        env['OUTPUT_FILE'] = args.output
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=False
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error building noise bank: {e}", file=sys.stderr)
        return 1


def cmd_train(args):
    """Execute train command."""
    print(f"🚀 Training model on {args.input}")
    print(f"   Epochs: {args.epochs}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    
    try:
        script_path = Path(__file__).parent.parent / 'train.py'
        
        env = os.environ.copy()
        env['CSV_FOLDER'] = args.input
        env['NOISE_BANK_PATH'] = args.noise_bank
        env['EPOCHS'] = str(args.epochs)
        env['BATCH_SIZE'] = str(args.batch_size)
        env['SEQ_LEN'] = str(args.seq_len)
        env['LR'] = str(args.learning_rate)
        env['PATIENCE'] = str(args.patience)
        
        if args.device != 'auto':
            env['DEVICE'] = args.device
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=False
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error during training: {e}", file=sys.stderr)
        return 1


def cmd_run(args):
    """Execute run command."""
    print(f"🎯 Running inference with {args.model} on {args.input}")
    
    try:
        script_path = Path(__file__).parent.parent / 'run.py'
        
        env = os.environ.copy()
        env['MODEL_PATH'] = args.model
        env['INPUT_CSV'] = args.input
        env['OUTPUT_CSV'] = args.output
        env['SEQ_LEN'] = str(args.seq_len)
        
        if args.device != 'auto':
            env['DEVICE'] = args.device
        
        if not args.plot:
            env['SKIP_PLOT'] = '1'
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            env=env,
            capture_output=False
        )
        
        return result.returncode
        
    except Exception as e:
        print(f"❌ Error during inference: {e}", file=sys.stderr)
        return 1


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate command handler
    if args.command == 'clean':
        return cmd_clean(args)
    elif args.command == 'noise-bank':
        return cmd_noise_bank(args)
    elif args.command == 'train':
        return cmd_train(args)
    elif args.command == 'run':
        return cmd_run(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
