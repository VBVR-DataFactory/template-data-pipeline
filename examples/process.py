#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         DATASET PROCESSING SCRIPT                            ║
║                                                                               ║
║  Run this to download and convert a dataset to standardized format.          ║
║  Customize TaskConfig and TaskPipeline in src/ for your dataset.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python examples/process.py --num-samples 100
    python examples/process.py --num-samples 100 --output data/my_dataset --split test
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(
        description="Download and process dataset to standardized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/process.py --num-samples 10
    python examples/process.py --num-samples 100 --output data/output --split test
        """
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/questions",
        help="Output directory (default: data/questions)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)"
    )

    args = parser.parse_args()

    print(f"📦 Processing dataset...")

    # ──────────────────────────────────────────────────────────────────────────
    #  Configure your pipeline here
    #  Add any additional TaskConfig parameters as needed
    # ──────────────────────────────────────────────────────────────────────────

    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
        split=args.split,
    )

    # Run pipeline: download → process → write
    pipeline = TaskPipeline(config)
    samples = pipeline.run()

    print(f"Wrote {len(samples)} samples to {args.output}/")


if __name__ == "__main__":
    main()
