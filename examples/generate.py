#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DATASET GENERATION SCRIPT                              ║
║                                                                               ║
║  Run this to download and convert a dataset to standardized format.          ║
║  Customize TaskConfig and TaskPipeline in src/ for your dataset.             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python examples/generate.py --num-samples 100
    python examples/generate.py --num-samples 100 --output data/my_dataset --split train
"""

import argparse
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(
        description="Download and generate dataset in standardized format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python examples/generate.py --num-samples 10
    python examples/generate.py --num-samples 100 --output data/output --split test
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
        default="train",
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Rendered frame width in pixels (default: 832)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Rendered frame height in pixels (default: 480)"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=81,
        help="Frames per ground-truth clip (default: 81)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Frames per second for generated clips (default: 16)"
    )
    parser.add_argument(
        "--lit-style",
        type=str,
        default="darken",
        choices=["darken", "red_border"],
        help="Highlight style for answer reveal (default: darken)"
    )

    args = parser.parse_args()

    print(f"Generating dataset...")

    # ──────────────────────────────────────────────────────────────────────────
    #  Configure your pipeline here
    #  Add any additional TaskConfig parameters as needed
    # ──────────────────────────────────────────────────────────────────────────

    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
        split=args.split,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        lit_style=args.lit_style,
    )

    # Run pipeline: download → transform → write
    pipeline = TaskPipeline(config)
    samples = pipeline.run()

    print(f"Wrote {len(samples)} samples to {args.output}/")


if __name__ == "__main__":
    main()
