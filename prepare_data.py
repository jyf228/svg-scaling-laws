#!/usr/bin/env python3
"""
Script to run the complete data preparation pipeline.
"""

import argparse
import logging

from src.data.pipeline import pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

def parse_args():
    p = argparse.ArgumentParser(description="Prepare SVG data")
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=["svg-icons-simple", "svg-emoji-simple", "svg-fonts-simple", "svg-stack-simple", "svgen-500k"],
        default=["svg-icons-simple"],
        help="Choose one or more datasets to download and process."
    )
    p.add_argument("--stats", action="store_true", help="Compute and print dataset stats.")
    p.add_argument("--render", action="store_true", help="Render a few sample SVGs to PNG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipeline(args.datasets, use_stats=args.stats, render=args.render)


if __name__ == "__main__":
    main()
