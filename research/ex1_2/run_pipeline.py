#!/usr/bin/env python3
"""Run the full ex1_2 pipeline end to end.

Steps:
1. Generate image outputs from source images using ex1_2/generate_ex2.py.
2. Compare generated outputs with reference images using ex1_2/auto_perception_ex2.py.
3. Plot charts from generated reports using ex1_2/plot_ex2.py.
4. Generate summary charts from the same reports using ex1_2/plot_summary.py.

Usage:
    python run_pipeline.py --image-dir ../Images/ex2_ai_art
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import generate_ex2
import auto_perception_ex2
import plot_ex2
import plot_summary


def run_pipeline(image_dir: Path, output_dir: Path, report_dir: Path, plot_dir: Path) -> None:
    image_dir = image_dir.resolve()
    output_dir = output_dir.resolve()
    report_dir = report_dir.resolve()
    plot_dir = plot_dir.resolve()

    print("\n--- ex1_2 pipeline start ---")
    print(f"Reference images: {image_dir}")
    print(f"Generated image output: {output_dir}")
    print(f"Report output: {report_dir}")
    print(f"Plot output: {plot_dir}\n")

    if not image_dir.exists() or not image_dir.is_dir():
        raise FileNotFoundError(f"Image input directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    generate_ex2.INPUT_DIR = image_dir
    generate_ex2.OUT_DIR_BASE = output_dir
    print("[1/3] Running generate_ex2.py...")
    generate_ex2.main()

    auto_perception_ex2.REPORT_OUTPUT_DIR = report_dir
    print("\n[2/3] Running auto_perception_ex2.py...")
    auto_perception_ex2.compare_folders(str(image_dir), str(output_dir))

    plot_ex2.REPORT_DIR = report_dir
    plot_ex2.OUT_ROOT = plot_dir
    plot_ex2.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("\n[3/4] Running plot_ex2.py...")
    plot_ex2.main()

    plot_summary_dir = plot_dir / "summary_ai_art"
    plot_summary_dir.mkdir(parents=True, exist_ok=True)
    print("\n[4/4] Running plot_summary.py...")
    plot_summary.build_summary_charts(report_dir, plot_summary_dir)

    print("\n--- Pipeline finished successfully ---")
    print(f"Charts available in: {plot_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ex1_2 image pipeline.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=BASE_DIR / "Images" / "ex2_ai_art",
        help="Source image folder to use for generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "Output",
        help="Directory where generated image outputs are written.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=BASE_DIR / "table",
        help="Directory where Excel reports are written.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=BASE_DIR / "plot",
        help="Directory where plots are written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.image_dir, args.output_dir, args.report_dir, args.plot_dir)
