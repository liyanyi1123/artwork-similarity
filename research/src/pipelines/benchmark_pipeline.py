#!/usr/bin/env python3
"""Run the ex1_2 pipeline for multiple batch sizes with precise per-step timing.

Usage:
    python benchmark_pipeline.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

import generate_ex2
import auto_perception_ex2
import plot_ex2
import plot_summary

TMP_BASE = Path("/tmp/ex2_timing")
OUT_BASE = Path("/tmp/ex2_timing_output")
RESULTS_JSON = Path(__file__).resolve().parent.parent / "time" / "benchmark_results.json"

BATCH_SIZES = [1, 10, 30, 50, 100]


def td(seconds: float) -> str:
    """Format seconds to H:MM:SS."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{int(h)}h {int(m)}m {s:.1f}s"
    return f"{int(m)}m {s:.1f}s"


def run_one(image_dir: Path, out_dir: Path, report_dir: Path, plot_dir: Path):
    image_dir = image_dir.resolve()
    out_dir = out_dir.resolve()
    report_dir = report_dir.resolve()
    plot_dir = plot_dir.resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── Step 1: generate_ex2 ──
    generate_ex2.INPUT_DIR = image_dir
    generate_ex2.OUT_DIR_BASE = out_dir
    t0 = time.time()
    generate_ex2.main()
    t1 = time.time()
    results["step1_generate_ex2"] = {"elapsed_s": round(t1 - t0, 3), "elapsed_fmt": td(t1 - t0)}
    print(f"  [TIMING] Step 1 generate_ex2: {td(t1 - t0)}")

    # ── Step 2: auto_perception_ex2 ──
    auto_perception_ex2.REPORT_OUTPUT_DIR = report_dir
    t0 = time.time()
    auto_perception_ex2.compare_folders(str(image_dir), str(out_dir))
    t1 = time.time()
    results["step2_auto_perception"] = {"elapsed_s": round(t1 - t0, 3), "elapsed_fmt": td(t1 - t0)}
    print(f"  [TIMING] Step 2 auto_perception: {td(t1 - t0)}")

    # ── Step 3: plot_ex2 ──
    plot_ex2.REPORT_DIR = report_dir
    plot_ex2.OUT_ROOT = plot_dir
    plot_ex2.OUT_ROOT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plot_ex2.main()
    t1 = time.time()
    results["step3_plot_ex2"] = {"elapsed_s": round(t1 - t0, 3), "elapsed_fmt": td(t1 - t0)}
    print(f"  [TIMING] Step 3 plot_ex2: {td(t1 - t0)}")

    # ── Step 4: plot_summary ──
    plot_summary_dir = plot_dir / "summary"
    plot_summary_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    plot_summary.build_summary_charts(report_dir, plot_summary_dir)
    t1 = time.time()
    results["step4_plot_summary"] = {"elapsed_s": round(t1 - t0, 3), "elapsed_fmt": td(t1 - t0)}
    print(f"  [TIMING] Step 4 plot_summary: {td(t1 - t0)}")

    return results


def main():
    all_results = {}
    total_start = time.time()

    for n in BATCH_SIZES:
        batch_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        in_dir = TMP_BASE / str(n)
        out_dir = OUT_BASE / str(n) / "Output"
        report_dir = OUT_BASE / str(n) / "table"
        plot_dir = OUT_BASE / str(n) / "plot"

        n_files = len(list(in_dir.iterdir()))
        print(f"\n{'='*60}")
        print(f"Batch: {n} images ({n_files} files)")
        print(f"  Start: {batch_start}")
        print(f"  Input:  {in_dir}")
        print(f"  Output: {out_dir}")
        print(f"{'='*60}")

        t0 = time.time()
        step_results = run_one(in_dir, out_dir, report_dir, plot_dir)
        t1 = time.time()

        total_elapsed = round(t1 - t0, 3)
        print(f"  [TIMING] Total for {n} images: {td(total_elapsed)}")

        all_results[str(n)] = {
            "batch_size": n,
            "start_time": batch_start,
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed_s": total_elapsed,
            "total_elapsed_fmt": td(total_elapsed),
            "steps": step_results,
        }

    total_end = time.time()
    print(f"\n{'='*60}")
    print(f"All batches complete. Total wall time: {td(total_end - total_start)}")
    print(f"Results saved to: {RESULTS_JSON}")

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
