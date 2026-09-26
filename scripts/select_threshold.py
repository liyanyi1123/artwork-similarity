#!/usr/bin/env python3
"""Reproduce the high-performance-region weighting recorded in the slides."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def load_grid(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    grid = {}
    for row in rows:
        key = (float(row["clip_threshold"]), float(row["vit_threshold"]))
        f1 = float(row["f1"])
        if not all(math.isfinite(x) for x in (*key, f1)) or not 0 <= f1 <= 1:
            raise ValueError(f"Invalid threshold/F1 in {path}")
        if key in grid:
            raise ValueError(f"Duplicate threshold pair in {path}: {key}")
        grid[key] = f1
    if not grid:
        raise ValueError(f"Empty grid: {path}")
    return grid


def select(grids: list[dict], cutoff: float):
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("F1 cutoff must be in [0, 1]")
    keys = set(grids[0])
    if any(set(grid) != keys for grid in grids):
        raise ValueError("Every dataset must use the same threshold grid")
    counts = [sum(value > cutoff for value in grid.values()) for grid in grids]
    if not sum(counts):
        raise ValueError("No grid point exceeds the F1 cutoff")
    weights = [count / sum(counts) for count in counts]
    # Deterministic first maximum: ViT ascending, then CLIP ascending.
    ordered = sorted(keys, key=lambda item: (item[1], item[0]))
    best = max(ordered, key=lambda key: sum(weight * grid[key] for weight, grid in zip(weights, grids)))
    return {"f1_cutoff": cutoff, "region_comparison": ">", "grid_points": len(keys),
            "region_counts": counts, "weights": weights, "clip_threshold": best[0],
            "vit_threshold": best[1], "dataset_f1": [grid[best] for grid in grids],
            "weighted_f1": sum(weight * grid[best] for weight, grid in zip(weights, grids)),
            "scope": "Selection on supplied grids; no independent test split is created."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grids", type=Path, nargs="+")
    parser.add_argument("--f1-cutoff", type=float, default=0.99)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = select([load_grid(path) for path in args.grids], args.f1_cutoff)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
