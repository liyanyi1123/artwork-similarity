#!/usr/bin/env python3
"""Exact AND-rule counts on a 101 x 101 threshold grid, without model inference."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate import load_scores, metrics_from_counts


def passing_counts(clip: np.ndarray, vit: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return counts[clip_index, vit_index] for strict > at every threshold.

    A score equal to a threshold does not pass it. Bin by the number of
    thresholds passed, then accumulate from the upper right. This avoids
    approximate score histograms and 10,201 repeated pair-array scans.
    """
    c = np.searchsorted(thresholds.astype(np.float64), clip.astype(np.float64), side="left")
    v = np.searchsorted(thresholds.astype(np.float64), vit.astype(np.float64), side="left")
    size = len(thresholds) + 1
    histogram = np.bincount(c * size + v, minlength=size * size).reshape(size, size)
    survival = histogram[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
    return survival[1:, 1:]


def sweep_rows(scores: dict, thresholds: np.ndarray):
    tp = passing_counts(scores["pos_clip"], scores["pos_vit"], thresholds)
    fp = passing_counts(scores["neg_clip"], scores["neg_vit"], thresholds)
    positives, negatives = len(scores["pos_clip"]), len(scores["neg_clip"])
    for vi, vt in enumerate(thresholds):
        for ci, ct in enumerate(thresholds):
            yield {"clip_threshold": round(float(ct), 2), "vit_threshold": round(float(vt), 2),
                   **metrics_from_counts(tp[ci, vi], fp[ci, vi], negatives - fp[ci, vi], positives - tp[ci, vi])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--score-mode", choices=("signed", "absolute"), default="signed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        rows = list(sweep_rows(load_scores(args.scores, args.score_mode), np.arange(0.0, 1.01, 0.01)))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["score_mode", "rule", *rows[0]])
        writer.writeheader()
        writer.writerows({"score_mode": args.score_mode, "rule": "strict AND", **row} for row in rows)
    best = max(rows, key=lambda row: row["f1"])
    print(f"Saved {len(rows):,} combinations to {args.output}")
    print(f"Best F1={best['f1']:.6f}, CLIP>{best['clip_threshold']:.2f}, ViT>{best['vit_threshold']:.2f}")


if __name__ == "__main__":
    main()
