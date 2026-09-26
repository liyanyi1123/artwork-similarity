#!/usr/bin/env python3
"""Plot WikiArt F1 against CLIP for the ten strongest ViT thresholds."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
SWEEP_DIR = BASE_DIR / "datasets6_sweep"
CSV_PATH = SWEEP_DIR / "threshold_sweep_summary.csv"
BEST_PATH = SWEEP_DIR / "best_results.json"
OUTPUT_PATH = SWEEP_DIR / "f1_vs_clip_top10_large.png"


def load_rows():
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_matrix(rows):
    vit_values = sorted({float(row["vit_threshold"]) for row in rows})
    clip_values = sorted({float(row["clip_threshold"]) for row in rows})
    vit_index = {value: index for index, value in enumerate(vit_values)}
    clip_index = {value: index for index, value in enumerate(clip_values)}
    f1_matrix = np.full((len(vit_values), len(clip_values)), np.nan)

    for row in rows:
        i = vit_index[float(row["vit_threshold"])]
        j = clip_index[float(row["clip_threshold"])]
        f1_matrix[i, j] = float(row["f1"])

    return np.array(vit_values), np.array(clip_values), f1_matrix


def top_vit_thresholds(rows):
    best_by_vit = {}
    for row in rows:
        vit = float(row["vit_threshold"])
        f1 = float(row["f1"])
        if vit not in best_by_vit or f1 > best_by_vit[vit]["f1"]:
            best_by_vit[vit] = {
                "vit": vit,
                "clip": float(row["clip_threshold"]),
                "f1": f1,
            }
    return sorted(best_by_vit.values(), key=lambda item: item["f1"], reverse=True)[:10]


def main():
    rows = load_rows()
    best = json.loads(BEST_PATH.read_text(encoding="utf-8"))["best_f1"]
    vit_values, clip_values, f1_matrix = build_matrix(rows)
    top_vits = top_vit_thresholds(rows)

    vit_indices = [
        int(np.argmin(np.abs(vit_values - item["vit"]))) for item in top_vits
    ]
    peak_clips = [item["clip"] for item in top_vits]
    peak_f1s = [item["f1"] for item in top_vits]
    peak_f1 = float(best["f1"])

    x_min = max(float(clip_values[0]), min(peak_clips) - 0.08)
    x_max = min(float(clip_values[-1]), max(peak_clips) + 0.02)
    y_min = max(0.0, min(peak_f1s) - 0.015)
    y_max = min(1.0, peak_f1 + 0.003)

    fig, axis = plt.subplots(figsize=(24, 13))
    colors = plt.cm.plasma(np.linspace(0.1, 0.95, len(top_vits)))

    for rank, (item, row_index) in enumerate(zip(top_vits, vit_indices), start=1):
        axis.plot(
            clip_values,
            f1_matrix[row_index, :],
            color=colors[rank - 1],
            linewidth=4,
            label=(
                f"#{rank} ViT>{item['vit']:.2f} "
                f"(best F1={item['f1']:.4f} @ CLIP>{item['clip']:.2f})"
            ),
        )

    best_clip = float(best["clip_threshold"])
    axis.plot(
        best_clip,
        peak_f1,
        "r*",
        markersize=50,
        markeredgecolor="darkred",
        markeredgewidth=4,
        zorder=10,
    )

    x_span = x_max - x_min
    y_span = y_max - y_min
    axis.annotate(
        f"GLOBAL BEST: {best['rule']}\nF1={peak_f1:.4f}  Acc={float(best['accuracy']):.4f}",
        xy=(best_clip, peak_f1),
        xytext=(best_clip - 0.04 * x_span, peak_f1 - 0.18 * y_span),
        horizontalalignment="right",
        fontsize=22,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "darkred", "lw": 3},
        bbox={
            "boxstyle": "round",
            "facecolor": "lightyellow",
            "edgecolor": "red",
            "alpha": 0.95,
            "linewidth": 2,
        },
    )

    axis.set_xlabel("CLIP Threshold", fontsize=28)
    axis.set_ylabel("F1 Score", fontsize=28)
    axis.set_title(
        "F1 vs CLIP Threshold - Top 10 ViT Thresholds (WikiArt-1K)",
        fontsize=30,
        pad=24,
    )
    axis.tick_params(axis="both", labelsize=20, width=2, length=8)
    axis.legend(loc="lower left", fontsize=18, framealpha=0.95)
    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.grid(True, alpha=0.3, linewidth=1.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {OUTPUT_PATH}")
    print(f"x range: {x_min:.2f} .. {x_max:.2f}")
    print(f"y range: {y_min:.4f} .. {y_max:.4f}")


if __name__ == "__main__":
    main()
