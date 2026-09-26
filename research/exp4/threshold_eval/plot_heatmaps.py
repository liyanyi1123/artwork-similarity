#!/usr/bin/env python3
"""Generate two heatmaps: F1 score and False Positives for threshold sweep."""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Helvetica"]

OUT = Path(__file__).resolve().parent

# ── Load data ──
with open(OUT / "threshold_sweep_summary.csv", "r") as f:
    rows = list(csv.DictReader(f))

vit_vals = sorted(set(float(r["vit_threshold"]) for r in rows))
clip_vals = sorted(set(float(r["clip_threshold"]) for r in rows))

def build_matrix(key, as_int=False):
    m = np.zeros((len(vit_vals), len(clip_vals)))
    for r in rows:
        i = vit_vals.index(float(r["vit_threshold"]))
        j = clip_vals.index(float(r["clip_threshold"]))
        m[i, j] = float(r[key])
    return m.astype(int) if as_int else m

f1 = build_matrix("f1")
fp = build_matrix("FP", as_int=True)

# ── Shared style ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Colormaps
cmap_f1 = plt.cm.YlOrRd
cmap_fp = plt.cm.RdYlGn_r   # reversed: red = many FP (bad), green = few FP (good)

titles = {
    "f1": "F1 Score",
    "FP": "False Positives (FP)",
}

for ax, key, cmap in zip(axes, ["f1", "FP"], [cmap_f1, cmap_fp]):
    m = build_matrix(key, as_int=(key == "FP"))
    im = ax.imshow(m, cmap=cmap, aspect="equal", origin="lower")

    # Annotate each cell
    for i in range(len(vit_vals)):
        for j in range(len(clip_vals)):
            val = m[i, j]
            text = f"{val:.4f}" if key == "f1" else f"{val:,}"
            # White text on dark cells, black on light
            if key == "f1":
                color = "white" if val > 0.991 else "black"
            else:
                color = "white" if val < 400 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(len(clip_vals)))
    ax.set_xticklabels([f"{v:.2f}" for v in clip_vals])
    ax.set_yticks(range(len(vit_vals)))
    ax.set_yticklabels([f"{v:.2f}" for v in vit_vals])
    ax.set_xlabel("CLIP Threshold", fontsize=12, fontweight="bold")
    ax.set_ylabel("ViT Threshold", fontsize=12, fontweight="bold")
    ax.set_title(titles[key], fontsize=14, fontweight="bold", pad=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(labelsize=8)

    # Baseline marker
    baseline_i = vit_vals.index(0.50)
    baseline_j = clip_vals.index(0.80)
    ax.add_patch(plt.Rectangle((baseline_j - 0.5, baseline_i - 0.5), 1, 1,
                                fill=False, edgecolor="#2196F3", linewidth=2.5, linestyle="--"))
    ax.text(baseline_j, baseline_i - 0.75, "baseline", ha="center", fontsize=7,
            color="#2196F3", fontweight="bold")

plt.tight_layout(pad=1.5)
out_path = OUT / "threshold_heatmaps.png"
fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out_path}")
