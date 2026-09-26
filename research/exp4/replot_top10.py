#!/usr/bin/env python3
"""
Re-render f1_vs_clip_top10.png from an existing threshold_sweep_summary.csv.

The original plotting code hard-coded ylim=(0.97, 1.0), which is only correct for
datasets6 (F1 ~ 0.99). For any sweep whose best F1 falls below 0.97 the curves are
drawn entirely outside the axes and the figure comes out blank.

Here the axis window is derived from the data:
  * y spans a fixed band below the global best F1, so the top-10 curves separate
  * x is cropped to the region where at least one curve is inside that band

Usage:
    python replot_top10.py datasets7_sweep "Datasets7"
    python replot_top10.py datasets6_sweep "Datasets6"
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Axis window is built around where the top-10 curves actually peak.
#   X_MARGIN  CLIP padding either side of the span of top-10 optimal thresholds
#   Y_PAD_LO  headroom below the weakest top-10 peak
#   Y_PAD_HI  headroom above the global best
# Using the peaks (rather than the full curves) matters because high-ViT curves
# are near-flat across most of the CLIP range — an envelope-based rule would
# always widen back out to the full 0..1 window and hide the detail.
X_MARGIN = 0.08
Y_PAD_LO = 0.015
Y_PAD_HI = 0.003


def load_rows(csv_path: Path):
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def build_matrix(rows):
    """Return (vit_vals, clip_vals, f1_mat) with f1_mat[i, j] = F1 at vit_i, clip_j."""
    vit_vals = sorted({float(r["vit_threshold"]) for r in rows})
    clip_vals = sorted({float(r["clip_threshold"]) for r in rows})
    vi = {v: i for i, v in enumerate(vit_vals)}
    ci = {c: j for j, c in enumerate(clip_vals)}
    f1_mat = np.full((len(vit_vals), len(clip_vals)), np.nan)
    for r in rows:
        f1_mat[vi[float(r["vit_threshold"])], ci[float(r["clip_threshold"])]] = float(r["f1"])
    return np.array(vit_vals), np.array(clip_vals), f1_mat


def top10_vit(rows):
    """Best (clip, f1) per ViT threshold, keeping the 10 strongest ViT rows."""
    best = {}
    for r in rows:
        vt, f1 = float(r["vit_threshold"]), float(r["f1"])
        if vt not in best or f1 > best[vt]["f1"]:
            best[vt] = {"vit": vt, "clip": float(r["clip_threshold"]), "f1": f1}
    return sorted(best.values(), key=lambda x: x["f1"], reverse=True)[:10]


def main(sweep_dir: Path, label: str):
    rows = load_rows(sweep_dir / "threshold_sweep_summary.csv")
    best_f1 = json.loads((sweep_dir / "best_results.json").read_text())["best_f1"]

    vit_vals, clip_vals, f1_mat = build_matrix(rows)
    top_vit = top10_vit(rows)

    peak = float(best_f1["f1"])

    # --- data-driven axis window -------------------------------------------
    idxs = [int(np.argmin(np.abs(vit_vals - v["vit"]))) for v in top_vit]
    peak_clips = [v["clip"] for v in top_vit]
    peak_f1s = [v["f1"] for v in top_vit]

    # Wider margin on the left (the run-up is worth seeing) than on the right,
    # where the curves collapse to zero within a couple of steps of the peak.
    x_lo = max(float(clip_vals[0]), min(peak_clips) - X_MARGIN)
    x_hi = min(float(clip_vals[-1]), max(peak_clips) + X_MARGIN / 4)
    if x_hi - x_lo < 1e-6:
        x_lo, x_hi = float(clip_vals[0]), float(clip_vals[-1])

    y_lo = max(0.0, min(peak_f1s) - Y_PAD_LO)
    y_hi = min(1.0, peak + Y_PAD_HI)

    # --- plot ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.plasma(np.linspace(0.1, 0.95, 10))
    for vi_rank, (v, row_idx) in enumerate(zip(top_vit, idxs)):
        ax.plot(clip_vals, f1_mat[row_idx, :], color=colors[vi_rank], lw=2,
                label=f"#{vi_rank+1} ViT>{v['vit']:.2f} "
                      f"(best F1={v['f1']:.4f} @ CLIP>{v['clip']:.2f})")

    bx = float(best_f1["clip_threshold"])
    ax.plot(bx, peak, "r*", ms=25, markeredgecolor="darkred", mew=2, zorder=10)

    # Annotation placed relative to the axis window, and flipped to the left of
    # the star when the peak sits near the right edge.
    xr, yr = x_hi - x_lo, y_hi - y_lo
    right_side = bx > x_lo + 0.65 * xr
    ax.annotate(
        f"GLOBAL BEST: {best_f1['rule']}\nF1={peak:.4f}  Acc={float(best_f1['accuracy']):.4f}",
        xy=(bx, peak),
        xytext=(bx - 0.03 * xr if right_side else bx + 0.02 * xr, peak - 0.10 * yr),
        ha="right" if right_side else "left",
        fontsize=11, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="red", alpha=0.95),
    )

    ax.set_xlabel("CLIP Threshold", fontsize=14)
    ax.set_ylabel("F1 Score", fontsize=14)
    ax.set_title(f"F1 vs CLIP — Top 10 ViT Thresholds ({label})", fontsize=15)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = sweep_dir / "f1_vs_clip_top10.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print(f"  x range: {x_lo:.2f} .. {x_hi:.2f}")
    print(f"  y range: {y_lo:.4f} .. {y_hi:.4f}   (peak F1 = {peak:.4f})")


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "datasets7_sweep")
    if not d.is_absolute():
        d = Path(__file__).parent / d
    main(d, sys.argv[2] if len(sys.argv) > 2 else d.name.replace("_sweep", "").capitalize())
