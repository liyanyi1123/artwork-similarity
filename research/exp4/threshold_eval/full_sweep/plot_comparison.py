#!/usr/bin/env python3
"""Generate comparison charts for full threshold sweep."""

import csv, json, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Load data
with open(OUT / "threshold_sweep_summary.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Parse into structured arrays
vit_vals = sorted(set(float(r["vit_threshold"]) for r in rows))
clip_vals = sorted(set(float(r["clip_threshold"]) for r in rows))

# Create matrices
n_vit, n_clip = len(vit_vals), len(clip_vals)
f1_mat = np.zeros((n_vit, n_clip))
acc_mat = np.zeros((n_vit, n_clip))
fp_mat = np.zeros((n_vit, n_clip))
prec_mat = np.zeros((n_vit, n_clip))

for r in rows:
    i = vit_vals.index(float(r["vit_threshold"]))
    j = clip_vals.index(float(r["clip_threshold"]))
    f1_mat[i, j] = float(r["f1"])
    acc_mat[i, j] = float(r["accuracy"])
    fp_mat[i, j] = float(r["FP"])
    prec_mat[i, j] = float(r["precision"])

# ── Figure 1: F1 Heatmap ──
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Full F1 heatmap
im1 = axes[0].imshow(f1_mat, aspect='auto', origin='lower',
                      extent=[clip_vals[0], clip_vals[-1], vit_vals[0], vit_vals[-1]],
                      cmap='RdYlGn', vmin=0.0, vmax=1.0)
axes[0].set_xlabel("CLIP Threshold", fontsize=13)
axes[0].set_ylabel("ViT Threshold", fontsize=13)
axes[0].set_title("F1 Score: Full Sweep (ViT 0-1 × CLIP 0-1, step 0.01)", fontsize=14)
# Mark best F1
best_f1 = max(rows, key=lambda r: float(r["f1"]))
axes[0].plot(float(best_f1["clip_threshold"]), float(best_f1["vit_threshold"]),
             'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
axes[0].annotate(f"Best: {best_f1['rule']}\nF1={best_f1['f1']}",
                 (float(best_f1["clip_threshold"])+0.02, float(best_f1["vit_threshold"])),
                 fontsize=9, color='black', fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
# Mark old search region
from matplotlib.patches import Rectangle
rect = Rectangle((0.80, 0.50), 0.05, 0.05, linewidth=2, edgecolor='blue',
                  facecolor='none', linestyle='--')
axes[0].add_patch(rect)
axes[0].annotate("Old search\nregion", (0.855, 0.53), fontsize=9, color='blue', fontweight='bold')
plt.colorbar(im1, ax=axes[0], label="F1 Score", shrink=0.8)

# F1 Zoom: CLIP 0.75-1.0, ViT 0-0.7
zoom_clip_start = clip_vals.index(0.75) if 0.75 in clip_vals else 75
zoom_clip_end = clip_vals.index(1.0)
zoom_vit_start = vit_vals.index(0.0)
zoom_vit_end = vit_vals.index(0.70) if 0.70 in vit_vals else 70

f1_zoom = f1_mat[zoom_vit_start:zoom_vit_end+1, zoom_clip_start:zoom_clip_end+1]
im2 = axes[1].imshow(f1_zoom, aspect='auto', origin='lower',
                      extent=[0.75, 1.0, 0.0, 0.70],
                      cmap='RdYlGn', vmin=0.95, vmax=1.0)
axes[1].set_xlabel("CLIP Threshold", fontsize=13)
axes[1].set_ylabel("ViT Threshold", fontsize=13)
axes[1].set_title("F1 Score Zoom: CLIP 0.75-1.0, ViT 0-0.7", fontsize=14)
axes[1].plot(float(best_f1["clip_threshold"]), float(best_f1["vit_threshold"]),
             'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
rect2 = Rectangle((0.80, 0.50), 0.05, 0.05, linewidth=2, edgecolor='blue',
                   facecolor='none', linestyle='--')
axes[1].add_patch(rect2)
plt.colorbar(im2, ax=axes[1], label="F1 Score", shrink=0.8)

plt.tight_layout()
fig.savefig(OUT / "f1_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved f1_heatmap.png")

# ── Figure 2: Accuracy vs F1 trade-off ──
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# FP (log scale) heatmap
fp_log = np.log10(fp_mat + 1)
im3 = axes[0].imshow(fp_log, aspect='auto', origin='lower',
                      extent=[clip_vals[0], clip_vals[-1], vit_vals[0], vit_vals[-1]],
                      cmap='RdYlGn_r')
axes[0].set_xlabel("CLIP Threshold", fontsize=13)
axes[0].set_ylabel("ViT Threshold", fontsize=13)
axes[0].set_title("False Positives (log10): Full Sweep", fontsize=14)
plt.colorbar(im3, ax=axes[0], label="log10(FP+1)", shrink=0.8)
# Mark low FP region
axes[0].axhline(y=0.50, color='white', linestyle=':', alpha=0.5)
axes[0].axvline(x=0.87, color='white', linestyle=':', alpha=0.5)

# Precision heatmap
im4 = axes[1].imshow(prec_mat, aspect='auto', origin='lower',
                      extent=[clip_vals[0], clip_vals[-1], vit_vals[0], vit_vals[-1]],
                      cmap='RdYlGn', vmin=0.0, vmax=1.0)
axes[1].set_xlabel("CLIP Threshold", fontsize=13)
axes[1].set_ylabel("ViT Threshold", fontsize=13)
axes[1].set_title("Precision: Full Sweep", fontsize=14)
plt.colorbar(im4, ax=axes[1], label="Precision", shrink=0.8)

plt.tight_layout()
fig.savefig(OUT / "fp_precision_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fp_precision_heatmap.png")

# ── Figure 3: Old vs New comparison table ──
fig, ax = plt.subplots(figsize=(16, 6))
ax.axis('off')

old_best_f1 = max(
    [r for r in rows if 0.50 <= float(r["vit_threshold"]) <= 0.55
     and 0.80 <= float(r["clip_threshold"]) <= 0.85],
    key=lambda r: float(r["f1"])
)

table_data = [
    ["Metric", "Old Best (ViT 0.50-0.55, CLIP 0.80-0.85)", "New Best (ViT 0-1, CLIP 0-1)", "Improvement"],
    ["Rule", old_best_f1["rule"], best_f1["rule"], "—"],
    ["F1 Score", old_best_f1["f1"], best_f1["f1"],
     f"+{float(best_f1['f1']) - float(old_best_f1['f1']):.4f}"],
    ["Accuracy", old_best_f1["accuracy"], best_f1["accuracy"],
     f"+{float(best_f1['accuracy']) - float(old_best_f1['accuracy']):.4f}"],
    ["Precision", old_best_f1["precision"], best_f1["precision"],
     f"+{float(best_f1['precision']) - float(old_best_f1['precision']):.4f}"],
    ["Recall", old_best_f1["recall"], best_f1["recall"],
     f"{float(best_f1['recall']) - float(old_best_f1['recall']):.4f}"],
    ["False Positives", old_best_f1["FP"], best_f1["FP"],
     f"{int(old_best_f1['FP']) - int(best_f1['FP'])} fewer"],
    ["False Negatives", old_best_f1["FN"], best_f1["FN"],
     f"+{int(best_f1['FN']) - int(old_best_f1['FN'])}"],
]

table = ax.table(cellText=table_data, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.8)
# Style header
for j in range(4):
    table[0, j].set_facecolor('#4472C4')
    table[0, j].set_text_props(color='white', fontweight='bold')
# Style data rows
for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 0:
            table[i, j].set_facecolor('#D6E4F0')
        else:
            table[i, j].set_facecolor('#FFFFFF')

ax.set_title("Threshold Comparison: Old vs New Full Sweep", fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
fig.savefig(OUT / "comparison_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved comparison_table.png")

# ── Figure 4: F1 vs CLIP threshold at key ViT values ──
fig, ax = plt.subplots(figsize=(14, 7))
key_vit = [0.30, 0.40, 0.45, 0.48, 0.49, 0.50, 0.51, 0.52, 0.55, 0.60]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(key_vit)))

for vi, vit_val in enumerate(key_vit):
    vit_idx = min(vit_vals, key=lambda x: abs(x - vit_val))
    vit_idx_i = vit_vals.index(vit_idx)
    f1_line = f1_mat[vit_idx_i, :]
    ax.plot(clip_vals, f1_line, color=colors[vi], linewidth=2,
            label=f"ViT>{vit_val:.2f}", alpha=0.8)

# Mark best
ax.plot(float(best_f1["clip_threshold"]), float(best_f1["f1"]), 'r*', markersize=20,
        markeredgecolor='darkred', markeredgewidth=1.5, zorder=10)
ax.annotate(f"Best: {best_f1['rule']}\nF1={best_f1['f1']}",
            (float(best_f1["clip_threshold"])+0.01, float(best_f1["f1"])-0.005),
            fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

ax.set_xlabel("CLIP Threshold", fontsize=13)
ax.set_ylabel("F1 Score", fontsize=13)
ax.set_title("F1 Score vs CLIP Threshold at Different ViT Thresholds", fontsize=14)
ax.legend(loc='lower left', fontsize=9, ncol=2)
ax.set_xlim(0.70, 1.0)
ax.set_ylim(0.96, 1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / "f1_vs_clip_by_vit.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved f1_vs_clip_by_vit.png")

print("\n✓ All charts generated!")
print(f"  → {OUT / 'f1_heatmap.png'}")
print(f"  → {OUT / 'fp_precision_heatmap.png'}")
print(f"  → {OUT / 'comparison_table.png'}")
print(f"  → {OUT / 'f1_vs_clip_by_vit.png'}")
