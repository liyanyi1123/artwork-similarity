"""Plot charts from test_report.xlsx — ViT/CLIP + hash similarities for 793 combinations."""
import openpyxl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent
INPUT_XLSX = BASE_DIR / "test_report.xlsx"
OUT_DIR = BASE_DIR / "plot" / "combinations_1_to_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Read data
# ---------------------------------------------------------------------------
wb = openpyxl.load_workbook(str(INPUT_XLSX))
ws = wb.active

rows = list(ws.iter_rows(min_row=2, values_only=True))
data = []
for row in rows:
    op_name = row[0]
    if op_name is None:
        continue
    ops = str(op_name).split("_")  # e.g. "B_K_N" → ["B", "K", "N"]
    n_ops = len(ops)
    data.append({
        "name": str(op_name),
        "ops": ops,
        "n_ops": n_ops,
        "vit":  row[3] if isinstance(row[3], (int, float)) else None,
        "clip": row[4] if isinstance(row[4], (int, float)) else None,
        "phash": row[5] if isinstance(row[5], (int, float)) else None,
        "ahash": row[6] if isinstance(row[6], (int, float)) else None,
        "dhash": row[7] if isinstance(row[7], (int, float)) else None,
        "whash": row[8] if isinstance(row[8], (int, float)) else None,
        "pdq":   row[9] if isinstance(row[9], (int, float)) else None,
    })

print(f"Loaded {len(data)} data rows")

# ---------------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})

###############################################################################
# 1. BAR CHART: Mean ViT & CLIP similarity per operation count (k=1..4)
###############################################################################
fig, ax = plt.subplots(figsize=(8, 5))
k_vals = [1, 2, 3, 4]
vit_means = [np.mean([d["vit"] for d in data if d["n_ops"] == k]) for k in k_vals]
clip_means = [np.mean([d["clip"] for d in data if d["n_ops"] == k]) for k in k_vals]

x = np.arange(len(k_vals))
w = 0.3
ax.bar(x - w/2, vit_means, w, color=(0.298, 0.447, 0.690), label="ViT", edgecolor="white")
ax.bar(x + w/2, clip_means, w, color=(0.769, 0.306, 0.322), label="CLIP", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}\n(n={len([d for d in data if d['n_ops']==k])})" for k in k_vals])
ax.set_ylabel("Cosine Similarity")
ax.set_title("Mean ViT & CLIP Similarity by Number of Operations")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i, (vm, cm) in enumerate(zip(vit_means, clip_means)):
    ax.text(i - w/2, vm + 0.01, f"{vm:.4f}", ha="center", fontsize=8)
    ax.text(i + w/2, cm + 0.01, f"{cm:.4f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "01_vit_clip_by_k_bar.png")
plt.close(fig)

###############################################################################
# 2. LINE CHART: ViT & CLIP sorted by ViT similarity (all 793 points)
###############################################################################
sorted_data = sorted(data, key=lambda d: d["vit"] or 0)
fig, ax = plt.subplots(figsize=(18, 5))
x_all = np.arange(len(sorted_data))
vit_all = [d["vit"] or 0 for d in sorted_data]
clip_all = [d["clip"] or 0 for d in sorted_data]
ax.plot(x_all, vit_all, linewidth=0.6, color=(0.298, 0.447, 0.690), label="ViT", alpha=0.8)
ax.plot(x_all, clip_all, linewidth=0.6, color=(0.769, 0.306, 0.322), label="CLIP", alpha=0.8)
# Mark k-group boundaries
for k in range(1, 5):
    k_start = next((i for i, d in enumerate(sorted_data) if d["n_ops"] == k), 0)
    for txt in [(12, "k=1"), (12+66, "k=2"), (12+66+220, "k=3")]:
        if txt[0] < len(sorted_data):
            ax.axvline(x=txt[0] - 0.5, color="gray", linestyle="--", alpha=0.4)
ax.set_title("ViT & CLIP Cosine Similarity — all 793 images (sorted by ViT)")
ax.set_ylabel("Cosine Similarity")
ax.set_xlabel("Images (sorted by ViT similarity, grouped by k=1..4)")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT_DIR / "02_vit_clip_sorted_line.png")
plt.close(fig)

###############################################################################
# 3. BAR CHART: Mean hash similarities per operation count
###############################################################################
fig, ax = plt.subplots(figsize=(10, 5))
hash_names = ["phash", "ahash", "dhash", "whash", "pdq"]
hash_labels = ["pHash", "aHash", "dHash", "wHash", "PDQ"]
hash_colors = [(0.298, 0.447, 0.690), (0.333, 0.659, 0.408),
               (0.769, 0.306, 0.322), (0.506, 0.447, 0.698), (0.800, 0.725, 0.455)]

n_h = len(hash_names)
w_h = 0.7 / n_h
for j, (hname, hcolor, hlabel) in enumerate(zip(hash_names, hash_colors, hash_labels)):
    means = [np.mean([d[hname] for d in data if d["n_ops"] == k and d[hname] is not None]) for k in k_vals]
    offset = (j - n_h/2 + 0.5) * w_h
    ax.bar(x + offset, means, w_h, color=hcolor, label=hlabel, edgecolor="white", linewidth=0.2)

ax.set_xticks(x)
ax.set_xticklabels([f"k={k}\n(n={len([d for d in data if d['n_ops']==k])})" for k in k_vals])
ax.set_ylabel("Similarity (0–1)")
ax.set_title("Mean Hash Similarities by Number of Operations")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "03_hash_by_k_bar.png")
plt.close(fig)

###############################################################################
# 4. LINE CHART: Hash similarities, sorted (all 793)
###############################################################################
sorted_by_phash = sorted(data, key=lambda d: d["phash"] or 0)
fig, ax = plt.subplots(figsize=(18, 5))
x_all = np.arange(len(sorted_by_phash))
for hname, hcolor, hlabel in zip(hash_names, hash_colors, hash_labels):
    vals = [d[hname] or 0 for d in sorted_by_phash]
    ax.plot(x_all, vals, linewidth=0.4, color=hcolor, label=hlabel, alpha=0.7)

ax.set_title("Perceptual Hash Similarities — all 793 images (sorted by pHash)")
ax.set_ylabel("Similarity (0–1)")
ax.set_xlabel("Images (sorted by pHash similarity, grouped by k=1..4)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.grid(alpha=0.2)
for boundary, label in [(12, "k=1"), (12+66, "k=2"), (12+66+220, "k=3")]:
    ax.axvline(x=boundary - 0.5, color="gray", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(OUT_DIR / "04_hash_sorted_line.png")
plt.close(fig)

###############################################################################
# 5. Impact heatmap: Which individual operation drops similarity most?
###############################################################################
# For k=1 data (single operations) compare metrics across the 12 ops
k1_data = [d for d in data if d["n_ops"] == 1]
k1_data.sort(key=lambda d: d["vit"] or 0)

op_names = [d["name"] for d in k1_data]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.bar(np.arange(len(k1_data)), [d["vit"] or 0 for d in k1_data], 0.35,
        color=(0.298, 0.447, 0.690), label="ViT", edgecolor="white")
ax1.bar(np.arange(len(k1_data)) + 0.35, [d["clip"] or 0 for d in k1_data], 0.35,
        color=(0.769, 0.306, 0.322), label="CLIP", edgecolor="white")
ax1.set_ylabel("Cosine Similarity")
ax1.set_title("ViT & CLIP: Single-Operation Impact (k=1)")
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

for j, (hname, hcolor, hlabel) in enumerate(zip(hash_names, hash_colors, hash_labels)):
    vals = [d[hname] or 0 for d in k1_data]
    offset = (j - n_h/2 + 0.5) * w_h
    ax2.bar(np.arange(len(k1_data)) + offset, vals, w_h, color=hcolor,
            label=hlabel, edgecolor="white", linewidth=0.2)

ax2.set_xticks(np.arange(len(k1_data)))
ax2.set_xticklabels(op_names, fontsize=8)
ax2.set_ylabel("Similarity (0–1)")
ax2.set_title("Hash Similarities: Single-Operation Impact (k=1)")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8, ncol=5, loc="upper left")
ax2.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_DIR / "05_single_op_impact.png")
plt.close(fig)

###############################################################################
# 6. Summary scatter: ViT vs CLIP for k=1..4, color by operation count
###############################################################################
fig, ax = plt.subplots(figsize=(8, 7))
k_colors = {1: (0.298, 0.447, 0.690), 2: (0.333, 0.659, 0.408),
            3: (0.769, 0.306, 0.322), 4: (0.506, 0.447, 0.698)}
for k in [1, 2, 3, 4]:
    kd = [d for d in data if d["n_ops"] == k]
    xv = [d["vit"] or 0 for d in kd]
    yv = [d["clip"] or 0 for d in kd]
    ax.scatter(xv, yv, c=[k_colors[k]], alpha=0.5, s=8, label=f"k={k} (n={len(kd)})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
ax.set_xlabel("ViT Cosine Similarity")
ax.set_ylabel("CLIP Cosine Similarity")
ax.set_title("ViT vs CLIP Similarity, Colored by Operation Count")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "06_vit_vs_clip_scatter.png")
plt.close(fig)

###############################################################################
# 7. ViT / CLIP decline: line chart of mean similarity as k increases
###############################################################################
fig, ax = plt.subplots(figsize=(7, 5))
# For each k, also show the spread (std band)
vit_std = [np.std([d["vit"] for d in data if d["n_ops"] == k]) for k in k_vals]
clip_std = [np.std([d["clip"] for d in data if d["n_ops"] == k]) for k in k_vals]

ax.plot(k_vals, vit_means, "o-", color=(0.298, 0.447, 0.690), linewidth=2, markersize=8, label="ViT")
ax.fill_between(k_vals, [m-s for m,s in zip(vit_means, vit_std)],
                [m+s for m,s in zip(vit_means, vit_std)],
                color=(0.298, 0.447, 0.690), alpha=0.15)
ax.plot(k_vals, clip_means, "s-", color=(0.769, 0.306, 0.322), linewidth=2, markersize=8, label="CLIP")
ax.fill_between(k_vals, [m-s for m,s in zip(clip_means, clip_std)],
                [m+s for m,s in zip(clip_means, clip_std)],
                color=(0.769, 0.306, 0.322), alpha=0.15)
ax.set_xticks(k_vals)
ax.set_xlabel("Number of Operations (k)")
ax.set_ylabel("Cosine Similarity")
ax.set_title("ViT & CLIP Similarity Decline with More Operations")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "07_vit_clip_decline_line.png")
plt.close(fig)

###############################################################################
# 8. Hash similarity decline: line chart of mean similarity as k increases
###############################################################################
fig, ax = plt.subplots(figsize=(8, 5))
for hname, hcolor, hlabel in zip(hash_names, hash_colors, hash_labels):
    means = [np.mean([d[hname] for d in data if d["n_ops"] == k and d[hname] is not None]) for k in k_vals]
    ax.plot(k_vals, means, "o-", color=hcolor, linewidth=1.5, markersize=6, label=hlabel)

ax.set_xticks(k_vals)
ax.set_xlabel("Number of Operations (k)")
ax.set_ylabel("Similarity (0–1)")
ax.set_title("Hash Similarity Decline with More Operations")
ax.set_ylim(0.4, 1.05)
ax.legend(fontsize=8, ncol=5)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "08_hash_increase_line.png")
plt.close(fig)

print(f"✅ 8 charts saved to {OUT_DIR.resolve()}")
for f in sorted(OUT_DIR.iterdir()):
    if f.is_file():
        print(f"   {f.name}")

