#!/usr/bin/env python3
"""Generate top-10 ViT negative pair visualization for Datasets5 — single-column layout."""

import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "analysis" / "ComprehensiveAnalysisAug3" / "top10_vit_negative_pairs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────
emb = np.load("analysis/threshold_eval_datasets5/embeddings/vit_embeddings.npz")
orig_vit = emb["orig_vit"]
n = 1000

sim_matrix = orig_vit @ orig_vit.T
triu_idx = np.triu_indices(n, k=1)
neg_sims = sim_matrix[triu_idx]

styles = ["art_nouveau","baroque","expressionism","impressionism",
          "post_impressionism","realism","renaissance","romanticism","surrealism","ukiyo_e"]

originals = []
for style in styles:
    style_dir = Path("Datasets5") / style
    for fpath in sorted(style_dir.iterdir()):
        if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            originals.append(fpath)

# Top 10
top_indices = np.argpartition(neg_sims, -10)[-10:]
top_indices = top_indices[np.argsort(neg_sims[top_indices])[::-1]]
top_pairs = []
for rank, idx in enumerate(top_indices, 1):
    i, j = int(triu_idx[0][idx]), int(triu_idx[1][idx])
    top_pairs.append({
        "rank": rank,
        "sim": neg_sims[idx],
        "img_a": originals[i],
        "img_b": originals[j],
    })

# ── Layout (matching datasets4 style) ──────────────────────────
DPI = 150
ROW_H = 2.3          # height per row
IMG_GAP = 0.08       # gap between two images in a row
LABEL_W = 0.7        # width for rank label
PAD_LEFT = 0.5
PAD_RIGHT = 0.3
PAD_TOP = 0.55
PAD_BOT = 0.2

N = 10
fig_width = 9.0
img_w = (fig_width - PAD_LEFT - PAD_RIGHT - LABEL_W - IMG_GAP) / 2
fig_height = PAD_TOP + PAD_BOT + N * ROW_H

print(f"Figure: {fig_width:.1f} x {fig_height:.1f} inches, img_w={img_w:.2f}")

fig = plt.figure(figsize=(fig_width, fig_height), dpi=DPI, facecolor="white")

for row_idx, pair in enumerate(top_pairs):
    img_a = Image.open(pair["img_a"]).convert("RGB")
    img_b = Image.open(pair["img_b"]).convert("RGB")

    target_h = int(ROW_H * DPI * 0.80)
    ar_a = img_a.width / img_a.height
    ar_b = img_b.width / img_b.height
    img_a_resized = img_a.resize((int(target_h * ar_a), target_h))
    img_b_resized = img_b.resize((int(target_h * ar_b), target_h))

    y_top = PAD_TOP + row_idx * ROW_H
    y_bottom = y_top + ROW_H

    # Rank label
    fig.text(PAD_LEFT / fig_width,
             1 - (y_top + ROW_H / 2) / fig_height,
             f"#{pair['rank']}\nsim={pair['sim']:.3f}",
             ha="left", va="center", fontsize=8.5, fontweight="bold",
             color="#C0392B", family="monospace")

    # Image A
    ax_a_left = PAD_LEFT + LABEL_W
    ax_a = fig.add_axes([ax_a_left / fig_width,
                         1 - y_bottom / fig_height,
                         img_w / fig_width,
                         ROW_H / fig_height])
    ax_a.imshow(img_a_resized)
    ax_a.axis("off")

    # Image B
    ax_b_left = ax_a_left + img_w + IMG_GAP
    ax_b = fig.add_axes([ax_b_left / fig_width,
                         1 - y_bottom / fig_height,
                         img_w / fig_width,
                         ROW_H / fig_height])
    ax_b.imshow(img_b_resized)
    ax_b.axis("off")

    # Filename labels
    name_a = f"{pair['img_a'].parent.name}/{pair['img_a'].name}"
    name_b = f"{pair['img_b'].parent.name}/{pair['img_b'].name}"
    max_len = 42
    if len(name_a) > max_len:
        name_a = name_a[:max_len-3] + "..."
    if len(name_b) > max_len:
        name_b = name_b[:max_len-3] + "..."

    fig.text((ax_a_left + img_w / 2) / fig_width,
             1 - (y_bottom + 0.01) / fig_height,
             name_a, ha="center", va="top", fontsize=5, color="#666666")
    fig.text((ax_b_left + img_w / 2) / fig_width,
             1 - (y_bottom + 0.01) / fig_height,
             name_b, ha="center", va="top", fontsize=5, color="#666666")

# Title
fig.text(0.5, 1 - 0.12 / fig_height,
         "Top 10 ViT Negative Pairs — Datasets5 (ArtBench-10)",
         ha="center", va="center", fontsize=13, fontweight="bold")

# Subtitle
fig.text(0.5, 1 - 0.32 / fig_height,
         "Highest ViT cosine similarity between different original images | 8/10 are within renaissance, 2 cross-style",
         ha="center", va="center", fontsize=7.5, color="#888888", style="italic")

out_path = OUT_DIR / "top10_vit_negative_pairs_datasets5.png"
fig.savefig(out_path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close(fig)
print(f"Saved: {out_path}")
