#!/usr/bin/env python3
"""
重新绘制 vit_negative_pairs_line.jpg：对 ViT 负样本余弦相似度取绝对值后排序绘图。
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

# ── 路径 ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "analysis/threshold_eval_clip08-vit05_datasets2_1/embeddings/vit_neg_sims.npy"
OUT_PATH  = BASE_DIR / "analysis/threshold_eval_clip08-vit05_datasets2_1/result_chart/vit_negative_pairs_line.jpg"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── 加载 & 处理 ──────────────────────────────────────────────
sims = np.load(DATA_PATH)                     # (499500,) float32
# 仅对负值取绝对值，正值保持不变
sims_adjusted = np.where(sims < 0, np.abs(sims), sims)
sims_sorted = np.sort(sims_adjusted)          # 升序排列

n_neg = (sims < 0).sum()
print(f"原始相似度: mean={sims.mean():.4f} std={sims.std():.4f} min={sims.min():.4f} max={sims.max():.4f}")
print(f"调整后相似度: mean={sims_adjusted.mean():.4f} std={sims_adjusted.std():.4f} min={sims_adjusted.min():.4f} max={sims_adjusted.max():.4f}")
print(f"负数个数: {n_neg} ({n_neg/len(sims)*100:.1f}%) — 已取绝对值，正值不变")

# ── 绘图 ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})

fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(sims_sorted))               # pair index

ax.plot(x, sims_sorted, linewidth=0.5, color="#2E75B6", alpha=0.9)
ax.set_xlabel("Negative Pair Index (sorted by cosine similarity)")
ax.set_ylabel("ViT Cosine Similarity (negative values → absolute)")
ax.set_title("ViT Cosine Similarity on Negative Pairs (499,500 pairs, negative values taken absolute)")

# 标注均值线
mean_val = sims_adjusted.mean()
ax.axhline(y=mean_val, color="red", linestyle="--", linewidth=1.2,
           label=f"Mean cos sim = {mean_val:.4f}")
ax.legend(loc="upper left")

ax.set_xlim(0, len(sims_sorted))
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(str(OUT_PATH), bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"图表已保存至: {OUT_PATH}")