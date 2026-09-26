#!/usr/bin/env python
"""
方法对比图 v2 —— 补充 PR 曲线与公平对比

改进点：
1. 用二维直方图 + 反向累积和，精确且极快地枚举全部 (t_C, t_V) 组合，
   得到联合规则的 Pareto 前沿，避免逐组合做 530k 次布尔运算。
2. CLIP 单路 / ViT 单路各自独立扫阈值（各自取自己的最优点），
   与论文"每个基线在标定集上自行选阈值"的口径一致 —— 这才是公平对比。
3. 决策空间图：画出全部 FP，仅对背景负样本降采样；图例移出绘图区。
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

W = str(Path(__file__).resolve().parents[2])
OUT = os.path.join(W, "PaperFigures_Sep14", "09_method_comparison")
os.makedirs(OUT, exist_ok=True)

CLIP_THR, VIT_THR = 0.87, 0.50

clip_d = np.load(os.path.join(W, "analysis/threshold_eval_datasets5/embeddings/clip_embeddings.npz"))
vit_d = np.load(os.path.join(W, "analysis/threshold_eval_datasets5/embeddings/vit_embeddings.npz"))
pos_clip = np.abs(clip_d["pos_sims"].astype(np.float64))
orig_clip = clip_d["orig_clip"].astype(np.float64)
orig_vit = vit_d["orig_vit"].astype(np.float64)
pos_vit = np.abs(np.load(os.path.join(W, "exp4/results/positive_vit_similarities.npy")).astype(np.float64))
neg_vit = np.abs(np.load(os.path.join(W, "analysis/threshold_eval_datasets5/embeddings/vit_neg_sims.npy")).astype(np.float64))

orig_clip = orig_clip / np.linalg.norm(orig_clip, axis=1, keepdims=True)
i, j = np.triu_indices(len(orig_clip), k=1)
neg_clip = np.abs(np.sum(orig_clip[i] * orig_clip[j], axis=1))

NP, NN = len(pos_clip), len(neg_clip)
print(f"正样本 {NP:,} / 负样本 {NN:,}")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#5F5E5A", "axes.linewidth": 0.6,
    "axes.labelcolor": "#1F1F1D", "text.color": "#1F1F1D",
    "xtick.color": "#444441", "ytick.color": "#444441",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
C_AND, C_CLIP, C_VIT = "#0C447C", "#EF9F27", "#A32D2D"

# ══════════ 1. 单路阈值扫描 ══════════
grid = np.arange(0.0, 1.0001, 0.002)


def sweep_1d(pos, neg):
    """返回 (thresholds, precision, recall, f1)"""
    P, R = [], []
    for t in grid:
        tp = (pos > t).sum(); fp = (neg > t).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / NP
        P.append(p); R.append(r)
    P, R = np.array(P), np.array(R)
    F1 = np.where(P + R > 0, 2 * P * R / (P + R), 0.0)
    return grid, P, R, F1


g_clip, P_clip, R_clip, F1_clip = sweep_1d(pos_clip, neg_clip)
g_vit, P_vit, R_vit, F1_vit = sweep_1d(pos_vit, neg_vit)

k = int(np.argmax(F1_clip)); print(f"CLIP-only 最优: t={g_clip[k]:.3f}  P={P_clip[k]:.4f} R={R_clip[k]:.4f} F1={F1_clip[k]:.4f}")
k = int(np.argmax(F1_vit));  print(f"ViT-only  最优: t={g_vit[k]:.3f}  P={P_vit[k]:.4f} R={R_vit[k]:.4f} F1={F1_vit[k]:.4f}")

# ══════════ 2. 联合规则：二维直方图 + 反向累积和 ══════════
NB = 500
edges = np.linspace(0.0, 1.0, NB + 1)
Hp, _, _ = np.histogram2d(pos_clip, pos_vit, bins=[edges, edges])
Hn, _, _ = np.histogram2d(neg_clip, neg_vit, bins=[edges, edges])
# 反向累积和：surv[k, l] = 落在 clip bin>=k 且 vit bin>=l 的数量
Sp = Hp[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]
Sn = Hn[::-1, ::-1].cumsum(0).cumsum(1)[::-1, ::-1]

cgrid = edges[:-1]
TP = Sp; FP = Sn
Pm = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
Rm = TP / NP
F1m = np.divide(2 * Pm * Rm, Pm + Rm, out=np.zeros_like(Pm), where=(Pm + Rm) > 0)

best = np.unravel_index(int(np.argmax(F1m)), F1m.shape)
print(f"联合规则最优: t_C={cgrid[best[0]]:.3f} t_V={cgrid[best[1]]:.3f} "
      f"P={Pm[best]:.4f} R={Rm[best]:.4f} F1={F1m[best]:.4f}")

# Pareto 前沿：按 recall 分箱取最大 precision，再做单调上包络
RB = np.linspace(0.0, 1.0, 501)
bidx = np.clip(np.digitize(Rm.ravel(), RB) - 1, 0, 499)
Pf_bin = np.full(500, -1.0)
np.maximum.at(Pf_bin, bidx, Pm.ravel())
Pf_mono = np.maximum.accumulate(Pf_bin[::-1])[::-1]
valid = Pf_bin >= 0
Rf, Pf = RB[:-1][valid], Pf_mono[valid]

# ══════════ 图 C：PR 曲线（放大高精度区）+ 各自最优 F1 ══════════
best_clip = int(np.argmax(F1_clip))
best_vit = int(np.argmax(F1_vit))
best_j = int(np.argmax(F1m.ravel()))
F1j = float(F1m.ravel()[best_j])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.0, 3.9),
                               gridspec_kw={"width_ratios": [1.25, 1]})

axL.plot(R_clip, P_clip, color=C_CLIP, lw=1.5,
         label=f"CLIP only  (best F1 = {F1_clip[best_clip]:.4f})")
axL.plot(R_vit, P_vit, color=C_VIT, lw=1.5,
         label=f"ViT only  (best F1 = {F1_vit[best_vit]:.4f})")
axL.plot(Rf, Pf, color=C_AND, lw=2.2, zorder=4,
         label=f"Joint AND rule  (best F1 = {F1j:.4f})")

axL.plot(R_clip[best_clip], P_clip[best_clip], "o", ms=5.5, color=C_CLIP, zorder=6)
axL.plot(R_vit[best_vit], P_vit[best_vit], "o", ms=5.5, color=C_VIT, zorder=6)
axL.plot(Rm.ravel()[best_j], Pm.ravel()[best_j], "o", ms=6, color=C_AND, zorder=7)

axL.set_xlabel("Recall"); axL.set_ylabel("Precision")
axL.set_xlim(0.974, 1.0015); axL.set_ylim(0.968, 1.0015)
axL.set_title("(a) Precision--recall, high-accuracy region", fontsize=9.5, pad=8)
axL.legend(frameon=False, fontsize=7.6, loc="lower left")
axL.grid(color="#F1EFE8", linewidth=0.5); axL.set_axisbelow(True)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)

labels = ["Joint AND\n(ours)", "CLIP only", "ViT only"]
vals = [F1j, float(F1_clip[best_clip]), float(F1_vit[best_vit])]
colors = [C_AND, C_CLIP, C_VIT]
ypos = np.arange(len(labels))[::-1]
bars = axR.barh(ypos, vals, height=0.52, color=colors, edgecolor="white", linewidth=0.6)
for b, v in zip(bars, vals):
    axR.text(v - 0.004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
             ha="right", va="center", fontsize=8, color="white", fontweight="bold")
axR.set_yticks(ypos); axR.set_yticklabels(labels, fontsize=8.5)
axR.set_xlim(0.90, 1.0005); axR.set_xlabel("Best F1 over each method's own threshold")
axR.set_title("(b) Each method tuned to its own optimum", fontsize=9.5, pad=8)
axR.grid(axis="x", color="#F1EFE8", linewidth=0.5); axR.set_axisbelow(True)
for s in ("top", "right", "left"):
    axR.spines[s].set_visible(False)

fig.tight_layout()
pC = os.path.join(OUT, "figC_precision_recall.png")
fig.savefig(pC, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", pC)

# ══════════ 图 B（修正版）：决策空间 ══════════
rng = np.random.default_rng(42)
fp_mask = (neg_clip > CLIP_THR) & (neg_vit > VIT_THR)
rescued = (neg_clip > CLIP_THR) & (neg_vit <= VIT_THR)   # CLIP 会误收、ViT 条件挡掉
HI = 0.80                                                 # 高 CLIP 相似度阈值
hi_neg = neg_clip > HI
bg_sel = rng.choice(np.flatnonzero(~hi_neg), 18000, replace=False)
pos_sel = rng.choice(NP, 12000, replace=False)

fig, ax = plt.subplots(figsize=(5.8, 4.6))
ax.scatter(neg_clip[bg_sel], neg_vit[bg_sel], s=1.6, c="#D3D1C7", alpha=0.35,
           linewidths=0, label=f"Negative pairs, background (18,000 shown)")
ax.scatter(pos_clip[pos_sel], pos_vit[pos_sel], s=1.6, c="#639922", alpha=0.35,
           linewidths=0, label=f"Positive pairs (12,000 of {NP:,} shown)")
ax.scatter(neg_clip[rescued], neg_vit[rescued], s=15, c="#D85A30", alpha=0.9,
           linewidths=0.3, edgecolors="white",
           label=f"Negatives rejected only by the ViT condition ({int(rescued.sum())})")
ax.scatter(neg_clip[fp_mask], neg_vit[fp_mask], s=17, c="#A32D2D", alpha=0.95,
           linewidths=0.3, edgecolors="white",
           label=f"False positives of the joint rule ({int(fp_mask.sum())})")

ax.add_patch(Rectangle((CLIP_THR, VIT_THR), 1.02 - CLIP_THR, 1.02 - VIT_THR,
                       facecolor="#185FA5", alpha=0.09, edgecolor="#185FA5", lw=1.3, zorder=0))

# 关键对比区 1：CLIP 会误收、但被 ViT 条件挡掉的负样本
n1 = int(rescued.sum())
ax.add_patch(Rectangle((CLIP_THR, -0.02), 1.02 - CLIP_THR, VIT_THR + 0.02,
                       facecolor="#D85A30", alpha=0.07, edgecolor="none", zorder=0))
ax.annotate(f"CLIP alone would accept these\n{n1} negative pairs; the ViT\n"
            f"condition $s_V>0.50$ rejects them",
            xy=(0.905, 0.28), xytext=(0.30, 0.20), fontsize=7.6, color="#993C1D",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#993C1D", lw=0.8))

# 关键对比区 2：ViT 会收、但被 CLIP 条件挡掉的正样本（联合规则的漏检来源）
band2 = (pos_clip <= CLIP_THR) & (pos_vit > VIT_THR)
n2 = int(band2.sum())
ax.add_patch(Rectangle((-0.01, VIT_THR), CLIP_THR + 0.01, 1.02 - VIT_THR,
                       facecolor="#BA7517", alpha=0.06, edgecolor="none", zorder=0))
ax.annotate(f"{n2} positive pairs missed by\nthe joint rule (caught by ViT alone)",
            xy=(0.82, 0.72), xytext=(0.36, 0.86), fontsize=7.6, color="#854F0B",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", color="#854F0B", lw=0.8))

ax.axvline(CLIP_THR, color="#185FA5", lw=1.0, ls="--", zorder=1)
ax.axhline(VIT_THR, color="#BA7517", lw=1.0, ls="--", zorder=1)
ax.text(0.995, 0.985, "$s_C>0.87$ AND $s_V>0.50$", ha="right", va="top",
        fontsize=8.5, color="#0C447C")
ax.text(CLIP_THR - 0.012, 0.62, "CLIP-only\nboundary", ha="right", va="center",
        fontsize=7.5, color="#185FA5")
ax.text(0.015, VIT_THR + 0.018, "ViT-only boundary", ha="left", va="bottom",
        fontsize=7.5, color="#BA7517")

ax.set_xlabel("CLIP absolute cosine similarity $s_C$")
ax.set_ylabel("ViT absolute cosine similarity $s_V$")
ax.set_xlim(-0.01, 1.02); ax.set_ylim(-0.02, 1.02)
ax.legend(frameon=False, fontsize=7.2, loc="upper center",
          bbox_to_anchor=(0.5, -0.14), markerscale=2.2, handletextpad=0.5)
ax.grid(color="#F1EFE8", linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout()
pB = os.path.join(OUT, "figB_decision_space.png")
fig.savefig(pB, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", pB)

# ══════════ 图 A：指标对比 + 误判结构（冻结操作点）══════════
RULES = [
    ("AND\n(ours)", (pos_clip > CLIP_THR) & (pos_vit > VIT_THR),
                    (neg_clip > CLIP_THR) & (neg_vit > VIT_THR), True),
    ("CLIP only",   pos_clip > CLIP_THR,  neg_clip > CLIP_THR,  False),
    ("ViT only",    pos_vit > VIT_THR,    neg_vit > VIT_THR,    False),
    ("OR",          (pos_clip > CLIP_THR) | (pos_vit > VIT_THR),
                    (neg_clip > CLIP_THR) | (neg_vit > VIT_THR), False),
]
res = []
for n, ph, nh, ours in RULES:
    TP, FP, FN = int(ph.sum()), int(nh.sum()), int((~ph).sum())
    P = TP / (TP + FP); R = TP / NP
    res.append((n, dict(precision=P, recall=R, f1=2 * P * R / (P + R), FP=FP, FN=FN), ours))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.5),
                               gridspec_kw={"width_ratios": [1.35, 1]})
names = [n for n, _, _ in res]
x = np.arange(len(names)); bw = 0.26
for k, (key, lab, col) in enumerate([("precision", "Precision", "#185FA5"),
                                     ("recall", "Recall", "#85B7EB"),
                                     ("f1", "F1", "#0C447C")]):
    vals = [m[key] for _, m, _ in res]
    bars = ax1.bar(x + (k - 1) * bw, vals, bw * 0.9, label=lab, color=col,
                   edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=7.2)
ax1.set_xticks(x); ax1.set_xticklabels(names, fontsize=8.5)
ax1.set_ylim(0.60, 1.10); ax1.set_ylabel("Score")
ax1.set_title("(a) Metrics at the frozen operating point", fontsize=9.5, pad=8)
ax1.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.32))
ax1.grid(axis="y", color="#E3E2DD", linewidth=0.5); ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

b1 = ax2.bar(x - bw / 2, [m["FP"] for _, m, _ in res], bw * 1.5,
             label="False positives", color="#A32D2D", edgecolor="white", linewidth=0.5)
b2 = ax2.bar(x + bw / 2, [m["FN"] for _, m, _ in res], bw * 1.5,
             label="False negatives", color="#EF9F27", edgecolor="white", linewidth=0.5)
for bars in (b1, b2):
    for b in bars:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.18,
                 f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=7.2)
ax2.set_yscale("log"); ax2.set_ylim(1, 4e4)
ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=8.5)
ax2.set_ylabel("Error count (log scale)")
ax2.set_title("(b) Error structure", fontsize=9.5, pad=8)
ax2.legend(frameon=False, fontsize=8, loc="upper left")
ax2.grid(axis="y", color="#E3E2DD", linewidth=0.5); ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout()
pA = os.path.join(OUT, "figA_rule_comparison.png")
fig.savefig(pA, dpi=300, bbox_inches="tight"); plt.close(fig)
print("saved", pA)

# ══════════ 汇总数字（供正文引用）══════════
def score(ph, nh):
    TP, FP, FN = int(ph.sum()), int(nh.sum()), int((~ph).sum())
    P = TP / (TP + FP); R = TP / NP
    return dict(TP=TP, FP=FP, FN=FN, P=P, R=R, F1=2 * P * R / (P + R))


print("\n── 同一操作点 (0.87, 0.50) 下的对比 ──")
rows = [
    ("AND (ours)", score((pos_clip > CLIP_THR) & (pos_vit > VIT_THR),
                         (neg_clip > CLIP_THR) & (neg_vit > VIT_THR))),
    ("CLIP only",  score(pos_clip > CLIP_THR, neg_clip > CLIP_THR)),
    ("ViT only",   score(pos_vit > VIT_THR,  neg_vit > VIT_THR)),
    ("OR",         score((pos_clip > CLIP_THR) | (pos_vit > VIT_THR),
                         (neg_clip > CLIP_THR) | (neg_vit > VIT_THR))),
]
for n, r in rows:
    print(f"  {n:11s} P={r['P']:.4f} R={r['R']:.4f} F1={r['F1']:.4f}  FP={r['FP']:>7,} FN={r['FN']:>4,}")

print("\n── 各方法在各自最优阈值下的最佳 F1 ──")
print(f"  CLIP only  best F1 = {F1_clip.max():.4f} @ t={g_clip[int(np.argmax(F1_clip))]:.3f}")
print(f"  ViT only   best F1 = {F1_vit.max():.4f} @ t={g_vit[int(np.argmax(F1_vit))]:.3f}")
print(f"  Joint AND  best F1 = {F1m.max():.4f} @ t_C={cgrid[best[0]]:.3f}, t_V={cgrid[best[1]]:.3f}")
