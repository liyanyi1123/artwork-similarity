#!/usr/bin/env python3
"""
Threshold Sweep: CLIP (positive) + ViT (negative) with 5×5 threshold grid.

- 100 originals from Datasets2_1 (10/class)
- 32 transforms per original, 31 non-F2 variants used for positive pairs
- 3100 positive pairs (CLIP) + 4950 negative pairs (ViT)
- 25 threshold combinations evaluated
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

# ── Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "Datasets2_1"
OUT_DIR = BASE_DIR / "experiments" / "threshold_sweep"
TRANSFORM_DIR = OUT_DIR / "transforms"

CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32

CLASSES = ["airplane", "bird", "car", "cat", "chair", "dog", "fish", "flower", "house", "tree"]
IMAGES_PER_CLASS = 10  # 10 × 10 = 100 originals

# Threshold grid
CLIP_THRESHOLDS = [0.75, 0.80, 0.85, 0.90, 0.95]
VIT_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

# ── 12 Transform operations (same as ex3) ──────────────

def op_F1(img): return cv2.flip(img, 1)
def op_F2(img): return cv2.flip(img, 0)

def op_C(img, s=1.0):
    h, w = img.shape[:2]; r = min(0.05*s, 0.95); cr = 1.0 - r
    nh, nw = int(h*cr), int(w*cr); y0, x0 = (h-nh)//2, (w-nw)//2
    return cv2.resize(img[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_LANCZOS4)

def op_S1(img, s=1.0):
    h, w = img.shape[:2]; M = np.float32([[1,0,int(0.05*w*s)],[0,1,0]])
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)

def op_S2(img, s=1.0):
    h, w = img.shape[:2]; M = np.float32([[1,0,0],[0,1,int(0.05*h*s)]])
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)

def op_R(img, s=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2), 2.0*s, 1.0)
    return cv2.warpAffine(img, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)

def op_B(img, s=1.0):
    return cv2.convertScaleAbs(img, alpha=1.0+0.04*s, beta=8.0*s)

def op_V(img, s=1.0):
    h, w = img.shape[:2]; y, x = np.ogrid[:h,:w]
    cx, cy = w/2, h/2; dx, dy = (x-cx)/(w/2), (y-cy)/(h/2)
    d = np.sqrt(dx**2+dy**2); red, exp = 0.25*s, 1.5*s
    mask = np.clip(1.0-d*red, 0, 1); mask = np.power(mask, exp)
    return (img.astype(np.float32)*mask[:,:,np.newaxis]).astype(np.uint8)

def op_T(img, s=1.0):
    r = img.astype(np.float32)
    r[:,:,2] = np.clip(r[:,:,2]*(1.0+0.06*s), 0, 255)
    r[:,:,1] = np.clip(r[:,:,1]*(1.0+0.02*s), 0, 255)
    r[:,:,0] = np.clip(r[:,:,0]*(1.0-0.05*s), 0, 255)
    return r.astype(np.uint8)

def op_Sh(img, s=1.0):
    sk = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    ik = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
    w = min(0.4*s, 0.95)
    return cv2.filter2D(img, -1, w*sk + (1-w)*ik)

def op_K(img, s=1.0):
    h, w = img.shape[:2]; m = 0.03*s
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[0,0],[w,h*m],[w,h*(1-m)],[0,h]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src,dst), (w,h),
                               borderMode=cv2.BORDER_REFLECT_101)

def op_N(img, s=1.0):
    rng = np.random.default_rng(42)
    return np.clip(img.astype(np.float32)+rng.normal(0,3.0*s,img.shape),0,255).astype(np.uint8)

OPERATIONS = [
    ("F1", op_F1, None),
    ("F2", op_F2, None),
    ("C",  op_C,  [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("S1", op_S1, [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("S2", op_S2, [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("R",  op_R,  [(1.5,"low"),(3.0,"mid"),(6.0,"high")]),
    ("B",  op_B,  [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("V",  op_V,  [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("T",  op_T,  [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("Sh", op_Sh, [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("K",  op_K,  [(0.33,"low"),(1.0,"mid"),(1.67,"high")]),
    ("N",  op_N,  [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
]


# ═══════════════════════════════════════════════════════
# Phase 1: Gather 100 originals + generate transforms
# ═══════════════════════════════════════════════════════

def gather_originals() -> list[dict]:
    """Pick first 10 images per class from Datasets2_1."""
    images = []
    for cls in CLASSES:
        cls_dir = DATASETS_DIR / cls
        if not cls_dir.exists():
            continue
        for fpath in sorted(cls_dir.glob("*.jpg"))[:IMAGES_PER_CLASS]:
            images.append({"path": fpath, "class": cls, "stem": fpath.stem})
    return images


def generate_transforms(originals: list[dict]) -> dict[str, Path]:
    """Generate 32 variants per original. Returns {stem: transform_dir}."""
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    dirs = {}

    print(f"Generating {len(originals)} × 32 = {len(originals)*32} transforms...")
    for info in tqdm(originals, desc="  Transforms"):
        img = cv2.imread(str(info["path"]))
        if img is None:
            continue
        out_dir = TRANSFORM_DIR / info["stem"]
        out_dir.mkdir(parents=True, exist_ok=True)

        for code, func, settings in OPERATIONS:
            if settings is None:
                result = func(img)
                cv2.imwrite(str(out_dir / f"{code}.jpg"), result)
            else:
                for strength, label in settings:
                    result = func(img, s=strength)
                    cv2.imwrite(str(out_dir / f"{code}_{label}.jpg"), result)
        dirs[info["stem"]] = out_dir

    return dirs


# ═══════════════════════════════════════════════════════
# Phase 2: Build pairs
# ═══════════════════════════════════════════════════════

def build_pairs(originals: list[dict]) -> tuple[list[dict], list[dict]]:
    """Build positive and negative pairs."""
    # Map stem -> original path
    stem_to_path = {info["stem"]: info["path"] for info in originals}

    # Positive pairs: each original vs its 31 non-F2 variants
    positive_pairs = []
    for info in originals:
        vdir = TRANSFORM_DIR / info["stem"]
        for vp in sorted(vdir.glob("*.jpg")):
            if vp.stem.startswith("F2"):  # exclude vertical flip
                continue
            positive_pairs.append({
                "path_A": info["path"], "path_B": vp,
                "name_A": info["stem"], "name_B": f"{info['stem']}/{vp.name}",
                "label": 1,
            })

    # Negative pairs: all C(100,2) different-original pairs
    negative_pairs = []
    for i in range(len(originals)):
        for j in range(i + 1, len(originals)):
            negative_pairs.append({
                "path_A": originals[i]["path"], "path_B": originals[j]["path"],
                "name_A": originals[i]["stem"], "name_B": originals[j]["stem"],
                "label": 0,
            })

    return positive_pairs, negative_pairs


# ═══════════════════════════════════════════════════════
# Phase 3: Compute similarities
# ═══════════════════════════════════════════════════════

def load_models():
    clip_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
    vit_path = str(VIT_MODEL_DIR) if VIT_MODEL_DIR.exists() else "google/vit-base-patch16-224"

    print(f"Loading CLIP from: {clip_path}")
    clip_proc = CLIPProcessor.from_pretrained(clip_path)
    clip_model = CLIPModel.from_pretrained(clip_path).to(DEVICE).eval()

    print(f"Loading ViT from: {vit_path}")
    vit_proc = ViTImageProcessor.from_pretrained(vit_path)
    vit_model = ViTModel.from_pretrained(vit_path).to(DEVICE).eval()

    return (clip_proc, clip_model), (vit_proc, vit_model)


@torch.no_grad()
def compute_clip_sim(paths_A, paths_B, proc, model) -> np.ndarray:
    sims = []
    for i in range(0, len(paths_A), BATCH_SIZE):
        ba = [Image.open(p).convert("RGB") for p in paths_A[i:i+BATCH_SIZE]]
        bb = [Image.open(p).convert("RGB") for p in paths_B[i:i+BATCH_SIZE]]
        ia = proc(images=ba, return_tensors="pt", padding=True)
        ib = proc(images=bb, return_tensors="pt", padding=True)
        ia = {k: v.to(DEVICE) for k, v in ia.items()}
        ib = {k: v.to(DEVICE) for k, v in ib.items()}
        fa = model.get_image_features(**ia)
        fb = model.get_image_features(**ib)
        fa = fa.pooler_output if hasattr(fa, "pooler_output") else fa
        fb = fb.pooler_output if hasattr(fb, "pooler_output") else fb
        if isinstance(fa, tuple): fa = fa[0]
        if isinstance(fb, tuple): fb = fb[0]
        fa = fa / fa.norm(dim=-1, keepdim=True)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        sims.append((fa * fb).sum(dim=-1).cpu().numpy())
    return np.concatenate(sims)


@torch.no_grad()
def compute_vit_sim(paths_A, paths_B, proc, model) -> np.ndarray:
    sims = []
    for i in range(0, len(paths_A), BATCH_SIZE):
        ba = [Image.open(p).convert("RGB") for p in paths_A[i:i+BATCH_SIZE]]
        bb = [Image.open(p).convert("RGB") for p in paths_B[i:i+BATCH_SIZE]]
        ia = proc(images=ba, return_tensors="pt")
        ib = proc(images=bb, return_tensors="pt")
        ia = {k: v.to(DEVICE) for k, v in ia.items()}
        ib = {k: v.to(DEVICE) for k, v in ib.items()}
        ca = model(**ia).last_hidden_state[:, 0, :]
        cb = model(**ib).last_hidden_state[:, 0, :]
        ca = ca / ca.norm(dim=-1, keepdim=True)
        cb = cb / cb.norm(dim=-1, keepdim=True)
        sims.append((ca * cb).sum(dim=-1).cpu().numpy())
    return np.concatenate(sims)


# ═══════════════════════════════════════════════════════
# Phase 4: Threshold grid evaluation
# ═══════════════════════════════════════════════════════

def evaluate_grid(pos_sims, neg_sims):
    """
    pos_sims: CLIP similarities for 3100 positive pairs (label=1)
    neg_sims: ViT similarities for 4950 negative pairs (label=0)
    """
    rows = []
    for ct in CLIP_THRESHOLDS:
        for vt in VIT_THRESHOLDS:
            # CLIP on positives: sim >= ct → predicted 1 (same)
            TP = int(np.sum(pos_sims >= ct))
            FN = int(np.sum(pos_sims < ct))
            # ViT on negatives: sim < vt → predicted 0 (different)
            TN = int(np.sum(neg_sims < vt))
            FP = int(np.sum(neg_sims >= vt))

            total = TP + TN + FP + FN
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            accuracy = (TP + TN) / total if total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            rows.append({
                "clip_threshold": ct, "vit_threshold": vt,
                "TP": TP, "TN": TN, "FP": FP, "FN": FN,
                "recall": round(recall, 4), "precision": round(precision, 4),
                "accuracy": round(accuracy, 4), "f1": round(f1, 4),
            })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════
# Phase 5: Plots
# ═══════════════════════════════════════════════════════

def make_plots(grid_df: pd.DataFrame):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- F1 Heatmap ---
        pivot_f1 = grid_df.pivot_table(values="f1", index="vit_threshold", columns="clip_threshold")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot_f1.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        for i in range(len(pivot_f1.index)):
            for j in range(len(pivot_f1.columns)):
                val = pivot_f1.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white" if val < 0.5 else "black")
        ax.set_xticks(range(len(pivot_f1.columns)))
        ax.set_xticklabels(pivot_f1.columns, fontsize=10)
        ax.set_yticks(range(len(pivot_f1.index)))
        ax.set_yticklabels(pivot_f1.index, fontsize=10)
        ax.set_xlabel("CLIP Threshold (Positive)", fontsize=12)
        ax.set_ylabel("ViT Threshold (Negative)", fontsize=12)
        ax.set_title("F1 Score — CLIP(Pos) × ViT(Neg) Threshold Grid", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="F1 Score")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "threshold_f1_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Accuracy Heatmap ---
        pivot_acc = grid_df.pivot_table(values="accuracy", index="vit_threshold", columns="clip_threshold")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot_acc.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        for i in range(len(pivot_acc.index)):
            for j in range(len(pivot_acc.columns)):
                val = pivot_acc.values[i, j]
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=11,
                        fontweight="bold", color="white" if val < 0.5 else "black")
        ax.set_xticks(range(len(pivot_acc.columns)))
        ax.set_xticklabels(pivot_acc.columns, fontsize=10)
        ax.set_yticks(range(len(pivot_acc.index)))
        ax.set_yticklabels(pivot_acc.index, fontsize=10)
        ax.set_xlabel("CLIP Threshold (Positive)", fontsize=12)
        ax.set_ylabel("ViT Threshold (Negative)", fontsize=12)
        ax.set_title("Accuracy — CLIP(Pos) × ViT(Neg) Threshold Grid", fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Accuracy")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "threshold_accuracy_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # --- Recall vs Precision curve across thresholds ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # For each ViT threshold, plot recall vs CLIP threshold
        ax = axes[0]
        for vt in VIT_THRESHOLDS:
            subset = grid_df[grid_df["vit_threshold"] == vt]
            ax.plot(subset["clip_threshold"], subset["recall"], "o-", label=f"ViT={vt}")
        ax.set_xlabel("CLIP Threshold", fontsize=11)
        ax.set_ylabel("Recall", fontsize=11)
        ax.set_title("Recall vs CLIP Threshold (by ViT threshold)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        for vt in VIT_THRESHOLDS:
            subset = grid_df[grid_df["vit_threshold"] == vt]
            ax.plot(subset["clip_threshold"], subset["precision"], "s-", label=f"ViT={vt}")
        ax.set_xlabel("CLIP Threshold", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title("Precision vs CLIP Threshold (by ViT threshold)", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(OUT_DIR / "threshold_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"  Plots saved to {OUT_DIR}/")
    except ImportError:
        print("  matplotlib not available, skipping plots")


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Threshold Sweep: CLIP(Pos) × ViT(Neg) — 5×5 Grid")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Gather originals
    originals = gather_originals()
    print(f"\n[1/5] {len(originals)} originals from Datasets2_1 ({len(CLASSES)} classes × {IMAGES_PER_CLASS})")

    # 2. Generate transforms
    print(f"\n[2/5] Generating transforms...")
    t0 = time.time()
    generate_transforms(originals)
    print(f"  Done in {time.time()-t0:.1f}s")

    # 3. Build pairs
    print(f"\n[3/5] Building evaluation pairs...")
    pos_pairs, neg_pairs = build_pairs(originals)
    print(f"  Positive pairs: {len(pos_pairs)} (expected 3100)")
    print(f"  Negative pairs: {len(neg_pairs)} (expected 4950)")

    # 4. Compute similarities
    print(f"\n[4/5] Loading models and computing similarities...")
    (clip_proc, clip_model), (vit_proc, vit_model) = load_models()

    print(f"\n  --- CLIP on Positive Pairs ({len(pos_pairs)}) ---")
    pos_A = [p["path_A"] for p in pos_pairs]
    pos_B = [p["path_B"] for p in pos_pairs]
    t1 = time.time()
    clip_pos_sims = compute_clip_sim(pos_A, pos_B, clip_proc, clip_model)
    clip_time = time.time() - t1
    print(f"  CLIP done: {clip_time:.1f}s")

    print(f"\n  --- ViT on Negative Pairs ({len(neg_pairs)}) ---")
    neg_A = [p["path_A"] for p in neg_pairs]
    neg_B = [p["path_B"] for p in neg_pairs]
    t2 = time.time()
    vit_neg_sims = compute_vit_sim(neg_A, neg_B, vit_proc, vit_model)
    vit_time = time.time() - t2
    print(f"  ViT done: {vit_time:.1f}s")

    # 5. Evaluate grid
    print(f"\n[5/5] Evaluating {len(CLIP_THRESHOLDS)}×{len(VIT_THRESHOLDS)} = {len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)} threshold combinations...")
    grid_df = evaluate_grid(clip_pos_sims, vit_neg_sims)

    # Save CSVs
    pd.DataFrame({
        "name_A": [p["name_A"] for p in pos_pairs],
        "name_B": [p["name_B"] for p in pos_pairs],
        "label": 1,
        "clip_similarity": clip_pos_sims,
    }).to_csv(OUT_DIR / "positive_pairs.csv", index=False, float_format="%.6f")

    pd.DataFrame({
        "name_A": [p["name_A"] for p in neg_pairs],
        "name_B": [p["name_B"] for p in neg_pairs],
        "label": 0,
        "vit_similarity": vit_neg_sims,
    }).to_csv(OUT_DIR / "negative_pairs.csv", index=False, float_format="%.6f")

    grid_df.to_csv(OUT_DIR / "threshold_grid.csv", index=False, float_format="%.4f")

    # Print best results
    best_f1 = grid_df.loc[grid_df["f1"].idxmax()]
    best_acc = grid_df.loc[grid_df["accuracy"].idxmax()]
    print(f"\n{'='*65}")
    print("Best Results")
    print(f"{'='*65}")
    print(f"  Best F1:  CLIP={best_f1['clip_threshold']}, ViT={best_f1['vit_threshold']}  "
          f"→ F1={best_f1['f1']:.4f}, Acc={best_f1['accuracy']:.4f}, "
          f"Recall={best_f1['recall']:.4f}, Prec={best_f1['precision']:.4f}")
    print(f"  Best Acc: CLIP={best_acc['clip_threshold']}, ViT={best_acc['vit_threshold']}  "
          f"→ Acc={best_acc['accuracy']:.4f}, F1={best_acc['f1']:.4f}")

    # Distribution stats
    print(f"\n  CLIP positive sims: mean={np.mean(clip_pos_sims):.4f} std={np.std(clip_pos_sims):.4f} "
          f"min={np.min(clip_pos_sims):.4f} max={np.max(clip_pos_sims):.4f}")
    print(f"  ViT negative sims:   mean={np.mean(vit_neg_sims):.4f} std={np.std(vit_neg_sims):.4f} "
          f"min={np.min(vit_neg_sims):.4f} max={np.max(vit_neg_sims):.4f}")

    # Plots
    print(f"\n  Generating plots...")
    make_plots(grid_df)

    # Timing summary
    timing = {
        "num_originals": len(originals),
        "positive_pairs": len(pos_pairs),
        "negative_pairs": len(neg_pairs),
        "clip_thresholds": CLIP_THRESHOLDS,
        "vit_thresholds": VIT_THRESHOLDS,
        "clip_time_sec": round(clip_time, 1),
        "vit_time_sec": round(vit_time, 1),
        "total_time_sec": round(clip_time + vit_time, 1),
        "best_f1": {"clip_threshold": float(best_f1["clip_threshold"]),
                     "vit_threshold": float(best_f1["vit_threshold"]),
                     "f1": float(best_f1["f1"]), "accuracy": float(best_f1["accuracy"])},
    }
    with open(OUT_DIR / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    print(f"\n{'='*65}")
    print(f"Done! Results saved to: {OUT_DIR}")
    print(f"  positive_pairs.csv   — {len(pos_pairs)} rows")
    print(f"  negative_pairs.csv   — {len(neg_pairs)} rows")
    print(f"  threshold_grid.csv   — {len(grid_df)} threshold combinations")
    print(f"  threshold_*.png      — heatmaps & curves")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
