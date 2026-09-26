#!/usr/bin/env python3
"""
Threshold Evaluation: CLIP + ViT two-stage pipeline on 1000 images (Datasets2_1).
Output format matches threshold_eval_clip08_vit05_archieve.

- 1000 originals (100/class × 10 classes) from Datasets2_1
- 31 transforms per original → 31,000 positive pairs (CLIP)
- C(1000,2) = 499,500 negative pairs (ViT)
- Grid: CLIP [0.80..0.85] × ViT [0.30..0.50] = 126 combinations
"""

from __future__ import annotations

import json
import time
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets2_1"
OUT_DIR = BASE_DIR / "analysis" / "threshold_eval_clip08-vit05_datasets2_1"
TRANSFORM_DIR = OUT_DIR / "transforms"

CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 64  # larger batch for MPS efficiency

CLASSES = ["airplane", "bird", "car", "cat", "chair", "dog", "fish", "flower", "house", "tree"]
IMAGES_PER_CLASS = 100  # full dataset

# Threshold grid matching archive format
CLIP_THRESHOLDS = [round(x/100, 2) for x in range(80, 86)]   # 0.80 .. 0.85
VIT_THRESHOLDS = [round(x/100, 2) for x in range(30, 51)]     # 0.30 .. 0.50

print(f"CLIP thresholds ({len(CLIP_THRESHOLDS)}): {CLIP_THRESHOLDS}")
print(f"ViT thresholds  ({len(VIT_THRESHOLDS)}): {VIT_THRESHOLDS}")
print(f"Grid: {len(CLIP_THRESHOLDS)} × {len(VIT_THRESHOLDS)} = {len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)}")
print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}")

# ── 12 Transform operations ──────────────────────────────────────

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

# ═══════════════════════════════════════════════════════════════════
# Phase 1: Gather originals + generate transforms
# ═══════════════════════════════════════════════════════════════════

def gather_originals() -> list[dict]:
    images = []
    for cls in CLASSES:
        cls_dir = DATASETS_DIR / cls
        if not cls_dir.exists():
            print(f"  WARNING: {cls_dir} not found, skipping")
            continue
        for fpath in sorted(cls_dir.glob("*.jpg"))[:IMAGES_PER_CLASS]:
            images.append({"path": fpath, "class": cls, "stem": fpath.stem})
    return images


def generate_transforms(originals: list[dict]) -> dict[str, Path]:
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    dirs = {}
    total = len(originals)
    print(f"  Generating {total} × 32 = {total*32} transforms...")
    for i, info in enumerate(tqdm(originals, desc="  Transforms")):
        img = cv2.imread(str(info["path"]))
        if img is None:
            continue
        out_dir = TRANSFORM_DIR / info["stem"]
        # Skip if already done
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == 32:
            dirs[info["stem"]] = out_dir
            continue
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


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Build pairs
# ═══════════════════════════════════════════════════════════════════

def build_pairs(originals: list[dict]) -> tuple[list[dict], list[dict]]:
    # Positive pairs: each original vs its 31 non-F2 variants
    positive_pairs = []
    for info in originals:
        vdir = TRANSFORM_DIR / info["stem"]
        for vp in sorted(vdir.glob("*.jpg")):
            if vp.stem.startswith("F2"):
                continue
            positive_pairs.append({
                "path_A": str(info["path"]), "path_B": str(vp),
                "name_A": info["stem"], "name_B": f"{info['stem']}/{vp.name}",
                "label": 1,
            })

    # Negative pairs: all C(N,2) different-original pairs
    negative_pairs = []
    N = len(originals)
    for i in tqdm(range(N), desc="  Building negative pairs"):
        for j in range(i + 1, N):
            negative_pairs.append({
                "path_A": str(originals[i]["path"]), "path_B": str(originals[j]["path"]),
                "name_A": originals[i]["stem"], "name_B": originals[j]["stem"],
                "label": 0,
            })
    return positive_pairs, negative_pairs


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Compute similarities
# ═══════════════════════════════════════════════════════════════════

def load_models():
    clip_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
    vit_path = str(VIT_MODEL_DIR) if VIT_MODEL_DIR.exists() else "google/vit-base-patch16-224"

    print(f"  Loading CLIP from: {clip_path}")
    clip_proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=CLIP_MODEL_DIR.exists())
    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=CLIP_MODEL_DIR.exists()).to(DEVICE).eval()

    print(f"  Loading ViT from: {vit_path}")
    vit_proc = ViTImageProcessor.from_pretrained(vit_path, local_files_only=VIT_MODEL_DIR.exists())
    vit_model = ViTModel.from_pretrained(vit_path, local_files_only=VIT_MODEL_DIR.exists()).to(DEVICE).eval()

    return (clip_proc, clip_model), (vit_proc, vit_model)


@torch.no_grad()
def compute_clip_sim(paths_A, paths_B, proc, model) -> np.ndarray:
    sims = []
    n = len(paths_A)
    for i in tqdm(range(0, n, BATCH_SIZE), desc="  CLIP inference"):
        batch_end = min(i + BATCH_SIZE, n)
        ba = [Image.open(p).convert("RGB") for p in paths_A[i:batch_end]]
        bb = [Image.open(p).convert("RGB") for p in paths_B[i:batch_end]]
        ia = proc(images=ba, return_tensors="pt", padding=True)
        ib = proc(images=bb, return_tensors="pt", padding=True)
        ia = {k: v.to(DEVICE) for k, v in ia.items()}
        ib = {k: v.to(DEVICE) for k, v in ib.items()}
        fa = model.get_image_features(**ia)
        fb = model.get_image_features(**ib)
        if hasattr(fa, "pooler_output"): fa = fa.pooler_output
        if hasattr(fb, "pooler_output"): fb = fb.pooler_output
        if isinstance(fa, tuple): fa = fa[0]
        if isinstance(fb, tuple): fb = fb[0]
        fa = fa / fa.norm(dim=-1, keepdim=True)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        sims.append((fa * fb).sum(dim=-1).cpu().numpy())
    return np.concatenate(sims)


@torch.no_grad()
def compute_vit_sim(paths_A, paths_B, proc, model) -> np.ndarray:
    sims = []
    n = len(paths_A)
    for i in tqdm(range(0, n, BATCH_SIZE), desc="  ViT inference"):
        batch_end = min(i + BATCH_SIZE, n)
        ba = [Image.open(p).convert("RGB") for p in paths_A[i:batch_end]]
        bb = [Image.open(p).convert("RGB") for p in paths_B[i:batch_end]]
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


# ═══════════════════════════════════════════════════════════════════
# Phase 4: Threshold grid evaluation (two-stage pipeline)
# ═══════════════════════════════════════════════════════════════════

def evaluate_grid(pos_sims, neg_sims):
    """
    Two-stage pipeline:
      Stage 1 (CLIP): For positive pairs, sim >= ct → TP, sim < ct → FN
      Stage 2 (ViT):   For negative pairs, sim < vt → TN, sim >= vt → FP
    Overall metrics combine both stages.
    """
    n_pos = len(pos_sims)
    n_neg = len(neg_sims)
    total = n_pos + n_neg

    rows = []
    for ct in CLIP_THRESHOLDS:
        # CLIP on positives
        TP = int(np.sum(pos_sims >= ct))
        FN = int(np.sum(pos_sims < ct))
        for vt in VIT_THRESHOLDS:
            # ViT on negatives
            TN = int(np.sum(neg_sims < vt))
            FP = int(np.sum(neg_sims >= vt))

            overall_TP = TP
            overall_TN = TN
            overall_FP = FP
            overall_FN = FN

            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            accuracy = (TP + TN) / total if total > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            rows.append({
                "clip_threshold": ct, "vit_threshold": vt,
                "clip_TP": TP, "clip_FN": FN,
                "vit_TN": TN, "vit_FP": FP,
                "overall_TP": overall_TP, "overall_TN": overall_TN,
                "overall_FP": overall_FP, "overall_FN": overall_FN,
                "accuracy": round(accuracy, 4), "recall": round(recall, 4),
                "precision": round(precision, 4), "f1": round(f1, 4),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Phase 5: Generate Excel output matching archive format
# ═══════════════════════════════════════════════════════════════════

def generate_excel(grid_df: pd.DataFrame, pos_pairs, neg_pairs,
                   clip_pos_sims, vit_neg_sims, timing: dict):
    """Generate Excel with 6 sheets matching the archive format."""
    try:
        import openpyxl
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.chart import LineChart, Reference
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  openpyxl not available, skipping Excel generation")
        return

    print("\n  Generating Excel output...")
    wb = openpyxl.Workbook()

    # ── Helper styles ──────────────────────────────────────────
    header_font = Font(bold=True, size=11)
    title_font = Font(bold=True, size=14)
    section_font = Font(bold=True, size=12)
    note_font = Font(italic=True, size=9, color="888888")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    # ── Sheet 00: overview ─────────────────────────────────────
    ws_ov = wb.active
    ws_ov.title = "00_overview"

    best_f1 = grid_df.loc[grid_df["f1"].idxmax()]
    best_f1_c = float(best_f1["clip_threshold"]); best_f1_v = float(best_f1["vit_threshold"])
    lowest_fp = grid_df.loc[grid_df["overall_FP"].idxmin()]
    row = 1
    ws_ov.cell(row=row, column=1, value="Threshold Experiment Overview").font = title_font; row += 1
    ws_ov.cell(row=row, column=1, value="model").font = header_font
    ws_ov.cell(row=row, column=2, value="threshold").font = header_font
    ws_ov.cell(row=row, column=5, value="Best F1").font = header_font
    ws_ov.cell(row=row, column=7, value="Lowest FP").font = header_font
    ws_ov.cell(row=row, column=6, value=f"CLIP={best_f1_c}, ViT={best_f1_v}")
    ws_ov.cell(row=row, column=8, value=f"CLIP={lowest_fp['clip_threshold']}, ViT={lowest_fp['vit_threshold']}")
    row += 1
    ws_ov.cell(row=row, column=1, value="Overall (CLIP + ViT)").font = header_font
    ws_ov.cell(row=row, column=2, value=f"{best_f1_c}/{best_f1_v}")
    ws_ov.cell(row=row, column=6, value=round(float(best_f1["f1"]), 4))
    ws_ov.cell(row=row, column=8, value=int(lowest_fp["overall_FP"]))
    row += 2

    ws_ov.cell(row=row, column=1, value="Baseline overall summary").font = section_font; row += 2
    for metric, val in [("TP", int(best_f1["overall_TP"])), ("TN", int(best_f1["overall_TN"])),
                         ("FP", int(best_f1["overall_FP"])), ("FN", int(best_f1["overall_FN"])),
                         ("accuracy", round(float(best_f1["accuracy"]), 4)),
                         ("recall", round(float(best_f1["recall"]), 4)),
                         ("precision", round(float(best_f1["precision"]), 4)),
                         ("f1", round(float(best_f1["f1"]), 4))]:
        ws_ov.cell(row=row, column=1, value=metric); ws_ov.cell(row=row, column=2, value=val); row += 1
    row += 1

    # Best threshold ranking
    ws_ov.cell(row=row, column=1, value="Best threshold combinations").font = section_font; row += 1
    headers = ["ranking", "clip_threshold", "vit_threshold", "clip_TP", "clip_FN",
               "vit_TN", "vit_FP", "overall_TP", "overall_TN", "overall_FP", "overall_FN",
               "accuracy", "recall", "precision", "f1"]
    for j, h in enumerate(headers):
        ws_ov.cell(row=row, column=j+1, value=h).font = header_font
    row += 1

    # Top 5 by F1
    top_f1 = grid_df.nlargest(5, "f1")
    for i, (_, r) in enumerate(top_f1.iterrows()):
        ws_ov.cell(row=row, column=1, value=f"top_f1_{i+1}")
        for j, h in enumerate(headers[1:], 2):
            ws_ov.cell(row=row, column=j, value=r[h] if not isinstance(r[h], float) else round(r[h], 4))
        row += 1

    # Top 5 by lowest FP
    low_fp = grid_df.nsmallest(5, "overall_FP")
    for i, (_, r) in enumerate(low_fp.iterrows()):
        ws_ov.cell(row=row, column=1, value=f"low_fp_{i+1}")
        for j, h in enumerate(headers[1:], 2):
            ws_ov.cell(row=row, column=j, value=r[h] if not isinstance(r[h], float) else round(r[h], 4))
        row += 1

    row += 1
    ws_ov.cell(row=row, column=1, value="Sheet guide").font = section_font; row += 1
    ws_ov.cell(row=row, column=1, value=f"01_vit_sweep: Single-variable ViT sweep with fixed CLIP={CLIP_THRESHOLDS[0]} (ViT {VIT_THRESHOLDS[0]}–{VIT_THRESHOLDS[-1]})")
    row += 1
    ws_ov.cell(row=row, column=1, value=f"02_clip_vit_grid: {len(grid_df)} CLIP × ViT raw grid results (ViT {VIT_THRESHOLDS[0]}–{VIT_THRESHOLDS[-1]})")
    row += 1
    ws_ov.cell(row=row, column=1, value="03_threshold_curves: Accuracy / Precision / F1 / FP curve chart data")
    row += 1
    ws_ov.cell(row=row, column=1, value="04_f1_heatmap: F1 score heatmap")
    row += 1
    ws_ov.cell(row=row, column=1, value="05_fp_heatmap: Overall false positives heatmap")

    # ── Sheet 01: vit_sweep (fixed CLIP=0.80) ──────────────────
    ws_vs = wb.create_sheet("01_vit_sweep")
    clip_fixed = CLIP_THRESHOLDS[0]
    sweep_df = grid_df[grid_df["clip_threshold"] == clip_fixed].copy()
    grid_headers = ["clip_threshold", "vit_threshold", "clip_TP", "clip_FN",
                    "vit_TN", "vit_FP", "overall_TP", "overall_TN", "overall_FP", "overall_FN",
                    "accuracy", "recall", "precision", "f1"]
    for j, h in enumerate(grid_headers):
        ws_vs.cell(row=1, column=j+1, value=h).font = header_font
    for i, (_, r) in enumerate(sweep_df.iterrows()):
        for j, h in enumerate(grid_headers):
            ws_vs.cell(row=i+2, column=j+1, value=r[h] if not isinstance(r[h], float) else round(r[h], 4))

    # ── Sheet 02: clip_vit_grid (all 126 combinations) ──────────
    ws_grid = wb.create_sheet("02_clip_vit_grid")
    for j, h in enumerate(grid_headers):
        ws_grid.cell(row=1, column=j+1, value=h).font = header_font
    for i, (_, r) in enumerate(grid_df.iterrows()):
        for j, h in enumerate(grid_headers):
            ws_grid.cell(row=i+2, column=j+1, value=r[h] if not isinstance(r[h], float) else round(r[h], 4))

    # ── Sheet 03: threshold_curves (data + charts) ──────────────
    ws_tc = wb.create_sheet("03_threshold_curves")
    clip_labels = [f"CLIP={t:.2f}" for t in CLIP_THRESHOLDS]
    vit_labels = VIT_THRESHOLDS

    metrics = [
        ("F1 Score", "f1"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("False Positives (FP)", "overall_FP"),
    ]

    ws_tc.cell(row=1, column=1, value="Threshold Curves - Chart Data").font = title_font
    ws_tc.cell(row=2, column=1, value=f"Each line corresponds to one CLIP threshold; X-axis is ViT threshold ({VIT_THRESHOLDS[0]}–{VIT_THRESHOLDS[-1]}).").font = note_font

    current_row = 4
    chart_colors = ["1F4E79", "2E75B6", "5B9BD5", "9DC3E6", "BDD7EE", "A9CCE3"]

    for metric_name, metric_col in metrics:
        ws_tc.cell(row=current_row, column=1, value=metric_name).font = section_font
        current_row += 1
        # Header
        ws_tc.cell(row=current_row, column=1, value="ViT Threshold").font = header_font
        for j, cl in enumerate(clip_labels):
            ws_tc.cell(row=current_row, column=2+j, value=cl).font = header_font
        current_row += 1
        # Data
        for vt in vit_labels:
            ws_tc.cell(row=current_row, column=1, value=vt)
            for j, ct in enumerate(CLIP_THRESHOLDS):
                match = grid_df[(grid_df["clip_threshold"] == ct) & (grid_df["vit_threshold"] == vt)]
                if len(match) > 0:
                    val = match.iloc[0][metric_col]
                    ws_tc.cell(row=current_row, column=2+j, value=round(float(val), 4) if metric_col != "overall_FP" else int(val))
            current_row += 1
        current_row += 1

    # Charts
    chart_sections = [
        ("F1 Score", 5, 26, 0.0, 1.0, "0.0000"),
        ("Accuracy", 29, 50, 0.0, 1.0, "0.0000"),
        ("Precision", 53, 74, 0.0, 1.0, "0.0000"),
        ("False Positives (FP)", 77, 98, None, None, "#,##0"),
    ]

    for sec_name, data_start, data_end, y_min, y_max, num_fmt in chart_sections:
        # Adjust data references
        pass  # Chart logic is complex, we create charts separately

    # Create charts using the data positions above
    chart_specs = [
        ("F1 Score", 5, 26),
        ("Accuracy", 29, 50),
        ("Precision", 53, 74),
        ("False Positives (FP)", 77, 98),
    ]
    for (sec_name, hdr_row, data_end) in chart_specs:
        chart = LineChart()
        chart.title = f"ViT Threshold vs {sec_name}"
        chart.style = 2
        chart.width = 20; chart.height = 12

        chart.y_axis.title = sec_name
        if sec_name == "False Positives (FP)":
            chart.y_axis.numFmt = '#,##0'
        else:
            chart.y_axis.numFmt = '0.0000'
            chart.y_axis.scaling.min = 0.0
            chart.y_axis.scaling.max = 1.0

        chart.x_axis.title = "ViT Threshold"
        chart.x_axis.numFmt = '0.00'

        for j in range(6):
            vals = Reference(ws_tc, min_col=2+j, min_row=hdr_row, max_row=data_end)
            chart.add_data(vals, titles_from_data=True)
            chart.series[j].graphicalProperties.line.width = 21000
            chart.series[j].graphicalProperties.line.solidFill = chart_colors[j]
            chart.series[j].smooth = False

        cats = Reference(ws_tc, min_col=1, min_row=hdr_row+1, max_row=data_end)
        chart.set_categories(cats)
        chart.legend.position = 'b'
        anchor_row = data_end + 2
        ws_tc.add_chart(chart, f"A{anchor_row}")

    # Add note
    note_row = 100
    ws_tc.cell(row=note_row, column=1, value="Note on chart interpretation:").font = Font(bold=True, size=10, color="333333")
    ws_tc.cell(row=note_row+1, column=1, value="FP chart: All 6 CLIP lines overlap exactly because ViT FP is independent of CLIP threshold — CLIP only affects FN, not FP.").font = note_font
    ws_tc.cell(row=note_row+2, column=1, value="F1 / Accuracy / Precision charts: Lines are very close because overall metrics are dominated by ViT threshold; CLIP FN differences have minimal impact.").font = note_font

    # ── Sheet 04: f1_heatmap ────────────────────────────────────
    ws_f1 = wb.create_sheet("04_f1_heatmap")
    ws_f1.cell(row=1, column=1, value="F1 Heatmap").font = title_font
    ws_f1.cell(row=2, column=1, value="Rows represent the ViT threshold and columns represent the CLIP threshold; the greener the color, the higher the F1 score.").font = note_font
    ws_f1.cell(row=4, column=1, value="clip_threshold").font = header_font
    for j, ct in enumerate(CLIP_THRESHOLDS):
        ws_f1.cell(row=4, column=2+j, value=ct)
    pivot_f1 = grid_df.pivot_table(values="f1", index="vit_threshold", columns="clip_threshold")
    for i, vt in enumerate(vit_labels):
        ws_f1.cell(row=5+i, column=1, value=vt)
        for j, ct in enumerate(CLIP_THRESHOLDS):
            val = pivot_f1.loc[vt, ct] if vt in pivot_f1.index and ct in pivot_f1.columns else None
            if val is not None:
                ws_f1.cell(row=5+i, column=2+j, value=round(float(val), 4))

    # ── Sheet 05: fp_heatmap ────────────────────────────────────
    ws_fp = wb.create_sheet("05_fp_heatmap")
    ws_fp.cell(row=1, column=1, value="Overall FP Heatmap").font = title_font
    ws_fp.cell(row=2, column=1, value="Rows represent the CLIP threshold and columns represent the ViT threshold; the greener the color, the fewer the false positives.").font = note_font
    ws_fp.cell(row=4, column=1, value="clip_threshold").font = header_font
    for j, vt in enumerate(vit_labels):
        ws_fp.cell(row=4, column=2+j, value=vt)
    pivot_fp = grid_df.pivot_table(values="overall_FP", index="clip_threshold", columns="vit_threshold")
    for i, ct in enumerate(CLIP_THRESHOLDS):
        ws_fp.cell(row=5+i, column=1, value=ct)
        for j, vt in enumerate(vit_labels):
            val = pivot_fp.loc[ct, vt] if ct in pivot_fp.index and vt in pivot_fp.columns else None
            if val is not None:
                ws_fp.cell(row=5+i, column=2+j, value=int(val))

    # Save
    excel_path = OUT_DIR / "overall_summary_excel_chart_datasets2_1.xlsx"
    wb.save(excel_path)
    print(f"  Excel saved to: {excel_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Threshold Evaluation: CLIP + ViT — Datasets2_1 (1000 images)")
    print(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE}")
    print(f"Grid: {len(CLIP_THRESHOLDS)} CLIP × {len(VIT_THRESHOLDS)} ViT = {len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)}")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    transform_cache = OUT_DIR / "transforms"
    sims_cache = OUT_DIR / "similarities.npz"

    # 1. Gather originals
    originals = gather_originals()
    print(f"\n[1/5] {len(originals)} originals from Datasets2_1")
    print(f"       Classes: {len(CLASSES)} × {IMAGES_PER_CLASS}")

    # 2. Generate transforms
    print(f"\n[2/5] Generating transforms...")
    t0 = time.time()
    generate_transforms(originals)
    t_trans = time.time() - t0
    print(f"  Done in {t_trans:.1f}s")

    # 3. Build pairs
    print(f"\n[3/5] Building evaluation pairs...")
    pos_pairs, neg_pairs = build_pairs(originals)
    print(f"  Positive pairs: {len(pos_pairs)}")
    print(f"  Negative pairs: {len(neg_pairs)}")
    print(f"  Total: {len(pos_pairs) + len(neg_pairs)}")

    t_clip = 0.0
    t_vit = 0.0

    # 4. Compute similarities (or load from cache)
    if sims_cache.exists():
        print(f"\n[4/5] Loading cached similarities from {sims_cache}...")
        data = np.load(sims_cache)
        clip_pos_sims = data["clip_pos_sims"]
        vit_neg_sims = data["vit_neg_sims"]
    else:
        print(f"\n[4/5] Loading models and computing similarities...")
        (clip_proc, clip_model), (vit_proc, vit_model) = load_models()

        print(f"\n  --- CLIP on Positive Pairs ({len(pos_pairs)}) ---")
        pos_paths_A = [p["path_A"] for p in pos_pairs]
        pos_paths_B = [p["path_B"] for p in pos_pairs]
        t1 = time.time()
        clip_pos_sims = compute_clip_sim(pos_paths_A, pos_paths_B, clip_proc, clip_model)
        t_clip = time.time() - t1
        print(f"  CLIP done: {t_clip:.1f}s ({len(pos_pairs)/t_clip:.0f} pairs/s)")

        print(f"\n  --- ViT on Negative Pairs ({len(neg_pairs)}) ---")
        neg_paths_A = [p["path_A"] for p in neg_pairs]
        neg_paths_B = [p["path_B"] for p in neg_pairs]
        t2 = time.time()
        vit_neg_sims = compute_vit_sim(neg_paths_A, neg_paths_B, vit_proc, vit_model)
        t_vit = time.time() - t2
        print(f"  ViT done: {t_vit:.1f}s ({len(neg_pairs)/t_vit:.0f} pairs/s)")

        # Cache similarities
        np.savez_compressed(sims_cache, clip_pos_sims=clip_pos_sims, vit_neg_sims=vit_neg_sims)
        print(f"  Similarities cached to {sims_cache}")

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

    # Best results
    best_f1 = grid_df.loc[grid_df["f1"].idxmax()]
    best_acc = grid_df.loc[grid_df["accuracy"].idxmax()]
    print(f"\n{'='*65}")
    print("Best Results")
    print(f"{'='*65}")
    print(f"  Best F1:  CLIP={best_f1['clip_threshold']:.2f}, ViT={best_f1['vit_threshold']:.2f}  "
          f"→ F1={best_f1['f1']:.4f}, Acc={best_f1['accuracy']:.4f}, "
          f"Recall={best_f1['recall']:.4f}, Prec={best_f1['precision']:.4f}")
    print(f"  Best Acc: CLIP={best_acc['clip_threshold']:.2f}, ViT={best_acc['vit_threshold']:.2f}  "
          f"→ Acc={best_acc['accuracy']:.4f}, F1={best_acc['f1']:.4f}")

    print(f"\n  CLIP positive sims: mean={np.mean(clip_pos_sims):.4f} std={np.std(clip_pos_sims):.4f} "
          f"min={np.min(clip_pos_sims):.4f} max={np.max(clip_pos_sims):.4f}")
    print(f"  ViT negative sims:   mean={np.mean(vit_neg_sims):.4f} std={np.std(vit_neg_sims):.4f} "
          f"min={np.min(vit_neg_sims):.4f} max={np.max(vit_neg_sims):.4f}")

    # Timing
    total_time = t_trans + t_clip + t_vit
    timing = {
        "num_originals": len(originals),
        "positive_pairs": len(pos_pairs),
        "negative_pairs": len(neg_pairs),
        "total_pairs": len(pos_pairs) + len(neg_pairs),
        "clip_thresholds": CLIP_THRESHOLDS,
        "vit_thresholds": VIT_THRESHOLDS,
        "grid_combinations": len(CLIP_THRESHOLDS) * len(VIT_THRESHOLDS),
        "transform_time_sec": round(t_trans, 1),
        "clip_time_sec": round(t_clip, 1),
        "vit_time_sec": round(t_vit, 1),
        "total_time_sec": round(total_time, 1),
        "best_f1": {
            "clip_threshold": float(best_f1["clip_threshold"]),
            "vit_threshold": float(best_f1["vit_threshold"]),
            "f1": float(best_f1["f1"]),
            "accuracy": float(best_f1["accuracy"]),
            "fp": int(best_f1["overall_FP"]),
        },
    }
    with open(OUT_DIR / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # Generate Excel
    generate_excel(grid_df, pos_pairs, neg_pairs, clip_pos_sims, vit_neg_sims, timing)

    print(f"\n{'='*65}")
    print(f"Done! Results saved to: {OUT_DIR}")
    print(f"  positive_pairs.csv       — {len(pos_pairs)} rows")
    print(f"  negative_pairs.csv       — {len(neg_pairs)} rows")
    print(f"  threshold_grid.csv       — {len(grid_df)} threshold combinations")
    print(f"  overall_summary_excel_chart_datasets2_1.xlsx")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
