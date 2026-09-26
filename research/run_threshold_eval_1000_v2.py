#!/usr/bin/env python3
"""
Optimized Threshold Evaluation — pre-compute embeddings, then matrix-multiply.
Datasets2_1: 1000 images → 31,000 pos + 499,500 neg pairs → 126 grid combos.
"""

from __future__ import annotations
import json, time, os, sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets2_1"
OUT_DIR = BASE_DIR / "analysis" / "threshold_eval_clip08-vit05_datasets2_1"
TRANSFORM_DIR = OUT_DIR / "transforms"
EMBEDDING_DIR = OUT_DIR / "embeddings"

CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 128

CLASSES = ["airplane", "bird", "car", "cat", "chair", "dog", "fish", "flower", "house", "tree"]
IMAGES_PER_CLASS = 100

CLIP_THRESHOLDS = [round(x/100, 2) for x in range(80, 86)]
VIT_THRESHOLDS = [round(x/100, 2) for x in range(30, 51)]

print(f"CLIP: {CLIP_THRESHOLDS}  |  ViT: {VIT_THRESHOLDS}")
print(f"Grid: {len(CLIP_THRESHOLDS)}×{len(VIT_THRESHOLDS)}={len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)}")
print(f"Device: {DEVICE}, Batch: {BATCH_SIZE}")

# ── Transforms (same as before) ────────────────────────────────
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
    h, w = img.shape[:2]; M = cv2.getRotationMatrix2D((w/2,h/2), 2.0*s, 1.0)
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
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src,dst), (w,h), borderMode=cv2.BORDER_REFLECT_101)
def op_N(img, s=1.0):
    rng = np.random.default_rng(42)
    return np.clip(img.astype(np.float32)+rng.normal(0,3.0*s,img.shape),0,255).astype(np.uint8)

OPERATIONS = [
    ("F1", op_F1, None), ("F2", op_F2, None),
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

# ═══════════════════════════════════════════════════════════════
# Phase 1: Gather originals + generate transforms
# ═══════════════════════════════════════════════════════════════

def gather_originals():
    images = []
    for cls in CLASSES:
        cls_dir = DATASETS_DIR / cls
        for fpath in sorted(cls_dir.glob("*.jpg"))[:IMAGES_PER_CLASS]:
            images.append({"path": fpath, "class": cls, "stem": fpath.stem})
    return images

def generate_transforms(originals):
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    total = len(originals)
    print(f"  Generating {total}×32={total*32} transforms...")
    for info in tqdm(originals, desc="  Transforms"):
        img = cv2.imread(str(info["path"]))
        if img is None: continue
        out_dir = TRANSFORM_DIR / info["stem"]
        if out_dir.exists() and len(list(out_dir.glob("*.jpg"))) == 32:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for code, func, settings in OPERATIONS:
            if settings is None:
                cv2.imwrite(str(out_dir / f"{code}.jpg"), func(img))
            else:
                for strength, label in settings:
                    cv2.imwrite(str(out_dir / f"{code}_{label}.jpg"), func(img, s=strength))

# ═══════════════════════════════════════════════════════════════
# Phase 2: Pre-compute all embeddings
# ═══════════════════════════════════════════════════════════════

def load_models():
    clip_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
    vit_path = str(VIT_MODEL_DIR) if VIT_MODEL_DIR.exists() else "google/vit-base-patch16-224"
    print(f"  CLIP: {clip_path}")
    clip_proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=CLIP_MODEL_DIR.exists())
    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=CLIP_MODEL_DIR.exists()).to(DEVICE).eval()
    print(f"  ViT:  {vit_path}")
    vit_proc = ViTImageProcessor.from_pretrained(vit_path, local_files_only=VIT_MODEL_DIR.exists())
    vit_model = ViTModel.from_pretrained(vit_path, local_files_only=VIT_MODEL_DIR.exists()).to(DEVICE).eval()
    return (clip_proc, clip_model), (vit_proc, vit_model)

@torch.no_grad()
def compute_clip_embeddings(image_paths, proc, model, desc="CLIP embed"):
    """Compute CLIP image embeddings, return normalized (N, 512) array."""
    embs = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=desc):
        batch = image_paths[i:i+BATCH_SIZE]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = proc(images=imgs, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        feats = model.get_image_features(**inputs)
        if hasattr(feats, "pooler_output"): feats = feats.pooler_output
        if isinstance(feats, tuple): feats = feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)

@torch.no_grad()
def compute_vit_embeddings(image_paths, proc, model, desc="ViT embed"):
    """Compute ViT [CLS] embeddings, return normalized (N, 768) array."""
    embs = []
    for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=desc):
        batch = image_paths[i:i+BATCH_SIZE]
        imgs = [Image.open(p).convert("RGB") for p in batch]
        inputs = proc(images=imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        cls = model(**inputs).last_hidden_state[:, 0, :]
        cls = cls / cls.norm(dim=-1, keepdim=True)
        embs.append(cls.cpu().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)

# ═══════════════════════════════════════════════════════════════
# Phase 3: Build similarity vectors via dot product
# ═══════════════════════════════════════════════════════════════

def build_similarities(originals, clip_proc, clip_model, vit_proc, vit_model):
    """Pre-compute all embeddings, then compute similarities via matrix multiply."""
    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

    n = len(originals)
    stem_to_idx = {info["stem"]: i for i, info in enumerate(originals)}

    # ── CLIP: originals + all 31 non-F2 transforms ──
    clip_cache = EMBEDDING_DIR / "clip_embeddings.npz"
    if clip_cache.exists():
        print("  Loading cached CLIP embeddings...")
        data = np.load(clip_cache)
        orig_clip = data["orig_clip"]
        transform_paths = list(data["transform_paths"])
        transform_clip = data["transform_clip"]
        # Build pos sims
        pos_sims = np.sum(orig_clip[data["pos_orig_idx"]] * transform_clip[data["pos_tf_idx"]], axis=1)
    else:
        print("  Computing CLIP embeddings for 1000 originals...")
        orig_paths = [str(info["path"]) for info in originals]
        orig_clip = compute_clip_embeddings(orig_paths, clip_proc, clip_model, "CLIP originals")

        # Collect all transform paths
        pos_orig_idx = []
        pos_tf_idx = []
        all_tf_paths = []
        tf_idx = 0
        for info in tqdm(originals, desc="  Collect transforms"):
            vdir = TRANSFORM_DIR / info["stem"]
            oi = stem_to_idx[info["stem"]]
            for vp in sorted(vdir.glob("*.jpg")):
                if vp.stem.startswith("F2"): continue
                all_tf_paths.append(str(vp))
                pos_orig_idx.append(oi)
                pos_tf_idx.append(tf_idx)
                tf_idx += 1
        pos_orig_idx = np.array(pos_orig_idx, dtype=np.int32)
        pos_tf_idx = np.array(pos_tf_idx, dtype=np.int32)

        print(f"  Computing CLIP embeddings for {len(all_tf_paths)} transforms...")
        transform_clip = compute_clip_embeddings(all_tf_paths, clip_proc, clip_model, "CLIP transforms")

        # Cosine similarity = dot product (already normalized)
        pos_sims = np.sum(orig_clip[pos_orig_idx] * transform_clip[pos_tf_idx], axis=1)

        np.savez_compressed(clip_cache,
                           orig_clip=orig_clip, transform_paths=np.array(all_tf_paths),
                           transform_clip=transform_clip, pos_orig_idx=pos_orig_idx,
                           pos_tf_idx=pos_tf_idx)

    print(f"  CLIP pos sims: {pos_sims.shape}, mean={pos_sims.mean():.4f}")

    # ── ViT: 1000 originals → all-pairs cosine similarity ──
    vit_cache = EMBEDDING_DIR / "vit_embeddings.npz"
    neg_sims_cache = EMBEDDING_DIR / "vit_neg_sims.npy"
    if neg_sims_cache.exists():
        print("  Loading cached ViT negative similarities...")
        neg_sims = np.load(neg_sims_cache)
    else:
        if vit_cache.exists():
            print("  Loading cached ViT embeddings...")
            orig_vit = np.load(vit_cache)["orig_vit"]
        else:
            print("  Computing ViT embeddings for 1000 originals...")
            orig_paths = [str(info["path"]) for info in originals]
            orig_vit = compute_vit_embeddings(orig_paths, vit_proc, vit_model, "ViT originals")
            np.savez_compressed(vit_cache, orig_vit=orig_vit)

        # All-pairs cosine similarity matrix: (1000, 1000) via matmul
        print("  Computing all-pairs ViT cosine similarity (1000×1000)...")
        t0 = time.time()
        sim_matrix = orig_vit @ orig_vit.T  # (1000, 1000)
        # Extract upper triangle (i < j) = C(1000,2) = 499500 values
        triu_idx = np.triu_indices(n, k=1)
        neg_sims = sim_matrix[triu_idx].astype(np.float32)
        print(f"  Done in {time.time()-t0:.1f}s, {len(neg_sims)} values")
        np.save(neg_sims_cache, neg_sims)

    print(f"  ViT neg sims: {neg_sims.shape}, mean={neg_sims.mean():.4f}")
    return pos_sims, neg_sims

# ═══════════════════════════════════════════════════════════════
# Phase 4: Grid evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_grid(pos_sims, neg_sims):
    n_pos = len(pos_sims); n_neg = len(neg_sims); total = n_pos + n_neg
    rows = []
    for ct in CLIP_THRESHOLDS:
        TP = int(np.sum(pos_sims >= ct))
        FN = n_pos - TP
        for vt in VIT_THRESHOLDS:
            TN = int(np.sum(neg_sims < vt))
            FP = n_neg - TN
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            accuracy = (TP + TN) / total if total > 0 else 0.0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
            rows.append({
                "clip_threshold": ct, "vit_threshold": vt,
                "clip_TP": TP, "clip_FN": FN, "vit_TN": TN, "vit_FP": FP,
                "overall_TP": TP, "overall_TN": TN, "overall_FP": FP, "overall_FN": FN,
                "accuracy": round(accuracy,4), "recall": round(recall,4),
                "precision": round(precision,4), "f1": round(f1,4),
            })
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
# Phase 5: Excel
# ═══════════════════════════════════════════════════════════════

def generate_excel(grid_df):
    import openpyxl
    from openpyxl.styles import Font
    from openpyxl.chart import LineChart, Reference

    wb = openpyxl.Workbook()
    hf = Font(bold=True, size=11)
    tf = Font(bold=True, size=14)
    sf = Font(bold=True, size=12)
    nf = Font(italic=True, size=9, color="888888")

    vit_labels = VIT_THRESHOLDS
    clip_labels = [f"CLIP={t:.2f}" for t in CLIP_THRESHOLDS]

    # ── 00_overview ──────────────────────────────────────────
    ws = wb.active; ws.title = "00_overview"
    best = grid_df.loc[grid_df["f1"].idxmax()]
    r = 1
    ws.cell(row=r, column=1, value="Threshold Experiment Overview").font = tf; r += 1
    bc, bv = float(best["clip_threshold"]), float(best["vit_threshold"])
    ws.cell(row=r, column=1, value="Overall (CLIP + ViT)").font = hf
    ws.cell(row=r, column=2, value=f"{bc}/{bv}")
    ws.cell(row=r, column=5, value="Best F1").font = hf
    ws.cell(row=r, column=6, value=f"CLIP={bc}, ViT={bv} → F1={best['f1']:.4f}")
    r += 2
    ws.cell(row=r, column=1, value="Baseline overall summary").font = sf; r += 2
    for m, v in [("TP", int(best["overall_TP"])), ("TN", int(best["overall_TN"])),
                 ("FP", int(best["overall_FP"])), ("FN", int(best["overall_FN"])),
                 ("accuracy", round(float(best["accuracy"]),4)),
                 ("recall", round(float(best["recall"]),4)),
                 ("precision", round(float(best["precision"]),4)),
                 ("f1", round(float(best["f1"]),4))]:
        ws.cell(row=r, column=1, value=m); ws.cell(row=r, column=2, value=v); r += 1
    r += 1
    ws.cell(row=r, column=1, value="Best threshold combinations").font = sf; r += 1
    headers = ["ranking","clip_threshold","vit_threshold","clip_TP","clip_FN","vit_TN","vit_FP",
               "overall_TP","overall_TN","overall_FP","overall_FN","accuracy","recall","precision","f1"]
    for j, h in enumerate(headers): ws.cell(row=r, column=j+1, value=h).font = hf
    r += 1
    for i, (_, row_data) in enumerate(grid_df.nlargest(5, "f1").iterrows()):
        ws.cell(row=r, column=1, value=f"top_f1_{i+1}")
        for j, h in enumerate(headers[1:], 2):
            val = row_data[h]
            ws.cell(row=r, column=j, value=round(float(val), 4) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else val)
        r += 1
    low_fp = grid_df.nsmallest(5, "overall_FP")
    for i, (_, row_data) in enumerate(low_fp.iterrows()):
        ws.cell(row=r, column=1, value=f"low_fp_{i+1}")
        for j, h in enumerate(headers[1:], 2):
            val = row_data[h]
            ws.cell(row=r, column=j, value=round(float(val), 4) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else val)
        r += 1
    r += 1
    ws.cell(row=r, column=1, value="Sheet guide").font = sf; r += 1
    for guide in [
        f"01_vit_sweep: Fixed CLIP={CLIP_THRESHOLDS[0]} ViT sweep ({VIT_THRESHOLDS[0]}–{VIT_THRESHOLDS[-1]})",
        f"02_clip_vit_grid: {len(grid_df)} CLIP×ViT grid results",
        "03_threshold_curves: Accuracy / Precision / F1 / FP curves",
        "04_f1_heatmap: F1 score heatmap",
        "05_fp_heatmap: Overall false positives heatmap"]:
        ws.cell(row=r, column=1, value=guide); r += 1

    # ── 01_vit_sweep ─────────────────────────────────────────
    ws1 = wb.create_sheet("01_vit_sweep")
    sweep = grid_df[grid_df["clip_threshold"] == CLIP_THRESHOLDS[0]]
    gh = ["clip_threshold","vit_threshold","clip_TP","clip_FN","vit_TN","vit_FP",
          "overall_TP","overall_TN","overall_FP","overall_FN","accuracy","recall","precision","f1"]
    for j, h in enumerate(gh): ws1.cell(row=1, column=j+1, value=h).font = hf
    for i, (_, row_data) in enumerate(sweep.iterrows()):
        for j, h in enumerate(gh):
            val = row_data[h]
            ws1.cell(row=i+2, column=j+1, value=round(float(val), 4) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else val)

    # ── 02_clip_vit_grid ─────────────────────────────────────
    ws2 = wb.create_sheet("02_clip_vit_grid")
    for j, h in enumerate(gh): ws2.cell(row=1, column=j+1, value=h).font = hf
    for i, (_, row_data) in enumerate(grid_df.iterrows()):
        for j, h in enumerate(gh):
            val = row_data[h]
            ws2.cell(row=i+2, column=j+1, value=round(float(val), 4) if isinstance(val, (float, np.floating)) else int(val) if isinstance(val, (int, np.integer)) else val)

    # ── 03_threshold_curves ──────────────────────────────────
    ws3 = wb.create_sheet("03_threshold_curves")
    ws3.cell(row=1, column=1, value="Threshold Curves - Chart Data").font = tf
    ws3.cell(row=2, column=1, value=f"Each line = one CLIP threshold; X-axis = ViT threshold ({VIT_THRESHOLDS[0]}–{VIT_THRESHOLDS[-1]}).").font = nf
    cr = 4
    metric_specs = [("F1 Score","f1"), ("Accuracy","accuracy"), ("Precision","precision"), ("False Positives (FP)","overall_FP")]
    for mname, mcol in metric_specs:
        ws3.cell(row=cr, column=1, value=mname).font = sf; cr += 1
        ws3.cell(row=cr, column=1, value="ViT Threshold").font = hf
        for j, cl in enumerate(clip_labels): ws3.cell(row=cr, column=2+j, value=cl).font = hf
        cr += 1
        for vt in vit_labels:
            ws3.cell(row=cr, column=1, value=vt)
            for j, ct in enumerate(CLIP_THRESHOLDS):
                match = grid_df[(grid_df["clip_threshold"]==ct)&(grid_df["vit_threshold"]==vt)]
                if len(match):
                    val = match.iloc[0][mcol]
                    ws3.cell(row=cr, column=2+j, value=round(float(val),4) if mcol!="overall_FP" else int(val))
            cr += 1
        cr += 1

    chart_specs = [("F1 Score",5,26), ("Accuracy",29,50), ("Precision",53,74), ("False Positives (FP)",77,98)]
    colors = ["1F4E79","2E75B6","5B9BD5","9DC3E6","BDD7EE","A9CCE3"]
    for sec_name, hdr_row, data_end in chart_specs:
        chart = LineChart(); chart.title = f"ViT Threshold vs {sec_name}"; chart.style = 2
        chart.width = 20; chart.height = 12; chart.y_axis.title = sec_name
        chart.y_axis.numFmt = '#,##0' if sec_name == "False Positives (FP)" else '0.0000'
        chart.x_axis.title = "ViT Threshold"; chart.x_axis.numFmt = '0.00'
        for j in range(6):
            vals = Reference(ws3, min_col=2+j, min_row=hdr_row, max_row=data_end)
            chart.add_data(vals, titles_from_data=True)
            chart.series[j].graphicalProperties.line.width = 21000
            chart.series[j].graphicalProperties.line.solidFill = colors[j]
        chart.set_categories(Reference(ws3, min_col=1, min_row=hdr_row+1, max_row=data_end))
        chart.legend.position = 'b'
        ws3.add_chart(chart, f"A{data_end+2}")
    nr = 100
    ws3.cell(row=nr, column=1, value="Note:").font = Font(bold=True, size=10, color="333333")
    ws3.cell(row=nr+1, column=1, value="FP chart: All CLIP lines overlap (ViT FP independent of CLIP).").font = nf
    ws3.cell(row=nr+2, column=1, value="F1/Acc/Prec charts: Lines very close (ViT dominates; CLIP FN differences minimal).").font = nf

    # ── 04_f1_heatmap ────────────────────────────────────────
    ws4 = wb.create_sheet("04_f1_heatmap")
    ws4.cell(row=1, column=1, value="F1 Heatmap").font = tf
    ws4.cell(row=2, column=1, value="Rows=ViT threshold, Cols=CLIP threshold; greener=higher F1.").font = nf
    ws4.cell(row=4, column=1, value="clip_threshold").font = hf
    for j, ct in enumerate(CLIP_THRESHOLDS): ws4.cell(row=4, column=2+j, value=ct)
    piv_f1 = grid_df.pivot_table(values="f1", index="vit_threshold", columns="clip_threshold")
    for i, vt in enumerate(vit_labels):
        ws4.cell(row=5+i, column=1, value=vt)
        for j, ct in enumerate(CLIP_THRESHOLDS):
            if vt in piv_f1.index and ct in piv_f1.columns:
                ws4.cell(row=5+i, column=2+j, value=round(float(piv_f1.loc[vt, ct]), 4))

    # ── 05_fp_heatmap ────────────────────────────────────────
    ws5 = wb.create_sheet("05_fp_heatmap")
    ws5.cell(row=1, column=1, value="Overall FP Heatmap").font = tf
    ws5.cell(row=2, column=1, value="Rows=CLIP threshold, Cols=ViT threshold; greener=fewer FP.").font = nf
    ws5.cell(row=4, column=1, value="clip_threshold").font = hf
    for j, vt in enumerate(vit_labels): ws5.cell(row=4, column=2+j, value=vt)
    piv_fp = grid_df.pivot_table(values="overall_FP", index="clip_threshold", columns="vit_threshold")
    for i, ct in enumerate(CLIP_THRESHOLDS):
        ws5.cell(row=5+i, column=1, value=ct)
        for j, vt in enumerate(vit_labels):
            if ct in piv_fp.index and vt in piv_fp.columns:
                ws5.cell(row=5+i, column=2+j, value=int(piv_fp.loc[ct, vt]))

    xl_path = OUT_DIR / "overall_summary_excel_chart_datasets2_1.xlsx"
    wb.save(xl_path)
    print(f"  Excel → {xl_path}")

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print(f"\n[1/5] {gather_originals.__doc__}")
    originals = gather_originals()
    print(f"  {len(originals)} originals")

    print(f"\n[2/5] Generating transforms...")
    t0 = time.time()
    generate_transforms(originals)
    t_trans = time.time() - t0
    print(f"  {t_trans:.1f}s")

    print(f"\n[3/5] Loading models...")
    (clip_proc, clip_model), (vit_proc, vit_model) = load_models()

    print(f"\n[4/5] Computing similarities (embedding-based, fast)...")
    t0 = time.time()
    pos_sims, neg_sims = build_similarities(originals, clip_proc, clip_model, vit_proc, vit_model)
    t_sim = time.time() - t0
    print(f"  {t_sim:.1f}s total")

    print(f"\n[5/5] Evaluating {len(CLIP_THRESHOLDS)}×{len(VIT_THRESHOLDS)}={len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)} grid...")
    grid_df = evaluate_grid(pos_sims, neg_sims)

    # Save CSVs
    grid_df.to_csv(OUT_DIR / "threshold_grid.csv", index=False, float_format="%.4f")
    np.savez_compressed(OUT_DIR / "similarities.npz", clip_pos_sims=pos_sims, vit_neg_sims=neg_sims)

    best = grid_df.loc[grid_df["f1"].idxmax()]
    print(f"\n  Best F1: CLIP={best['clip_threshold']:.2f} ViT={best['vit_threshold']:.2f} "
          f"→ F1={best['f1']:.4f} Acc={best['accuracy']:.4f} FP={int(best['overall_FP'])}")
    print(f"  CLIP sims: mean={pos_sims.mean():.4f} std={pos_sims.std():.4f}")
    print(f"  ViT sims:  mean={neg_sims.mean():.4f} std={neg_sims.std():.4f}")

    total_t = time.time() - t_start
    json.dump({
        "num_originals": len(originals), "positive_pairs": len(pos_sims), "negative_pairs": len(neg_sims),
        "clip_thresholds": CLIP_THRESHOLDS, "vit_thresholds": VIT_THRESHOLDS,
        "transform_time_sec": round(t_trans, 1), "similarity_time_sec": round(t_sim, 1),
        "total_time_sec": round(total_t, 1),
        "best_f1": {"clip_threshold": float(best["clip_threshold"]), "vit_threshold": float(best["vit_threshold"]),
                     "f1": float(best["f1"]), "accuracy": float(best["accuracy"]), "fp": int(best["overall_FP"])}
    }, open(OUT_DIR / "timing.json", "w"), indent=2)

    print("\nGenerating Excel...")
    generate_excel(grid_df)

    print(f"\nDone in {total_t:.0f}s → {OUT_DIR}")

if __name__ == "__main__":
    main()
