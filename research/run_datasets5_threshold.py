#!/usr/bin/env python3
"""
CLIP + ViT two-stage threshold evaluation on Datasets5 (ArtBench-10).
1000 images × 32 transforms = 31,000 positive + 499,500 negative pairs.
"""

from __future__ import annotations
import json, time, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "Datasets5"
TRANSFORM_DIR = BASE_DIR / "Transform" / "datasets5"
OUT_DIR = BASE_DIR / "analysis" / "threshold_eval_datasets5"
EMBEDDING_DIR = OUT_DIR / "embeddings"

MODEL_DIR = BASE_DIR / "models"
CLIP_MODEL_DIR = MODEL_DIR / "clip-vit-base-patch32"
VIT_MODEL_DIR  = MODEL_DIR / "vit-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 128

CLIP_THRESHOLDS = [round(x/100, 2) for x in range(80, 86)]  # 0.80–0.85
VIT_THRESHOLDS  = [round(x/100, 2) for x in range(40, 51)]  # 0.40–0.50

ART_STYLES = ["art_nouveau", "baroque", "expressionism", "impressionism",
              "post_impressionism", "realism", "renaissance", "romanticism",
              "surrealism", "ukiyo_e"]
IMAGES_PER_STYLE = 100

OUT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE} | Batch: {BATCH_SIZE}")
print(f"CLIP: {CLIP_THRESHOLDS}  |  ViT: {VIT_THRESHOLDS}")
print(f"Grid: {len(CLIP_THRESHOLDS)}×{len(VIT_THRESHOLDS)}={len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)}")

# ═══════════════════════════════════════════════════════════════
# Phase 1: Gather originals
# ═══════════════════════════════════════════════════════════════

def gather_originals():
    images = []
    for style in ART_STYLES:
        style_dir = DATASETS_DIR / style
        if not style_dir.exists(): continue
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for fpath in sorted(style_dir.iterdir()):
            if fpath.suffix.lower() in exts:
                images.append({"path": fpath, "style": style, "stem": fpath.stem})
    return images

# ═══════════════════════════════════════════════════════════════
# Phase 2: Load models
# ═══════════════════════════════════════════════════════════════

def load_models():
    clip_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
    vit_path  = str(VIT_MODEL_DIR)  if VIT_MODEL_DIR.exists()  else "google/vit-base-patch16-224"
    print(f"  CLIP: {clip_path}")
    lp = CLIP_MODEL_DIR.exists()
    clip_proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=lp)
    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=lp).to(DEVICE).eval()
    print(f"  ViT:  {vit_path}")
    vp = VIT_MODEL_DIR.exists()
    vit_proc = ViTImageProcessor.from_pretrained(vit_path, local_files_only=vp)
    vit_model = ViTModel.from_pretrained(vit_path, local_files_only=vp).to(DEVICE).eval()
    return (clip_proc, clip_model), (vit_proc, vit_model)

# ═══════════════════════════════════════════════════════════════
# Phase 3: Embeddings & similarities
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_clip_embeddings(image_paths, proc, model, desc="CLIP"):
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
def compute_vit_embeddings(image_paths, proc, model, desc="ViT"):
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

def build_similarities(originals, clip_proc, clip_model, vit_proc, vit_model):
    n = len(originals)
    # Map stem → index, and also style-category-qualified stem (folder name in Transform/datasets5)
    # Transform folders are named: {style}_{stem}
    folder_to_idx = {}
    for i, info in enumerate(originals):
        folder_name = f"{info['style']}_{info['stem']}"
        folder_to_idx[folder_name] = i

    # ── CLIP: originals (1000) + transforms (31000) ──
    clip_cache = EMBEDDING_DIR / "clip_embeddings.npz"
    if clip_cache.exists():
        print("  Loading cached CLIP embeddings...")
        data = np.load(clip_cache, allow_pickle=True)
        orig_clip = data["orig_clip"]
        pos_sims = data["pos_sims"]
    else:
        print("  Computing CLIP embeddings for originals...")
        orig_paths = [str(info["path"]) for info in originals]
        orig_clip = compute_clip_embeddings(orig_paths, clip_proc, clip_model, "CLIP originals")

        # Collect all transform images (skip F2)
        pos_orig_idx = []; all_tf_paths = []
        tf_count = 0
        for info in tqdm(originals, desc="  Collect transforms"):
            folder_name = f"{info['style']}_{info['stem']}"
            tf_dir = TRANSFORM_DIR / folder_name
            oi = folder_to_idx[folder_name]
            if not tf_dir.exists():
                continue
            for vp in sorted(tf_dir.glob("*.jpg")):
                if vp.stem.startswith("F2"): continue
                all_tf_paths.append(str(vp))
                pos_orig_idx.append(oi)
        pos_orig_idx = np.array(pos_orig_idx, dtype=np.int32)
        print(f"  Transforms: {len(all_tf_paths)} (expect ~31000)")

        print(f"  Computing CLIP embeddings for transforms...")
        transform_clip = compute_clip_embeddings(all_tf_paths, clip_proc, clip_model, "CLIP transforms")

        pos_sims = np.sum(orig_clip[pos_orig_idx] * transform_clip, axis=1)
        np.savez_compressed(clip_cache, orig_clip=orig_clip, pos_sims=pos_sims,
                           transform_paths=np.array(all_tf_paths),
                           pos_orig_idx=pos_orig_idx)

    print(f"  CLIP pos sims: {pos_sims.shape}, mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}")

    # ── ViT: originals (1000) → all-pairs ──
    neg_cache = EMBEDDING_DIR / "vit_neg_sims.npy"
    if neg_cache.exists():
        print("  Loading cached ViT negative similarities...")
        neg_sims = np.load(neg_cache)
    else:
        vit_cache = EMBEDDING_DIR / "vit_embeddings.npz"
        if vit_cache.exists():
            print("  Loading cached ViT embeddings...")
            orig_vit = np.load(vit_cache)["orig_vit"]
        else:
            print("  Computing ViT embeddings for originals...")
            orig_paths = [str(info["path"]) for info in originals]
            orig_vit = compute_vit_embeddings(orig_paths, vit_proc, vit_model, "ViT originals")
            np.savez_compressed(vit_cache, orig_vit=orig_vit)

        print("  Computing all-pairs ViT cosine similarity (1000×1000)...")
        t0 = time.time()
        sim_matrix = orig_vit @ orig_vit.T
        triu_idx = np.triu_indices(n, k=1)
        neg_sims = sim_matrix[triu_idx].astype(np.float32)
        print(f"  Done in {time.time()-t0:.1f}s, {len(neg_sims)} values")
        np.save(neg_cache, neg_sims)

    print(f"  ViT neg sims: {neg_sims.shape}, mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}")
    return pos_sims, neg_sims

# ═══════════════════════════════════════════════════════════════
# Phase 4: Grid evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_grid(pos_sims, neg_sims):
    n_pos, n_neg = len(pos_sims), len(neg_sims)
    total = n_pos + n_neg
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
                "accuracy": round(accuracy,6), "recall": round(recall,6),
                "precision": round(precision,6), "f1": round(f1,6),
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
    hf = Font(bold=True, size=11); tf = Font(bold=True, size=14)
    sf = Font(bold=True, size=12); nf = Font(italic=True, size=9, color="888888")

    vit_labels = VIT_THRESHOLDS
    clip_labels = [f"CLIP={t:.2f}" for t in CLIP_THRESHOLDS]

    # ── 00_overview ──
    ws = wb.active; ws.title = "00_overview"
    best = grid_df.loc[grid_df["f1"].idxmax()]
    r = 1
    ws.cell(row=r, column=1, value="Datasets5 Threshold Experiment (ArtBench-10)").font = tf; r += 1
    bc, bv = float(best["clip_threshold"]), float(best["vit_threshold"])
    ws.cell(row=r, column=1, value="Overall (CLIP + ViT)").font = hf
    ws.cell(row=r, column=2, value=f"CLIP={bc:.2f} / ViT={bv:.2f}"); r += 1
    ws.cell(row=r, column=5, value="Best F1").font = hf
    ws.cell(row=r, column=6, value=f"F1={best['f1']:.4f}  Acc={best['accuracy']:.4f}  FP={int(best['overall_FP'])}"); r += 2
    ws.cell(row=r, column=1, value="Baseline overview").font = sf; r += 2
    for m, v in [("TP", int(best["overall_TP"])), ("TN", int(best["overall_TN"])),
                 ("FP", int(best["overall_FP"])), ("FN", int(best["overall_FN"])),
                 ("accuracy", round(float(best["accuracy"]),6)),
                 ("recall", round(float(best["recall"]),6)),
                 ("precision", round(float(best["precision"]),6)),
                 ("f1", round(float(best["f1"]),6))]:
        ws.cell(row=r, column=1, value=m); ws.cell(row=r, column=2, value=v); r += 1
    r += 1
    ws.cell(row=r, column=1, value="Top 5 by F1").font = sf; r += 1
    gh = ["ranking","clip_threshold","vit_threshold","clip_TP","clip_FN","vit_TN","vit_FP",
          "overall_TP","overall_TN","overall_FP","overall_FN","accuracy","recall","precision","f1"]
    for j, h in enumerate(gh): ws.cell(row=r, column=j+1, value=h).font = hf
    r += 1
    for i, (_, row_data) in enumerate(grid_df.nlargest(5, "f1").iterrows()):
        ws.cell(row=r, column=1, value=f"top_{i+1}")
        for j, h in enumerate(gh[1:], 2):
            val = row_data[h]
            ws.cell(row=r, column=j, value=round(float(val),4) if isinstance(val,(float,np.floating)) else int(val))
        r += 1

    # ── 01_grid ──
    ws1 = wb.create_sheet("01_grid")
    for j, h in enumerate(gh[1:]): ws1.cell(row=1, column=j+1, value=h).font = hf
    for i, (_, row_data) in enumerate(grid_df.iterrows()):
        for j, h in enumerate(gh[1:], 2):
            val = row_data[h]
            ws1.cell(row=i+2, column=j, value=round(float(val),4) if isinstance(val,(float,np.floating)) else int(val))

    # ── 02_curves ──
    ws2 = wb.create_sheet("02_curves")
    ws2.cell(row=1, column=1, value="Threshold Curves — Datasets5").font = tf
    ws2.cell(row=2, column=1, value=f"X-axis = ViT threshold ({VIT_THRESHOLDS[0]}–{VIT_THRESHOLDS[-1]}); 6 lines = 6 CLIP thresholds").font = nf
    cr = 4
    for mname, mcol in [("F1","f1"),("Accuracy","accuracy"),("Precision","precision"),("FP","overall_FP")]:
        ws2.cell(row=cr, column=1, value=mname).font = sf; cr += 1
        ws2.cell(row=cr, column=1, value="ViT").font = hf
        for j, cl in enumerate(clip_labels): ws2.cell(row=cr, column=2+j, value=cl).font = hf
        cr += 1
        for vt in vit_labels:
            ws2.cell(row=cr, column=1, value=vt)
            for j, ct in enumerate(CLIP_THRESHOLDS):
                m = grid_df[(grid_df["clip_threshold"]==ct)&(grid_df["vit_threshold"]==vt)]
                if len(m):
                    val = m.iloc[0][mcol]
                    ws2.cell(row=cr, column=2+j, value=round(float(val),4) if mcol!="overall_FP" else int(val))
            cr += 1
        cr += 1

    colors = ["1F4E79","2E75B6","5B9BD5","9DC3E6","BDD7EE","A9CCE3"]
    for sec_name, hdr_row, end_row in [("F1",6,26),("Accuracy",29,49),("Precision",52,72),("FP",75,95)]:
        chart = LineChart(); chart.title = sec_name; chart.style = 2
        chart.width=20; chart.height=12; chart.y_axis.title=sec_name
        chart.x_axis.title="ViT Threshold"
        for j in range(6):
            vals = Reference(ws2, min_col=2+j, min_row=hdr_row, max_row=end_row)
            chart.add_data(vals, titles_from_data=True)
            chart.series[j].graphicalProperties.line.width = 21000
            chart.series[j].graphicalProperties.line.solidFill = colors[j]
        chart.set_categories(Reference(ws2, min_col=1, min_row=hdr_row+1, max_row=end_row))
        chart.legend.position='b'
        ws2.add_chart(chart, f"A{end_row+2}")

    # ── 03_heatmaps ──
    ws3 = wb.create_sheet("03_heatmaps")
    ws3.cell(row=1, column=1, value="F1 Heatmap").font = tf; r=4
    ws3.cell(row=4, column=1, value="ViT↓ CLIP→").font = hf
    for j, ct in enumerate(CLIP_THRESHOLDS): ws3.cell(row=4, column=2+j, value=ct)
    piv_f1 = grid_df.pivot_table(values="f1", index="vit_threshold", columns="clip_threshold")
    for i, vt in enumerate(vit_labels):
        ws3.cell(row=5+i, column=1, value=vt)
        for j, ct in enumerate(CLIP_THRESHOLDS):
            if vt in piv_f1.index and ct in piv_f1.columns:
                ws3.cell(row=5+i, column=2+j, value=round(float(piv_f1.loc[vt, ct]),4))

    ws3.cell(row=20, column=1, value="FP Heatmap").font = tf
    ws3.cell(row=22, column=1, value="ViT↓ CLIP→").font = hf
    for j, ct in enumerate(CLIP_THRESHOLDS): ws3.cell(row=22, column=2+j, value=ct)
    piv_fp = grid_df.pivot_table(values="overall_FP", index="vit_threshold", columns="clip_threshold")
    for i, vt in enumerate(vit_labels):
        ws3.cell(row=23+i, column=1, value=vt)
        for j, ct in enumerate(CLIP_THRESHOLDS):
            if vt in piv_fp.index and ct in piv_fp.columns:
                ws3.cell(row=23+i, column=2+j, value=int(piv_fp.loc[vt, ct]))

    xl_path = OUT_DIR / "overall_summary_datasets5.xlsx"
    wb.save(xl_path)
    print(f"  Excel → {xl_path}")

# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    t_start = time.time()

    print(f"\n[1/4] Gathering originals from Datasets5...")
    originals = gather_originals()
    print(f"  {len(originals)} originals across {len(set(o['style'] for o in originals))} styles")

    # Verify transforms exist
    missing = 0
    for info in originals:
        folder = TRANSFORM_DIR / f"{info['style']}_{info['stem']}"
        if not folder.exists() or len(list(folder.glob("*.jpg"))) < 32:
            missing += 1
    if missing:
        print(f"  ⚠ {missing} originals missing transforms")
    else:
        print(f"  All transforms present ✓")

    print(f"\n[2/4] Loading models...")
    (clip_proc, clip_model), (vit_proc, vit_model) = load_models()

    print(f"\n[3/4] Computing similarities...")
    t0 = time.time()
    pos_sims, neg_sims = build_similarities(originals, clip_proc, clip_model, vit_proc, vit_model)
    t_sim = time.time() - t0
    print(f"  Similarity time: {t_sim:.1f}s")

    print(f"\n[4/4] Grid evaluation ({len(CLIP_THRESHOLDS)}×{len(VIT_THRESHOLDS)}={len(CLIP_THRESHOLDS)*len(VIT_THRESHOLDS)})...")
    grid_df = evaluate_grid(pos_sims, neg_sims)
    grid_df.to_csv(OUT_DIR / "threshold_grid.csv", index=False, float_format="%.6f")
    np.savez_compressed(OUT_DIR / "similarities.npz",
                       clip_pos_sims=pos_sims, vit_neg_sims=neg_sims)

    best = grid_df.loc[grid_df["f1"].idxmax()]
    print(f"\n  {'='*55}")
    print(f"  Best: CLIP={best['clip_threshold']:.2f}  ViT={best['vit_threshold']:.2f}")
    print(f"  F1={best['f1']:.4f}  Acc={best['accuracy']:.4f}  "
          f"TP={int(best['overall_TP'])}  TN={int(best['overall_TN'])}  "
          f"FP={int(best['overall_FP'])}  FN={int(best['overall_FN'])}")
    print(f"  Positive pairs: {len(pos_sims)}  |  Negative pairs: {len(neg_sims)}")
    print(f"  CLIP sims: mean={pos_sims.mean():.4f}  std={pos_sims.std():.4f}  "
          f"min={pos_sims.min():.4f}  max={pos_sims.max():.4f}")
    print(f"  ViT sims:  mean={neg_sims.mean():.4f}  std={neg_sims.std():.4f}  "
          f"min={neg_sims.min():.4f}  max={neg_sims.max():.4f}")
    print(f"  {'='*55}")

    total_t = time.time() - t_start
    json.dump({
        "dataset": "Datasets5 (ArtBench-10)", "num_originals": len(originals),
        "num_styles": len(set(o["style"] for o in originals)),
        "positive_pairs": int(len(pos_sims)), "negative_pairs": int(len(neg_sims)),
        "clip_thresholds": CLIP_THRESHOLDS, "vit_thresholds": VIT_THRESHOLDS,
        "similarity_time_sec": round(t_sim, 1), "total_time_sec": round(total_t, 1),
        "clip_sim_stats": {"mean": float(pos_sims.mean()), "std": float(pos_sims.std())},
        "vit_sim_stats": {"mean": float(neg_sims.mean()), "std": float(neg_sims.std())},
        "best": {"clip_threshold": float(best["clip_threshold"]),
                 "vit_threshold": float(best["vit_threshold"]),
                 "f1": float(best["f1"]), "accuracy": float(best["accuracy"]),
                 "TP": int(best["overall_TP"]), "TN": int(best["overall_TN"]),
                 "FP": int(best["overall_FP"]), "FN": int(best["overall_FN"])},
    }, open(OUT_DIR / "timing.json", "w"), indent=2)

    print("\nGenerating Excel...")
    generate_excel(grid_df)

    print(f"\nDone! ({total_t:.0f}s) → {OUT_DIR}")

if __name__ == "__main__":
    main()
