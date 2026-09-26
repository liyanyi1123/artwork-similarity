#!/usr/bin/env python3
"""
Complete pipeline for Datasets6 threshold sweep:
1. Organize images into style subdirectories
2. Generate 32 transforms per image
3. Compute CLIP/ViT similarities (positive + negative pairs)
4. Run full sweep (ViT 0-1, CLIP 0-1, step 0.01)
5. Generate comparison charts

All output saved to exp4/datasets6_sweep/
"""

from __future__ import annotations
import csv, json, time, sys, subprocess
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS6 = BASE_DIR / "Datasets6"
TRANSFORM_DIR = BASE_DIR / "Transform" / "datasets6"
OUT_DIR = Path(__file__).resolve().parent / "datasets6_sweep"
EMBED_DIR = OUT_DIR / "embeddings"

ART_STYLES = ["art_nouveau", "baroque", "expressionism", "impressionism",
              "post_impressionism", "realism", "renaissance", "romanticism",
              "surrealism", "ukiyo_e"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEVICE = None  # Set after torch import


def step1_organize_images():
    """Create style subdirectories in Datasets6 with symlinks."""
    print("=" * 60)
    print("STEP 1: Organize Datasets6 images into style subdirectories")
    print("=" * 60)

    # Read sources.csv
    sources_csv = DATASETS6 / "sources.csv"
    if not sources_csv.exists():
        print("ERROR: sources.csv not found in Datasets6")
        sys.exit(1)

    with open(sources_csv, newline="", encoding="utf-8") as f:
        sources = list(csv.DictReader(f))

    style_counts = {}
    for style in ART_STYLES:
        style_dir = DATASETS6 / style
        style_dir.mkdir(exist_ok=True)
        style_counts[style] = 0

    # Symlink images into style directories
    for s in sources:
        style = s.get("style", "")
        if style not in style_counts:
            continue
        fname = s["filename"]
        src = DATASETS6 / fname
        if not src.exists():
            continue
        dst = DATASETS6 / style / fname
        if not dst.exists():
            dst.symlink_to(src.relative_to(dst.parent, walk_up=True)
                          if hasattr(Path, 'walk_up') else f"../{fname}")
        style_counts[style] += 1

    for style in ART_STYLES:
        actual = len(list((DATASETS6 / style).glob("*")))
        print(f"  {style}: {actual} images")

    return sources


def step2_generate_transforms():
    """Generate 32 transforms per image (reuse existing transform logic)."""
    print("\n" + "=" * 60)
    print("STEP 2: Generate transforms (32 per image)")
    print("=" * 60)

    import cv2

    # Reuse the exact same operations from generate_datasets5_transforms.py
    def op_C(img, strength=1.0):
        h, w = img.shape[:2]
        removal = min(0.05 * strength, 0.95)
        crop_ratio = 1.0 - removal
        new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
        y0, x0 = (h - new_h) // 2, (w - new_w) // 2
        cropped = img[y0:y0 + new_h, x0:x0 + new_w]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    def op_S1(img, strength=1.0):
        h, w = img.shape[:2]
        shift_px = int(0.05 * w * strength)
        M = np.float32([[1, 0, shift_px], [0, 1, 0]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def op_S2(img, strength=1.0):
        h, w = img.shape[:2]
        shift_px = int(0.05 * h * strength)
        M = np.float32([[1, 0, 0], [0, 1, shift_px]])
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def op_R(img, strength=1.0):
        h, w = img.shape[:2]
        angle = 2.0 * strength
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def op_B(img, strength=1.0):
        alpha = 1.0 + 0.04 * strength
        beta = 8.0 * strength
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def op_V(img, strength=1.0):
        h, w = img.shape[:2]
        y, x = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2
        dx, dy = (x - cx) / (w / 2), (y - cy) / (h / 2)
        dist = np.sqrt(dx ** 2 + dy ** 2)
        reduction = 0.25 * strength
        exponent = 1.5 * strength
        mask = np.clip(1.0 - dist * reduction, 0.0, 1.0)
        mask = np.power(mask, exponent)
        return (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)

    def op_T(img, strength=1.0):
        result = img.astype(np.float32)
        result[:, :, 2] = np.clip(result[:, :, 2] * (1.0 + 0.06 * strength), 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] * (1.0 + 0.02 * strength), 0, 255)
        result[:, :, 0] = np.clip(result[:, :, 0] * (1.0 - 0.05 * strength), 0, 255)
        return result.astype(np.uint8)

    def op_Sh(img, strength=1.0):
        sk = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        ik = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        w = min(0.4 * strength, 0.95)
        return cv2.filter2D(img, -1, w * sk + (1.0 - w) * ik)

    def op_K(img, strength=1.0):
        h, w = img.shape[:2]
        margin = 0.03 * strength
        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([[0, 0], [w, h * margin], [w, h * (1 - margin)], [0, h]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    def op_N(img, strength=1.0):
        rng = np.random.default_rng(42)
        noise_std = 3.0 * strength
        noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def op_F1(img): return cv2.flip(img, 1)
    def op_F2(img): return cv2.flip(img, 0)

    OP_SETTINGS = {
        "C": [(0.4, "low"), (1.0, "mid"), (1.6, "high")],
        "S1": [(0.4, "low"), (1.0, "mid"), (1.6, "high")],
        "S2": [(0.4, "low"), (1.0, "mid"), (1.6, "high")],
        "R": [(1.5, "low"), (3.0, "mid"), (6.0, "high")],
        "B": [(0.5, "low"), (1.0, "mid"), (1.5, "high")],
        "V": [(0.4, "low"), (1.0, "mid"), (1.6, "high")],
        "T": [(0.5, "low"), (1.0, "mid"), (1.5, "high")],
        "Sh": [(0.5, "low"), (1.0, "mid"), (1.5, "high")],
        "K": [(0.33, "low"), (1.0, "mid"), (1.67, "high")],
        "N": [(0.5, "low"), (1.0, "mid"), (1.5, "high")],
    }

    OPERATIONS = [
        ("F1", op_F1, False), ("F2", op_F2, False),
        ("C", op_C, True), ("S1", op_S1, True), ("S2", op_S2, True),
        ("R", op_R, True), ("B", op_B, True), ("V", op_V, True),
        ("T", op_T, True), ("Sh", op_Sh, True), ("K", op_K, True), ("N", op_N, True),
    ]

    def process_one(args):
        img_path, output_dir = args
        category = img_path.parent.name
        stem = img_path.stem
        out_dir = output_dir / f"{category}_{stem}"
        img = cv2.imread(str(img_path))
        if img is None:
            return (img_path.name, False, 0)
        out_dir.mkdir(parents=True, exist_ok=True)
        generated = 0
        for symbol, func, tunable in OPERATIONS:
            if not tunable:
                result = func(img)
                cv2.imwrite(str(out_dir / f"{symbol}_{symbol}.jpg"), result)
                generated += 1
            else:
                for strength, label in OP_SETTINGS[symbol]:
                    result = func(img, strength=strength)
                    cv2.imwrite(str(out_dir / f"{symbol}_{label}_{symbol}.jpg"), result)
                    generated += 1
        return (img_path.name, True, generated)

    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)

    # Gather images from style subdirectories
    images = []
    for style in ART_STYLES:
        d = DATASETS6 / style
        if d.exists():
            images.extend(p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS)

    # Skip if already done
    existing_transforms = len(list(TRANSFORM_DIR.glob("*")))
    if existing_transforms >= len(images):
        print(f"  Transforms already exist: {existing_transforms} folders → skipping")
        return

    # Check what's already done
    already_done = set(d.name for d in TRANSFORM_DIR.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) >= 32)
    tasks = [(p, TRANSFORM_DIR) for p in images if f"{p.parent.name}_{p.stem}" not in already_done]

    if not tasks:
        print("  All transforms already generated → skipping")
        return

    print(f"  {len(tasks)} images to process ({len(already_done)} already done)")
    t0 = time.time()
    done, failed = 0, 0

    with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for f in as_completed(futures):
            name, ok, gen = f.result()
            if ok: done += 1
            else: failed += 1
            if (done + failed) % 100 == 0:
                elapsed = time.time() - t0
                rate = (done + failed) / elapsed
                eta = (len(tasks) - done - failed) / rate if rate > 0 else 0
                print(f"  [{done+failed}/{len(tasks)}] done={done} failed={failed}  {rate:.1f}/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"  Transforms done in {elapsed:.1f}s: {done} ok, {failed} failed")


def step3_compute_similarities():
    """Compute CLIP and ViT similarities for positive and negative pairs."""
    print("\n" + "=" * 60)
    print("STEP 3: Compute CLIP/ViT similarities")
    print("=" * 60)

    import torch
    from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
    from PIL import Image

    global DEVICE
    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    BATCH_SIZE = 128

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    MODEL_DIR = BASE_DIR / "models"
    CLIP_DIR = MODEL_DIR / "clip-vit-base-patch32"
    VIT_DIR = MODEL_DIR / "vit-base-patch16-224"

    clip_path = str(CLIP_DIR) if CLIP_DIR.exists() else "openai/clip-vit-base-patch32"
    vit_path = str(VIT_DIR) if VIT_DIR.exists() else "google/vit-base-patch16-224"

    print(f"  Device: {DEVICE}, Batch: {BATCH_SIZE}")
    print(f"  CLIP: {clip_path}")
    print(f"  ViT:  {vit_path}")

    clip_proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=CLIP_DIR.exists())
    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=CLIP_DIR.exists()).to(DEVICE).eval()
    vit_proc = ViTImageProcessor.from_pretrained(vit_path, local_files_only=VIT_DIR.exists())
    vit_model = ViTModel.from_pretrained(vit_path, local_files_only=VIT_DIR.exists()).to(DEVICE).eval()

    @torch.no_grad()
    def get_clip_embs(paths, desc="CLIP"):
        embs = []
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc):
            batch = paths[i:i+BATCH_SIZE]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = clip_proc(images=imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            feats = clip_model.get_image_features(**inputs)
            if isinstance(feats, tuple): feats = feats[0]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)

    @torch.no_grad()
    def get_vit_embs(paths, desc="ViT"):
        embs = []
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc):
            batch = paths[i:i+BATCH_SIZE]
            imgs = [Image.open(p).convert("RGB") for p in batch]
            inputs = vit_proc(images=imgs, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            cls = vit_model(**inputs).last_hidden_state[:, 0, :]
            cls = cls / cls.norm(dim=-1, keepdim=True)
            embs.append(cls.cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)

    # Gather originals
    originals = []
    for style in ART_STYLES:
        d = DATASETS6 / style
        if d.exists():
            originals.extend(sorted(d.iterdir()))
    originals = [p for p in originals if p.suffix.lower() in IMG_EXTS]
    n_orig = len(originals)
    print(f"\n  Originals: {n_orig}")

    # Build folder-to-index map
    folder_to_idx = {}
    for i, p in enumerate(originals):
        folder_name = f"{p.parent.name}_{p.stem}"
        folder_to_idx[folder_name] = i

    # ── CLIP positive similarities ──
    clip_cache = EMBED_DIR / "clip_embeddings.npz"
    if clip_cache.exists():
        print("  Loading cached CLIP embeddings...")
        data = np.load(clip_cache, allow_pickle=True)
        pos_clip = data["pos_sims"]
        orig_clip = data["orig_clip"]
        pos_orig_idx = data["pos_orig_idx"]
    else:
        orig_paths = [str(p) for p in originals]
        orig_clip = get_clip_embs(orig_paths, "CLIP originals")

        # Collect transforms (skip F2)
        pos_orig_idx = []
        all_tf_paths = []
        for p in tqdm(originals, desc="  Collect transforms"):
            folder_name = f"{p.parent.name}_{p.stem}"
            tf_dir = TRANSFORM_DIR / folder_name
            oi = folder_to_idx.get(folder_name)
            if oi is None or not tf_dir.exists():
                continue
            for vp in sorted(tf_dir.glob("*.jpg")):
                if vp.stem.startswith("F2"): continue
                all_tf_paths.append(str(vp))
                pos_orig_idx.append(oi)

        pos_orig_idx = np.array(pos_orig_idx, dtype=np.int32)
        print(f"  Transforms: {len(all_tf_paths)}")

        transform_clip = get_clip_embs(all_tf_paths, "CLIP transforms")
        pos_clip = np.sum(orig_clip[pos_orig_idx] * transform_clip, axis=1).astype(np.float32)

        np.savez_compressed(clip_cache, orig_clip=orig_clip, pos_sims=pos_clip,
                           transform_paths=np.array(all_tf_paths),
                           pos_orig_idx=pos_orig_idx)

    print(f"  CLIP pos sims: {pos_clip.shape}, mean={pos_clip.mean():.4f}, std={pos_clip.std():.4f}")

    # ── ViT positive similarities ──
    vit_pos_cache = EMBED_DIR / "vit_pos_sims.npy"
    if vit_pos_cache.exists():
        print("  Loading cached ViT positive similarities...")
        pos_vit = np.load(vit_pos_cache)
    else:
        # Load or compute ViT originals
        vit_orig_cache = EMBED_DIR / "vit_embeddings.npz"
        if vit_orig_cache.exists():
            orig_vit = np.load(vit_orig_cache)["orig_vit"]
        else:
            orig_paths = [str(p) for p in originals]
            orig_vit = get_vit_embs(orig_paths, "ViT originals")
            np.savez_compressed(vit_orig_cache, orig_vit=orig_vit)

        # Compute ViT positive similarities (original vs transforms)
        all_tf_paths = list(clip_cache_data["transform_paths"]) if 'clip_cache_data' in dir() else \
            list(np.load(clip_cache, allow_pickle=True)["transform_paths"])

        # Rebuild transform paths from clip_cache
        data = np.load(clip_cache, allow_pickle=True)
        all_tf_paths = list(data["transform_paths"])
        pos_orig_idx = data["pos_orig_idx"]

        transform_vit = get_vit_embs(all_tf_paths, "ViT transforms")
        pos_vit = np.sum(orig_vit[pos_orig_idx] * transform_vit, axis=1).astype(np.float32)
        np.save(vit_pos_cache, pos_vit)

    print(f"  ViT pos sims: {pos_vit.shape}, mean={pos_vit.mean():.4f}, std={pos_vit.std():.4f}")

    # ── ViT negative similarities ──
    vit_neg_cache = EMBED_DIR / "vit_neg_sims.npy"
    if vit_neg_cache.exists():
        print("  Loading cached ViT negative similarities...")
        vit_neg = np.load(vit_neg_cache)
    else:
        vit_orig_cache = EMBED_DIR / "vit_embeddings.npz"
        orig_vit = np.load(vit_orig_cache)["orig_vit"]
        t0 = time.time()
        sim_matrix = orig_vit @ orig_vit.T
        tri_i, tri_j = np.triu_indices(n_orig, k=1)
        vit_neg = sim_matrix[tri_i, tri_j].astype(np.float32)
        print(f"  ViT neg sims computed in {time.time()-t0:.1f}s")
        np.save(vit_neg_cache, vit_neg)

    print(f"  ViT neg sims: {vit_neg.shape}, mean={vit_neg.mean():.4f}, std={vit_neg.std():.4f}")

    # ── CLIP negative similarities ──
    # Actually, compute these inline like the full_sweep does
    # Normalize orig_clip, compute triu
    if clip_cache.exists():
        data = np.load(clip_cache, allow_pickle=True)
        orig_clip = data["orig_clip"].astype(np.float32)
    orig_clip_norm = orig_clip / np.linalg.norm(orig_clip, axis=1, keepdims=True)
    tri_i, tri_j = np.triu_indices(n_orig, k=1)
    neg_clip = np.sum(orig_clip_norm[tri_i] * orig_clip_norm[tri_j], axis=1).astype(np.float32)

    print(f"  CLIP neg sims: {neg_clip.shape}, mean={neg_clip.mean():.4f}, std={neg_clip.std():.4f}")
    print(f"\n  Positive pairs: {len(pos_clip)}")
    print(f"  Negative pairs: {len(neg_clip)}")

    return pos_clip, pos_vit, neg_clip, vit_neg, originals


def step4_run_sweep(pos_clip, pos_vit, neg_clip, vit_neg, originals):
    """Run full threshold sweep: ViT 0-1, CLIP 0-1, step 0.01."""
    print("\n" + "=" * 60)
    print("STEP 4: Full threshold sweep (10,201 combinations)")
    print("=" * 60)

    VIT_THRESHOLDS = np.arange(0.00, 1.01, 0.01)
    CLIP_THRESHOLDS = np.arange(0.00, 1.01, 0.01)

    def metrics(tp, fp, tn, fn):
        total = tp + fp + tn + fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        return {
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "total_pairs": int(total),
            "accuracy": round((tp + tn) / total, 6) if total else 0.0,
            "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6),
            "specificity": round(specificity, 6),
            "false_positive_rate": round(1 - specificity, 6),
        }

    t0 = time.time()
    rows = []
    for i, vit_thr in enumerate(VIT_THRESHOLDS):
        for j, clip_thr in enumerate(CLIP_THRESHOLDS):
            pos_pred = (pos_clip > clip_thr) & (pos_vit > vit_thr)
            neg_pred = (neg_clip > clip_thr) & (vit_neg > vit_thr)
            tp, fn = int(pos_pred.sum()), int((~pos_pred).sum())
            fp, tn = int(neg_pred.sum()), int((~neg_pred).sum())
            m = metrics(tp, fp, tn, fn)
            rows.append({
                "vit_threshold": round(vit_thr, 2),
                "clip_threshold": round(clip_thr, 2),
                "rule": f"CLIP>{clip_thr:.2f} AND ViT>{vit_thr:.2f}",
                **m,
            })
        if i % 20 == 0:
            print(f"  ViT>{vit_thr:.2f} done ({i+1}/{len(VIT_THRESHOLDS)}), {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  Sweep done in {elapsed:.1f}s")

    # Save CSV
    csv_path = OUT_DIR / "threshold_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "vit_threshold", "clip_threshold", "rule",
            "TP", "FP", "TN", "FN",
            "total_pairs", "accuracy", "precision", "recall", "f1",
            "specificity", "false_positive_rate",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Sorted CSV
    sorted_rows = sorted(rows, key=lambda r: (r["f1"], r["accuracy"]), reverse=True)
    with (OUT_DIR / "threshold_sweep_summary_sorted.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(sorted_rows)

    # Best
    best_f1 = max(rows, key=lambda r: r["f1"])
    best_acc = max(rows, key=lambda r: r["accuracy"])
    top10_f1 = sorted_rows[:10]

    print(f"\n  🏆 Best F1:  {best_f1['rule']}  →  F1={best_f1['f1']:.4f}  Acc={best_f1['accuracy']:.4f}  FP={best_f1['FP']}")
    print(f"  🏆 Best Acc: {best_acc['rule']}  →  Acc={best_acc['accuracy']:.4f}  F1={best_acc['f1']:.4f}")

    print(f"\n  Top 10 by F1:")
    for i, r in enumerate(top10_f1):
        print(f"  {i+1}. {r['rule']:35s} F1={r['f1']:.4f} Acc={r['accuracy']:.4f} FP={r['FP']:,}")

    # Save best results JSON
    best_results = {
        "dataset": "Datasets6 (wikiart 1000 images)",
        "positive_pairs": int(len(pos_clip)),
        "negative_pairs": int(len(neg_clip)),
        "combinations": len(rows),
        "best_f1": {k: best_f1[k] for k in ["vit_threshold","clip_threshold","rule","f1","accuracy","precision","recall","TP","FP","TN","FN"]},
        "best_accuracy": {k: best_acc[k] for k in ["vit_threshold","clip_threshold","rule","f1","accuracy","precision","recall","TP","FP","TN","FN"]},
        "top10_f1": [{k: r[k] for k in ["vit_threshold","clip_threshold","rule","f1","accuracy"]} for r in top10_f1],
        "runtime_seconds": round(elapsed, 1),
    }
    with (OUT_DIR / "best_results.json").open("w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)

    return rows


def step5_generate_charts():
    """Generate comparison charts from sweep results."""
    print("\n" + "=" * 60)
    print("STEP 5: Generate charts")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(OUT_DIR / "threshold_sweep_summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    vit_vals = sorted(set(float(r["vit_threshold"]) for r in rows))
    clip_vals = sorted(set(float(r["clip_threshold"]) for r in rows))
    n_vit, n_clip = len(vit_vals), len(clip_vals)

    f1_mat = np.zeros((n_vit, n_clip))
    fp_mat = np.zeros((n_vit, n_clip))
    for r in rows:
        i = vit_vals.index(float(r["vit_threshold"]))
        j = clip_vals.index(float(r["clip_threshold"]))
        f1_mat[i, j] = float(r["f1"])
        fp_mat[i, j] = float(r["FP"])

    best_f1 = max(rows, key=lambda r: float(r["f1"]))

    # Figure 1: F1 heatmap
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    im1 = axes[0].imshow(f1_mat, aspect='auto', origin='lower',
                          extent=[0, 1, 0, 1], cmap='RdYlGn', vmin=0.0, vmax=1.0)
    axes[0].set_xlabel("CLIP Threshold"); axes[0].set_ylabel("ViT Threshold")
    axes[0].set_title("F1 Score: Datasets6 Full Sweep (ViT 0-1 × CLIP 0-1, step 0.01)")
    axes[0].plot(float(best_f1["clip_threshold"]), float(best_f1["vit_threshold"]),
                 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    axes[0].annotate(f"Best: {best_f1['rule']}\nF1={best_f1['f1']}",
                     (float(best_f1["clip_threshold"])+0.02, float(best_f1["vit_threshold"])),
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    plt.colorbar(im1, ax=axes[0], label="F1 Score", shrink=0.8)

    # Zoom
    f1_zoom = f1_mat[:71, 75:]  # ViT 0-0.7, CLIP 0.75-1.0
    im2 = axes[1].imshow(f1_zoom, aspect='auto', origin='lower',
                          extent=[0.75, 1.0, 0.0, 0.70], cmap='RdYlGn', vmin=0.95, vmax=1.0)
    axes[1].set_xlabel("CLIP Threshold"); axes[1].set_ylabel("ViT Threshold")
    axes[1].set_title("F1 Score Zoom: CLIP 0.75-1.0, ViT 0-0.7")
    axes[1].plot(float(best_f1["clip_threshold"]), float(best_f1["vit_threshold"]),
                 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1.5)
    plt.colorbar(im2, ax=axes[1], label="F1 Score", shrink=0.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "f1_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved f1_heatmap.png")

    # Figure 2: Top 10 ViT curves
    vit_best_f1 = {}
    for r in rows:
        vt = float(r["vit_threshold"])
        f1 = float(r["f1"])
        if vt not in vit_best_f1 or f1 > vit_best_f1[vt]["f1"]:
            vit_best_f1[vt] = {"vit": vt, "clip": float(r["clip_threshold"]), "f1": f1}
    top_vit = sorted(vit_best_f1.values(), key=lambda x: x["f1"], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.plasma(np.linspace(0.1, 0.95, 10))
    for vi, v in enumerate(top_vit):
        vit_idx = min(range(len(vit_vals)), key=lambda x: abs(vit_vals[x] - v["vit"]))
        ax.plot(clip_vals, f1_mat[vit_idx, :], color=colors[vi], linewidth=2,
                label=f"#{vi+1} ViT>{v['vit']:.2f} (best F1={v['f1']:.4f} @ CLIP>{v['clip']:.2f})")
    ax.plot(float(best_f1["clip_threshold"]), float(best_f1["f1"]), 'r*', markersize=25,
            markeredgecolor='darkred', markeredgewidth=2, zorder=10)
    ax.annotate(f"GLOBAL BEST: {best_f1['rule']}\nF1={best_f1['f1']}",
                (float(best_f1["clip_threshold"])+0.01, float(best_f1["f1"])-0.003),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=0.95))
    ax.set_xlabel("CLIP Threshold", fontsize=14); ax.set_ylabel("F1 Score", fontsize=14)
    ax.set_title("F1 Score vs CLIP — Top 10 ViT Thresholds (Datasets6)", fontsize=15)
    ax.legend(loc='lower left', fontsize=9); ax.set_xlim(0.70, 1.0); ax.set_ylim(0.97, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "f1_vs_clip_top10.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved f1_vs_clip_top10.png")

    # Figure 3: FP heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    fp_log = np.log10(fp_mat + 1)
    im = ax.imshow(fp_log, aspect='auto', origin='lower',
                    extent=[0, 1, 0, 1], cmap='RdYlGn_r')
    ax.set_xlabel("CLIP Threshold"); ax.set_ylabel("ViT Threshold")
    ax.set_title("False Positives (log10): Datasets6")
    plt.colorbar(im, ax=ax, label="log10(FP+1)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fp_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved fp_heatmap.png")

    print("\n✓ All charts saved to", OUT_DIR)


def main():
    t_total = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Organize images into style subdirectories
    sources = step1_organize_images()

    # Step 2: Generate transforms
    step2_generate_transforms()

    # Step 3: Compute similarities
    pos_clip, pos_vit, neg_clip, vit_neg, originals = step3_compute_similarities()

    # Step 4: Run full sweep
    rows = step4_run_sweep(pos_clip, pos_vit, neg_clip, vit_neg, originals)

    # Step 5: Generate charts
    step5_generate_charts()

    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"✓ Pipeline complete in {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"✓ Results → {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
