#!/usr/bin/env python3
"""
Ex3 Pipeline: 12 transformations × 1000 images × 7 similarity algorithms.

Phase 1: Generate 32 variants per image → 32,000 images
Phase 2: Compute similarity (original vs each variant) using 7 algorithms
Phase 3: Save tables to experiments/ex3/tables/
Phase 4: Record timing to experiments/time/ex3_timing.json
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

import imagehash
from pdqhash import compute as pdq_compute

# ── Paths ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "datasets2"
EX3_RESULTS = BASE_DIR / "experiments" / "ex3" / "results"
EX3_TABLES = BASE_DIR / "experiments" / "ex3" / "tables"
TIME_DIR = BASE_DIR / "experiments" / "time"

CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16

# ── 10 classes ────────────────────────────────────────
CLASSES = ["airplane", "bird", "car", "cat", "chair", "dog", "fish", "flower", "house", "tree"]

# ── 12 Transform operations ───────────────────────────

def op_F1(img):
    return cv2.flip(img, 1)

def op_F2(img):
    return cv2.flip(img, 0)

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
    reduction, exponent = 0.25 * strength, 1.5 * strength
    mask = np.clip(1.0 - dist * reduction, 0.0, 1.0)
    mask = np.power(mask, exponent)
    return (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)

def op_T(img, strength=1.0):
    r = img.astype(np.float32)
    r[:, :, 2] = np.clip(r[:, :, 2] * (1.0 + 0.06 * strength), 0, 255)
    r[:, :, 1] = np.clip(r[:, :, 1] * (1.0 + 0.02 * strength), 0, 255)
    r[:, :, 0] = np.clip(r[:, :, 0] * (1.0 - 0.05 * strength), 0, 255)
    return r.astype(np.uint8)

def op_Sh(img, strength=1.0):
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    identity = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
    w_sh = min(0.4 * strength, 0.95)
    kernel = w_sh * sharpen + (1.0 - w_sh) * identity
    return cv2.filter2D(img, -1, kernel)

def op_K(img, strength=1.0):
    h, w = img.shape[:2]
    margin = 0.03 * strength
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[0,0],[w,h*margin],[w,h*(1-margin)],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_N(img, strength=1.0):
    rng = np.random.default_rng(42)
    noise_std = 3.0 * strength
    noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

# Operation definitions: (code, func, name, settings)
# settings: list of (strength, label) or None for non-tunable
OPERATIONS = [
    ("F1", op_F1, "Hflip",     None),
    ("F2", op_F2, "Vflip",     None),
    ("C",  op_C,  "Crop",      [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("S1", op_S1, "Hshift",    [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("S2", op_S2, "Vshift",    [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("R",  op_R,  "Rotate",    [(1.5,"low"),(3.0,"mid"),(6.0,"high")]),
    ("B",  op_B,  "Brightness",[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("V",  op_V,  "Vignette",  [(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("T",  op_T,  "ColorTemp", [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("Sh", op_Sh, "Sharpen",   [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("K",  op_K,  "Skew",      [(0.33,"low"),(1.0,"mid"),(1.67,"high")]),
    ("N",  op_N,  "Noise",     [(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
]

# Build variant name list (32 entries)
VARIANT_NAMES = []
for code, func, name, settings in OPERATIONS:
    if settings is None:
        VARIANT_NAMES.append(f"{code}_{name}")
    else:
        for strength, label in settings:
            VARIANT_NAMES.append(f"{code}_{label}_{name}")


# ═══════════════════════════════════════════════════════
# Phase 1: Image Generation
# ═══════════════════════════════════════════════════════

def gather_originals() -> list[dict]:
    """Collect all 1000 images from datasets2/."""
    images = []
    for cls in CLASSES:
        cls_dir = DATASETS_DIR / cls
        for fpath in sorted(cls_dir.glob("*.jpg")):
            images.append({
                "path": fpath,
                "class": cls,
                "name": f"{cls}/{fpath.name}",
                "stem": fpath.stem,
            })
    return images


def generate_variants(originals: list[dict]) -> dict:
    """Generate 32 variants per original. Returns mapping: stem -> list of variant paths."""
    EX3_RESULTS.mkdir(parents=True, exist_ok=True)
    variant_map: dict[str, list[Path]] = {}

    print(f"Generating {len(originals)} × {len(VARIANT_NAMES)} = "
          f"{len(originals) * len(VARIANT_NAMES)} images...")

    for img_info in tqdm(originals, desc="  Transform"):
        img = cv2.imread(str(img_info["path"]))
        if img is None:
            continue

        out_dir = EX3_RESULTS / img_info["stem"]
        out_dir.mkdir(parents=True, exist_ok=True)
        variants = []

        vidx = 0
        for code, func, name, settings in OPERATIONS:
            if settings is None:
                result = func(img)
                fname = f"{VARIANT_NAMES[vidx]}.jpg"
                out_path = out_dir / fname
                cv2.imwrite(str(out_path), result)
                variants.append(out_path)
                vidx += 1
            else:
                for strength, label in settings:
                    result = func(img, strength=strength)
                    fname = f"{VARIANT_NAMES[vidx]}.jpg"
                    out_path = out_dir / fname
                    cv2.imwrite(str(out_path), result)
                    variants.append(out_path)
                    vidx += 1

        variant_map[img_info["stem"]] = variants

    return variant_map


# ═══════════════════════════════════════════════════════
# Phase 2: Similarity Computation
# ═══════════════════════════════════════════════════════

def load_pil(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def compute_similarity_table(
    originals: list[dict],
    variant_map: dict[str, list[Path]],
    alg_key: str,
    alg_label: str,
    is_embed: bool,
    timings: dict,
) -> pd.DataFrame:
    """Compute similarity for one algorithm. Returns DataFrame: rows=originals, cols=variants."""
    print(f"\n  [{alg_label}]")

    t_start = time.time()

    if is_embed:
        # ── CLIP or ViT embedding ──
        if alg_key == "clip":
            model_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_path)
            model = CLIPModel.from_pretrained(model_path).to(DEVICE).eval()
        else:
            model_path = str(VIT_MODEL_DIR) if VIT_MODEL_DIR.exists() else "google/vit-base-patch16-224"
            processor = ViTImageProcessor.from_pretrained(model_path)
            model = ViTModel.from_pretrained(model_path).to(DEVICE).eval()

        # Collect all images: originals + all variants
        all_paths: list[Path] = []
        for img_info in originals:
            all_paths.append(img_info["path"])
            for vp in variant_map[img_info["stem"]]:
                all_paths.append(vp)

        t_embed_start = time.time()
        embeddings = _extract_embeddings(all_paths, processor, model, alg_key)
        t_embed = time.time() - t_embed_start
        timings[f"{alg_key}_embed_sec"] = round(t_embed, 1)
        print(f"    Embeddings: {embeddings.shape} in {t_embed:.1f}s")

        # Compute cosine similarity: each original vs its 32 variants
        t_sim_start = time.time()
        N = len(originals)
        M = len(VARIANT_NAMES)
        sim_matrix = np.zeros((N, M), dtype=np.float32)

        # embeddings layout: [orig0, var0_0..var0_31, orig1, var1_0..var1_31, ...]
        # So orig[i] is at position i * (M + 1), variants at i*(M+1)+1 .. i*(M+1)+M
        orig_idx = 0
        for i, img_info in enumerate(originals):
            orig_emb = embeddings[orig_idx]
            var_start = orig_idx + 1
            var_embs = embeddings[var_start:var_start + M]
            sims = np.dot(var_embs, orig_emb)  # cosine (already normalized)
            sim_matrix[i] = np.clip(sims, 0.0, 1.0)
            orig_idx += (M + 1)

        t_sim = time.time() - t_sim_start
        timings[f"{alg_key}_sim_sec"] = round(t_sim, 1)
        print(f"    Similarity matrix: {sim_matrix.shape} in {t_sim:.1f}s")

    else:
        # ── Perceptual hash ──
        hash_fn = {
            "ahash": lambda pil: _ahash_bits(pil),
            "phash": lambda pil: _phash_bits(pil),
            "dhash": lambda pil: _dhash_bits(pil),
            "whash": lambda pil: _whash_bits(pil),
            "pdq":   lambda pil: _pdq_bits(pil),
        }[alg_key]
        max_bits = 256 if alg_key == "pdq" else 64

        # Pre-compute hashes for all images
        t_hash_start = time.time()
        all_hashes: dict[str, np.ndarray] = {}

        for img_info in tqdm(originals, desc=f"    {alg_key} originals"):
            pil = load_pil(img_info["path"])
            all_hashes[img_info["stem"] + "_orig"] = hash_fn(pil)

        for stem, variants in variant_map.items():
            for vp in variants:
                pil = load_pil(vp)
                all_hashes[f"{stem}_{vp.stem}"] = hash_fn(pil)

        t_hash = time.time() - t_hash_start
        timings[f"{alg_key}_hash_sec"] = round(t_hash, 1)
        print(f"    Hashes computed in {t_hash:.1f}s")

        # Build similarity matrix
        t_sim_start = time.time()
        N = len(originals)
        M = len(VARIANT_NAMES)
        sim_matrix = np.zeros((N, M), dtype=np.float32)

        for i, img_info in enumerate(originals):
            orig_hash = all_hashes[img_info["stem"] + "_orig"]
            for j, vname in enumerate(VARIANT_NAMES):
                vhash = all_hashes[f"{img_info['stem']}_{vname}"]
                hamming = np.sum(orig_hash != vhash)
                sim_matrix[i, j] = 1.0 - hamming / max_bits

        t_sim = time.time() - t_sim_start
        timings[f"{alg_key}_sim_sec"] = round(t_sim, 1)
        print(f"    Similarity matrix: {sim_matrix.shape} in {t_sim:.1f}s")

    elapsed = time.time() - t_start
    timings[f"{alg_key}_total_sec"] = round(elapsed, 1)
    print(f"    Total: {elapsed:.1f}s")

    # Build DataFrame
    row_names = [img["stem"] for img in originals]
    df = pd.DataFrame(sim_matrix, index=row_names, columns=VARIANT_NAMES)
    return df


@torch.no_grad()
def _extract_embeddings(paths: list[Path], processor, model, alg_key: str) -> np.ndarray:
    """Batch-extract embeddings for a list of image paths."""
    emb_list = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=f"    {alg_key} embed"):
        batch_paths = paths[i:i + BATCH_SIZE]
        pil_imgs = [load_pil(p) for p in batch_paths]
        if alg_key == "clip":
            inputs = processor(images=pil_imgs, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=pil_imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        if alg_key == "clip":
            outputs = model.get_image_features(**inputs)
            feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        else:
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]

        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        emb_list.append(feats.cpu().numpy())
    return np.concatenate(emb_list, axis=0)


def _ahash_bits(pil: Image.Image) -> np.ndarray:
    h = imagehash.average_hash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)

def _phash_bits(pil: Image.Image) -> np.ndarray:
    h = imagehash.phash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)

def _dhash_bits(pil: Image.Image) -> np.ndarray:
    h = imagehash.dhash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)

def _whash_bits(pil: Image.Image) -> np.ndarray:
    h = imagehash.whash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)

def _pdq_bits(pil: Image.Image) -> np.ndarray:
    arr = np.array(pil)
    bits, _ = pdq_compute(arr)
    return bits.astype(np.int8)


# ═══════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════

ALGORITHMS = [
    ("clip",  "CLIP (ViT-B/32)",  True),
    ("vit",   "ViT  (ViT-B/16)",  True),
    ("ahash", "aHash (Average)",   False),
    ("phash", "pHash (DCT)",       False),
    ("dhash", "dHash (Diff)",      False),
    ("whash", "wHash (Wavelet)",   False),
    ("pdq",   "PDQ Hash (256-bit)",False),
]


def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timing: dict = {
        "experiment": "ex3",
        "num_originals": 1000,
        "num_variants_per_image": 32,
        "num_transformed_total": 32000,
        "num_algorithms": 7,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "start_time": now_str,
        "steps": {},
    }

    print("=" * 65)
    print("Ex3: 12 Transforms × 1000 Images × 7 Similarity Algorithms")
    print(f"Device: {DEVICE}  |  Batch: {BATCH_SIZE}")
    print("=" * 65)

    # ── Phase 1: Generate ──────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 1: Image Generation")
    print("─" * 65)

    originals = gather_originals()
    print(f"Found {len(originals)} images in {len(CLASSES)} classes")

    t1_start = time.time()
    variant_map = generate_variants(originals)
    t1_elapsed = time.time() - t1_start
    timing["steps"]["generate"] = {
        "description": "1000 originals → 32,000 transformed",
        "elapsed_sec": round(t1_elapsed, 1),
        "elapsed_fmt": f"{int(t1_elapsed // 60)}m {t1_elapsed % 60:.1f}s",
    }
    print(f"  Phase 1 done: {t1_elapsed:.1f}s ({t1_elapsed/60:.1f} min)")

    # Verify variant count
    for stem, paths in variant_map.items():
        if len(paths) != 32:
            print(f"  WARNING: {stem} has {len(paths)} variants (expected 32)")

    # ── Phase 2: Similarity ────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 2: Similarity Computation (7 algorithms)")
    print("─" * 65)

    EX3_TABLES.mkdir(parents=True, exist_ok=True)
    timing["steps"]["similarity"] = {}

    all_dfs = {}

    for alg_key, alg_label, is_embed in ALGORITHMS:
        print(f"\n{'='*50}")
        print(f"  {alg_label}")
        print(f"{'='*50}")

        algo_timing: dict = {}
        df = compute_similarity_table(
            originals, variant_map, alg_key, alg_label, is_embed, algo_timing
        )
        all_dfs[alg_key] = df

        # Save CSV
        csv_path = EX3_TABLES / f"{alg_key}_similarity.csv"
        df.to_csv(csv_path, float_format="%.6f")
        print(f"    Saved: {csv_path}")

        timing["steps"]["similarity"][alg_key] = algo_timing

    # ── Summary stats ──────────────────────────────
    print("\n" + "─" * 65)
    print("PHASE 3: Summary Statistics")
    print("─" * 65)

    summary_rows = []
    for alg_key, alg_label, _ in ALGORITHMS:
        df = all_dfs[alg_key]
        all_vals = df.values.flatten()
        summary_rows.append({
            "algorithm": alg_label,
            "mean": round(float(np.mean(all_vals)), 6),
            "std": round(float(np.std(all_vals)), 6),
            "min": round(float(np.min(all_vals)), 6),
            "max": round(float(np.max(all_vals)), 6),
            "median": round(float(np.median(all_vals)), 6),
            "p25": round(float(np.percentile(all_vals, 25)), 6),
            "p75": round(float(np.percentile(all_vals, 75)), 6),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = EX3_TABLES / "summary_stats.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    print(f"Saved: {summary_path}")
    print(summary_df.to_string(index=False))

    # ── Timing ─────────────────────────────────────
    total_elapsed = time.time() - t1_start
    timing["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timing["total_elapsed_sec"] = round(total_elapsed, 1)
    timing["total_elapsed_fmt"] = f"{int(total_elapsed // 60)}m {total_elapsed % 60:.1f}s"

    TIME_DIR.mkdir(parents=True, exist_ok=True)
    timing_path = TIME_DIR / "ex3_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    print(f"\nTiming saved: {timing_path}")
    print(f"Total elapsed: {timing['total_elapsed_fmt']}")

    print(f"\n{'=' * 65}")
    print("Ex3 Pipeline Complete!")
    print(f"  Results: {EX3_RESULTS}")
    print(f"  Tables:  {EX3_TABLES}")
    print(f"  Timing:  {timing_path}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
