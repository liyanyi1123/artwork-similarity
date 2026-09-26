"""
Threshold Evaluation @ 0.87 — CLIP + pHash fallback for positives
=================================================================
Positive pairs: each original image vs its 31 non-F2 variations → 3100 pairs (label=1)
Negative pairs: all pairs of different original images → 4950 pairs (label=0)

Metrics: TP, TN, FP, FN, Recall, Precision, Accuracy, F1
"""

import json
import time
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import imagehash
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
project_root = Path(__file__).parents[3]
sys.path.append(str(project_root))

from src.utils.file_manager import ProjectPaths, save_dataframe, save_json

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = project_root
ARCHIVE_DIR = BASE_DIR / "Archive"
OUTPUT_IMAGE_DIR = BASE_DIR / "ex1_2" / "OutputImage"

# 使用路径管理器
paths = ProjectPaths(BASE_DIR)
THRESHOLD_EVAL_DIR = paths.threshold_evaluation

CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"

THRESHOLD = 0.87

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ===================================================================
# 1. Build the dataset
# ===================================================================
def build_evaluation_pairs():
    """
    Returns:
      positive_pairs: list of (original_path, variation_path, original_name)
      negative_pairs: list of (original_path_A, original_path_B, name_A, name_B)
    """
    # Map stem -> archive path
    archive_images = {}
    for fpath in ARCHIVE_DIR.rglob("*"):
        if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
            archive_images[fpath.stem] = fpath

    # Map output folder name -> original path
    output_folders = [d for d in OUTPUT_IMAGE_DIR.iterdir() if d.is_dir()]

    positive_pairs = []
    skipped = 0

    for folder in sorted(output_folders):
        stem = folder.name
        if stem not in archive_images:
            skipped += 1
            continue
        original_path = archive_images[stem]
        # Find all variation images in this folder, exclude F2 (vertical flip)
        for var_path in sorted(folder.iterdir()):
            if var_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            if var_path.stem.startswith("F2_"):  # exclude vertical flip
                continue
            positive_pairs.append({
                "original_path": original_path,
                "variation_path": var_path,
                "original_name": f"{original_path.parent.name}/{original_path.name}",
                "variation_name": f"{folder.name}/{var_path.name}",
            })

    # Negative pairs: all different-original pairs from the 100 originals
    originals = sorted(archive_images.values(), key=lambda p: (p.parent.name, p.name))
    negative_pairs = []
    for i in range(len(originals)):
        for j in range(i + 1, len(originals)):
            negative_pairs.append({
                "path_A": originals[i],
                "path_B": originals[j],
                "name_A": f"{originals[i].parent.name}/{originals[i].name}",
                "name_B": f"{originals[j].parent.name}/{originals[j].name}",
            })

    return positive_pairs, negative_pairs, skipped


# ===================================================================
# 2. Load models
# ===================================================================
def load_models():
    clip_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"

    print(f"Loading CLIP from: {clip_path}")
    clip_proc = CLIPProcessor.from_pretrained(clip_path)
    clip_model = CLIPModel.from_pretrained(clip_path).to(DEVICE).eval()

    return clip_proc, clip_model


# ===================================================================
# 3. Compute similarities
# ===================================================================
@torch.no_grad()
def compute_clip_similarity(image_paths_A: list[Path], image_paths_B: list[Path],
                            processor, model) -> np.ndarray:
    """Compute cosine similarity between pairs (A_i, B_i). Returns (N,) array."""
    sims = []
    for i in tqdm(range(0, len(image_paths_A), BATCH_SIZE), desc="  CLIP pairs"):
        batch_A = [Image.open(p).convert("RGB") for p in image_paths_A[i:i+BATCH_SIZE]]
        batch_B = [Image.open(p).convert("RGB") for p in image_paths_B[i:i+BATCH_SIZE]]
        inputs_A = processor(images=batch_A, return_tensors="pt", padding=True)
        inputs_B = processor(images=batch_B, return_tensors="pt", padding=True)
        inputs_A = {k: v.to(DEVICE) for k, v in inputs_A.items()}
        inputs_B = {k: v.to(DEVICE) for k, v in inputs_B.items()}
        feat_A = model.get_image_features(**inputs_A)
        feat_B = model.get_image_features(**inputs_B)
        fa = feat_A.pooler_output if hasattr(feat_A, "pooler_output") else feat_A
        fb = feat_B.pooler_output if hasattr(feat_B, "pooler_output") else feat_B
        if isinstance(fa, tuple): fa = fa[0]
        if isinstance(fb, tuple): fb = fb[0]
        fa = fa / fa.norm(dim=-1, keepdim=True)
        fb = fb / fb.norm(dim=-1, keepdim=True)
        cos_sim = (fa * fb).sum(dim=-1).cpu().numpy()
        sims.append(cos_sim)
    return np.concatenate(sims)


def compute_phash_similarity(image_path_a: Path, image_path_b: Path) -> float:
    """Convert pHash Hamming distance to a normalized similarity in [0, 1]."""
    with Image.open(image_path_a) as opened_a, Image.open(image_path_b) as opened_b:
        image_a = opened_a.convert("RGB")
        image_b = opened_b.convert("RGB")
        hash_a = imagehash.phash(image_a)
        hash_b = imagehash.phash(image_b)
    return 1.0 - (hash_a - hash_b) / len(hash_a.hash.ravel())


def compute_positive_predictions_with_fallback(
    positive_pairs: list[dict], clip_sims: np.ndarray, threshold: float
):
    """
    Positive prediction rule:
      1. CLIP similarity >= threshold => TP
      2. otherwise compute pHash similarity; if >= threshold => TP
      3. otherwise FN
    """
    final_positive = clip_sims >= threshold
    phash_sims = np.full(len(clip_sims), np.nan, dtype=float)
    fallback_mask = clip_sims < threshold

    for idx in tqdm(np.where(fallback_mask)[0], desc="  pHash fallback"):
        pair = positive_pairs[idx]
        phash_sims[idx] = compute_phash_similarity(pair["original_path"], pair["variation_path"])
        if phash_sims[idx] >= threshold:
            final_positive[idx] = True

    return final_positive, phash_sims, fallback_mask


# ===================================================================
# 4. Evaluate metrics
# ===================================================================
def evaluate(
    positive_sims: np.ndarray,
    negative_sims: np.ndarray,
    threshold: float,
    model_name: str,
    positive_predictions: np.ndarray | None = None,
    phash_sims: np.ndarray | None = None,
):
    """Compute TP, TN, FP, FN, Recall, Precision, Accuracy, F1."""
    if positive_predictions is None:
        positive_predictions = positive_sims >= threshold

    TP = int(np.sum(positive_predictions))
    FN = int(len(positive_predictions) - TP)
    # Negative pairs: should be < threshold (label=0)
    TN = int(np.sum(negative_sims < threshold))
    FP = int(np.sum(negative_sims >= threshold))

    total = TP + TN + FP + FN
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy  = (TP + TN) / total if total > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'─' * 50}")
    print(f"{model_name} — Threshold = {threshold}")
    print(f"{'─' * 50}")
    print(f"  Positive samples: {len(positive_sims)}  |  Negative samples: {len(negative_sims)}")
    print(f"  Distribution of positive similarities:")
    print(f"    mean={np.mean(positive_sims):.4f}  std={np.std(positive_sims):.4f}  "
          f"min={np.min(positive_sims):.4f}  max={np.max(positive_sims):.4f}")
    pos_below = np.sum(positive_sims < threshold)
    print(f"    < {threshold}: {pos_below}/{len(positive_sims)} ({100*pos_below/len(positive_sims):.1f}%)")
    if phash_sims is not None:
        fallback_count = int(np.sum(~np.isnan(phash_sims)))
        fallback_rescued = int(np.sum((~np.isnan(phash_sims)) & positive_predictions))
        print(f"  pHash fallback checked: {fallback_count}")
        print(f"  pHash fallback rescued: {fallback_rescued}")
    print(f"  Distribution of negative similarities:")
    print(f"    mean={np.mean(negative_sims):.4f}  std={np.std(negative_sims):.4f}  "
          f"min={np.min(negative_sims):.4f}  max={np.max(negative_sims):.4f}")
    neg_above = np.sum(negative_sims >= threshold)
    print(f"    >= {threshold}: {neg_above}/{len(negative_sims)} ({100*neg_above/len(negative_sims):.1f}%)")
    print(f"\n  ┌──────────┬─────────┐")
    print(f"  │   TP     │  {TP:>5d}   │")
    print(f"  │   TN     │  {TN:>5d}   │")
    print(f"  │   FP     │  {FP:>5d}   │")
    print(f"  │   FN     │  {FN:>5d}   │")
    print(f"  ├──────────┼─────────┤")
    print(f"  │ Total    │  {total:>5d}   │")
    print(f"  └──────────┴─────────┘")
    print(f"\n  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    result = {
        "model": model_name,
        "threshold": threshold,
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "positive_mean": round(float(np.mean(positive_sims)), 4),
        "positive_std": round(float(np.std(positive_sims)), 4),
        "negative_mean": round(float(np.mean(negative_sims)), 4),
        "negative_std": round(float(np.std(negative_sims)), 4),
    }
    if phash_sims is not None:
        valid_phash = phash_sims[~np.isnan(phash_sims)]
        result.update({
            "phash_fallback_checked": int(valid_phash.size),
            "phash_fallback_rescued": int(np.sum((~np.isnan(phash_sims)) & positive_predictions)),
        })
        if valid_phash.size:
            result["phash_fallback_mean"] = round(float(np.mean(valid_phash)), 4)
    return result
 
# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 60)
    print(f"Threshold Evaluation @ {THRESHOLD}")
    print("=" * 60)

    # 1. Build dataset
    print("\n[1/4] Building evaluation pairs ...")
    pos_pairs, neg_pairs, skipped = build_evaluation_pairs()
    print(f"  Positive pairs:  {len(pos_pairs)}  (expected 3100)")
    print(f"  Negative pairs:  {len(neg_pairs)}  (expected 4950 = C(100,2))")
    if skipped:
        print(f"  Skipped folders (no original): {skipped}")

    # 2. Load models
    print("\n[2/4] Loading models ...")
    clip_proc, clip_model = load_models()

    # 3. Compute similarities
    print("\n[3/4] Computing similarities ...")

    # --- Positive pairs (variation vs original) ---
    print(f"\n  --- Positive pairs ({len(pos_pairs)}) ---")
    pos_orig = [p["original_path"] for p in pos_pairs]
    pos_vari = [p["variation_path"] for p in pos_pairs]

    t0 = time.time()
    clip_pos_sims = compute_clip_similarity(pos_orig, pos_vari, clip_proc, clip_model)
    print(f"  CLIP positive: {time.time() - t0:.1f}s")
    clip_positive_predictions, phash_sims, _ = compute_positive_predictions_with_fallback(
        pos_pairs, clip_pos_sims, THRESHOLD
    )

    # --- Negative pairs (original vs original) ---
    print(f"\n  --- Negative pairs ({len(neg_pairs)}) ---")
    neg_A = [p["path_A"] for p in neg_pairs]
    neg_B = [p["path_B"] for p in neg_pairs]

    t0 = time.time()
    clip_neg_sims = compute_clip_similarity(neg_A, neg_B, clip_proc, clip_model)
    print(f"  CLIP negative: {time.time() - t0:.1f}s")

    # 4. Evaluate
    print("\n[4/4] Evaluating ...")
    clip_result = evaluate(
        clip_pos_sims,
        clip_neg_sims,
        THRESHOLD,
        "CLIP (ViT-B/32) + pHash fallback",
        positive_predictions=clip_positive_predictions,
        phash_sims=phash_sims,
    )

    # Save detailed results
    save_json([clip_result], "threshold_evaluation", THRESHOLD_EVAL_DIR / "results")

    # Save all similarity values for further analysis
    df_pos = pd.DataFrame({
        "type": "positive",
        "model": "clip",
        "original": [p["original_name"] for p in pos_pairs],
        "variation": [p["variation_name"] for p in pos_pairs],
        "clip_similarity": clip_pos_sims,
        "phash_similarity": phash_sims,
        "final_prediction": clip_positive_predictions,
    })
    save_dataframe(df_pos, "threshold_positive_pairs", THRESHOLD_EVAL_DIR / "data", "csv")

    df_neg = pd.DataFrame({
        "type": "negative",
        "model": "clip",
        "image_A": [p["name_A"] for p in neg_pairs],
        "image_B": [p["name_B"] for p in neg_pairs],
        "similarity": clip_neg_sims,
    })
    save_dataframe(df_neg, "threshold_negative_pairs", THRESHOLD_EVAL_DIR / "data", "csv")

    print(f"\n{'=' * 60}")
    print(f"Results saved to:")
    print(f"  {THRESHOLD_EVAL_DIR / 'results'}")
    print(f"  {THRESHOLD_EVAL_DIR / 'data'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
