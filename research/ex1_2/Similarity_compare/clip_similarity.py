"""
CLIP, ViT & Perceptual Hash Image Similarity Comparison
========================================================
Compares all 100 images from the Archive directory using 7 algorithms:

  Deep embeddings:
    1. CLIP  (ViT-B/32)  — 512-dim joint vision-language
    2. ViT   (ViT-B/16)  — 768-dim [CLS] token

  Perceptual hashes:
    3. aHash — Average Hash          (64-bit)
    4. pHash — Perceptual / DCT Hash (64-bit)
    5. dHash — Difference Hash       (64-bit)
    6. wHash — Wavelet Hash          (64-bit)
    7. PDQ   — Facebook PDQ Hash     (256-bit)

For hashes: similarity = 1 − Hamming_distance / max_bits
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

import imagehash
from pdqhash import compute as pdq_compute

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent.parent  # ex1_2/clip_similarity_compare -> repo root
ARCHIVE_DIR = BASE_DIR / "Archive"
OUTPUT_DIR = Path(__file__).parent

CLIP_MODEL_DIR = BASE_DIR / "clip-vit-base-patch32"
VIT_MODEL_DIR  = BASE_DIR / "vit-base-patch16-224"

MODEL_PATHS = {
    "clip": str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32",
    "vit":  str(VIT_MODEL_DIR)  if VIT_MODEL_DIR.exists()  else "google/vit-base-patch16-224",
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# All algorithms: (key, label, is_embedding_based)
ALGORITHMS = [
    ("clip",  "CLIP (ViT-B/32)",  True),
    ("vit",   "ViT  (ViT-B/16)",  True),
    ("ahash", "aHash (Average)",   False),
    ("phash", "pHash (DCT)",       False),
    ("dhash", "dHash (Diff)",      False),
    ("whash", "wHash (Wavelet)",   False),
    ("pdq",   "PDQ Hash (256-bit)",False),
]


# ===================================================================
# 1. Gather all images
# ===================================================================
def gather_images(archive_dir: Path) -> list[dict]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
    images = []
    for fpath in sorted(archive_dir.rglob("*")):
        if fpath.suffix.lower() in extensions:
            images.append({
                "path": fpath,
                "category": fpath.parent.name,
                "name": f"{fpath.parent.name}/{fpath.name}",
            })
    return images


def load_pil_images(images: list[dict]) -> list[Image.Image]:
    """Pre-load all images as PIL RGB for reuse across hash methods."""
    return [Image.open(img["path"]).convert("RGB") for img in images]


# ===================================================================
# 2. Embedding-based models (CLIP, ViT)
# ===================================================================
def load_clip_model(model_path: str, device: str):
    print(f"    Loading from: {model_path}")
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path).to(device).eval()
    return processor, model

def load_vit_model(model_path: str, device: str):
    print(f"    Loading from: {model_path}")
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTModel.from_pretrained(model_path).to(device).eval()
    return processor, model

@torch.no_grad()
def extract_clip_embeddings(images: list[dict], processor, model, device: str,
                            batch_size: int) -> np.ndarray:
    embeddings_list = []
    for i in tqdm(range(0, len(images), batch_size), desc="    CLIP embedding"):
        batch = images[i:i + batch_size]
        pil_images = [Image.open(item["path"]).convert("RGB") for item in batch]
        inputs = processor(images=pil_images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.get_image_features(**inputs)
        feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        if isinstance(feats, tuple):
            feats = feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeddings_list.append(feats.cpu().numpy())
    return np.concatenate(embeddings_list, axis=0)

@torch.no_grad()
def extract_vit_embeddings(images: list[dict], processor, model, device: str,
                           batch_size: int) -> np.ndarray:
    embeddings_list = []
    for i in tqdm(range(0, len(images), batch_size), desc="    ViT  embedding"):
        batch = images[i:i + batch_size]
        pil_images = [Image.open(item["path"]).convert("RGB") for item in batch]
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
        embeddings_list.append(cls_emb.cpu().numpy())
    return np.concatenate(embeddings_list, axis=0)


# ===================================================================
# 3. Perceptual hash computation
# ===================================================================
def compute_hash_similarity_matrix(hash_vectors: np.ndarray, max_bits: int) -> np.ndarray:
    """
    Compute pairwise similarity from hash bit vectors.
    similarity = 1 − Hamming_distance / max_bits
    hash_vectors: (N, max_bits) bool or int array
    """
    N = len(hash_vectors)
    # Broadcast XOR to compute Hamming distance
    # For each pair (i,j): hamming = sum of XORed bits
    hamming = np.zeros((N, N), dtype=np.float32)
    # Vectorized: for each bit position, accumulate differences
    for b in range(max_bits):
        col = hash_vectors[:, b].astype(np.int8).reshape(N, 1)
        hamming += (col != col.T).astype(np.float32)
    sim = 1.0 - hamming / max_bits
    return np.clip(sim, 0.0, 1.0)


def compute_hashes_ahash(pil_images: list[Image.Image]) -> np.ndarray:
    desc = "    aHash compute"
    hashes = []
    for img in tqdm(pil_images, desc=desc):
        h = imagehash.average_hash(img, hash_size=8)
        bits = np.array(h.hash.flatten(), dtype=np.int8)  # (64,)
        hashes.append(bits)
    return np.stack(hashes, axis=0)

def compute_hashes_phash(pil_images: list[Image.Image]) -> np.ndarray:
    desc = "    pHash compute"
    hashes = []
    for i, img in enumerate(tqdm(pil_images, desc=desc)):
        h = imagehash.phash(img, hash_size=8)
        bits = np.array(h.hash.flatten(), dtype=np.int8)
        hashes.append(bits)
    return np.stack(hashes, axis=0)

def compute_hashes_dhash(pil_images: list[Image.Image]) -> np.ndarray:
    desc = "    dHash compute"
    hashes = []
    for img in tqdm(pil_images, desc=desc):
        h = imagehash.dhash(img, hash_size=8)
        bits = np.array(h.hash.flatten(), dtype=np.int8)
        hashes.append(bits)
    return np.stack(hashes, axis=0)

def compute_hashes_whash(pil_images: list[Image.Image]) -> np.ndarray:
    desc = "    wHash compute"
    hashes = []
    for img in tqdm(pil_images, desc=desc):
        h = imagehash.whash(img, hash_size=8)
        bits = np.array(h.hash.flatten(), dtype=np.int8)
        hashes.append(bits)
    return np.stack(hashes, axis=0)

def compute_hashes_pdq(pil_images: list[Image.Image]) -> np.ndarray:
    """PDQ hash: 256-bit, returns (N, 256) bool array."""
    desc = "    PDQ  compute"
    hashes = []
    for img in tqdm(pil_images, desc=desc):
        arr = np.array(img)
        hash_bits, quality = pdq_compute(arr)
        hashes.append(hash_bits.astype(np.int8))
    return np.stack(hashes, axis=0)


# ===================================================================
# 4. Similarity matrix (cosine for embeddings)
# ===================================================================
def compute_cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim = embeddings @ embeddings.T
    return np.clip(sim, 0.0, 1.0)


# ===================================================================
# 5. Save outputs
# ===================================================================
def save_results(images: list[dict], sim_matrix: np.ndarray, output_dir: Path,
                 prefix: str, label: str):
    N = len(images)
    names = [img["name"] for img in images]
    categories = [img["category"] for img in images]

    # --- NumPy matrix ---
    np.save(output_dir / f"{prefix}_similarity_matrix.npy", sim_matrix)

    # --- CSV matrix ---
    pd.DataFrame(sim_matrix, index=names, columns=names) \
      .to_csv(output_dir / f"{prefix}_similarity_matrix.csv", float_format="%.6f")

    # --- Top-200 pairs ---
    triu_ix = np.triu_indices(N, k=1)
    triu_vals = sim_matrix[triu_ix]
    top_k = min(200, len(triu_vals))
    top_idx = np.argpartition(triu_vals, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(triu_vals[top_idx])[::-1]]

    pairs = []
    for idx in top_idx:
        i, j = triu_ix[0][idx], triu_ix[1][idx]
        pairs.append({
            "rank": len(pairs) + 1,
            "image_A": names[i], "image_B": names[j],
            "category_A": categories[i], "category_B": categories[j],
            "same_category": categories[i] == categories[j],
            "similarity": round(float(sim_matrix[i, j]), 6),
        })
    pd.DataFrame(pairs).to_csv(output_dir / f"{prefix}_top_similar_pairs.csv", index=False)

    # --- Category-level ---
    unique_cats = sorted(set(categories))
    cat_to_idx = {c: [] for c in unique_cats}
    for idx, cat in enumerate(categories):
        cat_to_idx[cat].append(idx)

    cat_rows = []
    for ca in unique_cats:
        row = {"category": ca}
        for cb in unique_cats:
            ia, ib = cat_to_idx[ca], cat_to_idx[cb]
            if ca == cb:
                vals = [sim_matrix[a, b] for a in ia for b in ib if a != b]
            else:
                vals = [sim_matrix[a, b] for a in ia for b in ib]
            row[cb] = round(float(np.mean(vals)), 6) if vals else 0.0
        cat_rows.append(row)
    pd.DataFrame(cat_rows).to_csv(output_dir / f"{prefix}_category_similarity.csv", index=False)

    # --- Heatmap ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(28, 12))

        ax = axes[0]
        im = ax.imshow(sim_matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto", origin="lower")
        ax.set_title(f"{label} — Similarity Matrix (100×100)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Image Index"); ax.set_ylabel("Image Index")
        plt.colorbar(im, ax=ax, label="Similarity")

        ax2 = axes[1]
        cat_df = pd.DataFrame(cat_rows).set_index("category")
        cat_df = cat_df[unique_cats]
        im2 = ax2.imshow(cat_df.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto", origin="lower")
        plt.colorbar(im2, ax=ax2, label="Mean Similarity")
        for i in range(len(cat_df)):
            for j in range(len(cat_df.columns)):
                ax2.text(j, i, f"{cat_df.values[i, j]:.3f}", ha="center", va="center",
                         fontsize=7, color="white" if cat_df.values[i, j] > 0.55 else "black")
        ax2.set_title(f"{label} — Category-Level", fontsize=14, fontweight="bold")
        ax2.set_xticks(range(len(unique_cats)))
        ax2.set_xticklabels(unique_cats, rotation=45, ha="right", fontsize=8)
        ax2.set_yticks(range(len(unique_cats)))
        ax2.set_yticklabels(unique_cats, fontsize=8)

        plt.tight_layout()
        fig.savefig(output_dir / f"{prefix}_similarity_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        pass  # silently skip if no matplotlib

    # --- Stats ---
    all_vals = sim_matrix[triu_ix]
    stats = {
        "algorithm": label,
        "num_images": N,
        "num_categories": len(unique_cats),
        "similarity_stats": {
            "mean":   round(float(np.mean(all_vals)), 6),
            "std":    round(float(np.std(all_vals)), 6),
            "min":    round(float(np.min(all_vals)), 6),
            "max":    round(float(np.max(all_vals)), 6),
            "median": round(float(np.median(all_vals)), 6),
            "p25":    round(float(np.percentile(all_vals, 25)), 6),
            "p75":    round(float(np.percentile(all_vals, 75)), 6),
            "p95":    round(float(np.percentile(all_vals, 95)), 6),
            "p99":    round(float(np.percentile(all_vals, 99)), 6),
        },
        "categories": unique_cats,
        "images_per_category": {c: len(cat_to_idx[c]) for c in unique_cats},
    }
    with open(output_dir / f"{prefix}_embedding_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    s = stats["similarity_stats"]
    print(f"    mean={s['mean']:.4f}  std={s['std']:.4f}  "
          f"min={s['min']:.4f}  max={s['max']:.4f}  median={s['median']:.4f}")

    return triu_vals, triu_ix


# ===================================================================
# 6. Unified pipeline
# ===================================================================
def run_pipeline(alg_key: str, label: str, is_embed: bool,
                 images: list[dict], pil_images: list[Image.Image],
                 counter: tuple[int, int]):
    """Run similarity pipeline for one algorithm. Returns (triu_values, triu_indices)."""
    i, total = counter
    print(f"\n{'─' * 60}")
    print(f"[{i}/{total}] {label}")
    print(f"{'─' * 60}")

    t0 = time.time()

    if is_embed:
        # ── Embedding-based: CLIP or ViT ──
        if alg_key == "clip":
            processor, model = load_clip_model(MODEL_PATHS[alg_key], DEVICE)
            embeddings = extract_clip_embeddings(images, processor, model, DEVICE, BATCH_SIZE)
        else:
            processor, model = load_vit_model(MODEL_PATHS[alg_key], DEVICE)
            embeddings = extract_vit_embeddings(images, processor, model, DEVICE, BATCH_SIZE)
        print(f"    Embeddings: {embeddings.shape}  |  Time: {time.time() - t0:.1f}s")
        t1 = time.time()
        sim_matrix = compute_cosine_similarity_matrix(embeddings)
        print(f"    Cosine sim matrix: {time.time() - t1:.2f}s")
    else:
        # ── Hash-based ──
        hash_fn = {
            "ahash": compute_hashes_ahash,
            "phash": compute_hashes_phash,
            "dhash": compute_hashes_dhash,
            "whash": compute_hashes_whash,
            "pdq":   compute_hashes_pdq,
        }[alg_key]
        hash_vecs = hash_fn(pil_images)
        max_bits = hash_vecs.shape[1]
        print(f"    Hash vectors: {hash_vecs.shape}  ({max_bits}-bit)  "
              f"|  Time: {time.time() - t0:.1f}s")
        t1 = time.time()
        sim_matrix = compute_hash_similarity_matrix(hash_vecs, max_bits)
        print(f"    Hamming→sim matrix: {time.time() - t1:.2f}s")

    print(f"    Saving {alg_key}_* ...")
    triu_vals, triu_ix = save_results(images, sim_matrix, OUTPUT_DIR, alg_key, label)

    return triu_vals, triu_ix


# ===================================================================
# 7. Main
# ===================================================================
def main():
    print("=" * 60)
    print("Image Similarity — CLIP, ViT & 5 Perceptual Hashes")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Gather images once
    images = gather_images(ARCHIVE_DIR)
    print(f"\nFound {len(images)} images in {len(set(i['category'] for i in images))} categories:")
    for cat in sorted(set(i["category"] for i in images)):
        count = sum(1 for i in images if i["category"] == cat)
        print(f"  {cat}: {count} images")

    # Pre-load PIL images for hash methods
    print("\nPre-loading all 100 images as PIL ...")
    pil_images = load_pil_images(images)

    # Run all algorithms
    total = len(ALGORITHMS)
    all_results = {}  # alg_key -> (triu_vals, triu_ix)
    for idx, (key, label, is_embed) in enumerate(ALGORITHMS, 1):
        triu_vals, triu_ix = run_pipeline(key, label, is_embed, images, pil_images, (idx, total))
        all_results[key] = triu_vals

    # ── Cross-method correlation matrix ──
    print(f"\n{'─' * 60}")
    print("Cross-Method Pearson Correlation Matrix")
    print(f"{'─' * 60}")
    keys = [k for k, _, _ in ALGORITHMS]
    corr_mat = np.zeros((len(keys), len(keys)))
    for i, k1 in enumerate(keys):
        for j, k2 in enumerate(keys):
            corr_mat[i, j] = np.corrcoef(all_results[k1], all_results[k2])[0, 1]

    corr_df = pd.DataFrame(corr_mat, index=keys, columns=keys)
    corr_df.to_csv(OUTPUT_DIR / "cross_method_correlation.csv", float_format="%.4f")
    print(corr_df.to_string(float_format=lambda x: f"{x:.4f}"))

    # Heatmap for correlation matrix
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
        plt.colorbar(im, ax=ax, label="Pearson r")
        for i in range(len(keys)):
            for j in range(len(keys)):
                ax.text(j, i, f"{corr_mat[i, j]:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if corr_mat[i, j] < 0.4 else "black")
        ax.set_title("Cross-Method Similarity Correlation", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "cross_method_correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("\n  ✓ cross_method_correlation.png")
    except ImportError:
        pass

    # ── Summary table ──
    print(f"\n{'─' * 60}")
    print("Summary Statistics")
    print(f"{'─' * 60}")
    print(f"{'Algorithm':<22s} {'Mean':>7s} {'Std':>7s} {'Median':>7s} {'Min':>7s} {'Max':>7s}")
    print("-" * 57)
    for key, label, _ in ALGORITHMS:
        with open(OUTPUT_DIR / f"{key}_embedding_stats.json") as f:
            s = json.load(f)["similarity_stats"]
        print(f"{label:<22s} {s['mean']:>7.4f} {s['std']:>7.4f} "
              f"{s['median']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f}")

    print(f"\n{'=' * 60}")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"7 algorithms × 6 output files + cross_method_correlation = done ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
