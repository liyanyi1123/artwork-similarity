#!/usr/bin/env python3
"""Resume Ex3 from ViT after CLIP completed. Skips Phase 1 and CLIP."""

from __future__ import annotations
import json, time, sys
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import imagehash
from pdqhash import compute as pdq_compute

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASETS_DIR = BASE_DIR / "datasets2"
EX3_RESULTS = BASE_DIR / "experiments" / "ex3" / "results"
EX3_TABLES = BASE_DIR / "experiments" / "ex3" / "tables"
TIME_DIR = BASE_DIR / "experiments" / "time"
CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 16
CLASSES = ["airplane","bird","car","cat","chair","dog","fish","flower","house","tree"]

VARIANT_NAMES = [
    "F1_Hflip","F2_Vflip",
    "C_low_Crop","C_mid_Crop","C_high_Crop",
    "S1_low_Hshift","S1_mid_Hshift","S1_high_Hshift",
    "S2_low_Vshift","S2_mid_Vshift","S2_high_Vshift",
    "R_low_Rotate","R_mid_Rotate","R_high_Rotate",
    "B_low_Brightness","B_mid_Brightness","B_high_Brightness",
    "V_low_Vignette","V_mid_Vignette","V_high_Vignette",
    "T_low_ColorTemp","T_mid_ColorTemp","T_high_ColorTemp",
    "Sh_low_Sharpen","Sh_mid_Sharpen","Sh_high_Sharpen",
    "K_low_Skew","K_mid_Skew","K_high_Skew",
    "N_low_Noise","N_mid_Noise","N_high_Noise",
]

def load_pil(path): return Image.open(path).convert("RGB")

def gather_state():
    """Reconstruct originals and variant_map from existing files."""
    originals = []
    for cls in CLASSES:
        for fpath in sorted((DATASETS_DIR / cls).glob("*.jpg")):
            originals.append({"path": fpath, "class": cls, "stem": fpath.stem, "name": f"{cls}/{fpath.name}"})

    variant_map = {}
    for img_info in originals:
        vdir = EX3_RESULTS / img_info["stem"]
        variants = sorted(vdir.glob("*.jpg"))
        variant_map[img_info["stem"]] = variants

    print(f"Found {len(originals)} originals, {sum(len(v) for v in variant_map.values())} variants")
    return originals, variant_map

@torch.no_grad()
def extract_embeddings(paths, processor, model, alg_key):
    emb_list = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=f"    {alg_key} embed"):
        batch = paths[i:i+BATCH_SIZE]
        imgs = [load_pil(p) for p in batch]
        if alg_key == "clip":
            inputs = processor(images=imgs, return_tensors="pt", padding=True)
        else:
            inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        if alg_key == "clip":
            outputs = model.get_image_features(**inputs)
            feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        else:
            outputs = model(**inputs)
            feats = outputs.last_hidden_state[:, 0, :]
        if isinstance(feats, tuple): feats = feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        emb_list.append(feats.cpu().numpy())
    return np.concatenate(emb_list, axis=0)

def bits_ahash(pil):
    h = imagehash.average_hash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)
def bits_phash(pil):
    h = imagehash.phash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)
def bits_dhash(pil):
    h = imagehash.dhash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)
def bits_whash(pil):
    h = imagehash.whash(pil, hash_size=8)
    return np.array(h.hash.flatten(), dtype=np.int8)
def bits_pdq(pil):
    arr = np.array(pil)
    b, _ = pdq_compute(arr)
    return b.astype(np.int8)

def run_algo(originals, variant_map, alg_key, alg_label, is_embed, timing):
    print(f"\n{'='*50}\n  {alg_label}\n{'='*50}")
    t0 = time.time()
    M = len(VARIANT_NAMES)
    N = len(originals)

    if is_embed:
        if alg_key == "vit":
            path = str(VIT_MODEL_DIR) if VIT_MODEL_DIR.exists() else "google/vit-base-patch16-224"
            proc = ViTImageProcessor.from_pretrained(path)
            model = ViTModel.from_pretrained(path).to(DEVICE).eval()
        else:
            path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
            proc = CLIPProcessor.from_pretrained(path)
            model = CLIPModel.from_pretrained(path).to(DEVICE).eval()

        all_paths = []
        for info in originals:
            all_paths.append(info["path"])
            all_paths.extend(variant_map[info["stem"]])

        t1 = time.time()
        embs = extract_embeddings(all_paths, proc, model, alg_key)
        t_emb = time.time() - t1
        timing[f"{alg_key}_embed_sec"] = round(t_emb, 1)
        print(f"    Embeddings: {embs.shape} in {t_emb:.1f}s")

        t2 = time.time()
        sim = np.zeros((N, M), dtype=np.float32)
        oi = 0
        for i, info in enumerate(originals):
            oe = embs[oi]
            ve = embs[oi+1:oi+M+1]
            sim[i] = np.clip(np.dot(ve, oe), 0, 1)
            oi += M + 1
        t_sim = time.time() - t2
        timing[f"{alg_key}_sim_sec"] = round(t_sim, 1)
        print(f"    Similarity: {sim.shape} in {t_sim:.1f}s")
    else:
        hash_fn = {"ahash": bits_ahash, "phash": bits_phash, "dhash": bits_dhash, "whash": bits_whash, "pdq": bits_pdq}[alg_key]
        max_bits = 256 if alg_key == "pdq" else 64

        t1 = time.time()
        all_h = {}
        for info in tqdm(originals, desc=f"    {alg_key} originals"):
            pil = load_pil(info["path"])
            all_h[info["stem"] + "_orig"] = hash_fn(pil)
        for stem, vars_ in variant_map.items():
            for vp in vars_:
                pil = load_pil(vp)
                all_h[f"{stem}_{vp.stem}"] = hash_fn(pil)
        t_hash = time.time() - t1
        timing[f"{alg_key}_hash_sec"] = round(t_hash, 1)
        print(f"    Hashes in {t_hash:.1f}s")

        t2 = time.time()
        sim = np.zeros((N, M), dtype=np.float32)
        for i, info in enumerate(originals):
            oh = all_h[info["stem"] + "_orig"]
            for j, vn in enumerate(VARIANT_NAMES):
                vh = all_h[f"{info['stem']}_{vn}"]
                sim[i,j] = 1.0 - np.sum(oh != vh) / max_bits
        t_sim = time.time() - t2
        timing[f"{alg_key}_sim_sec"] = round(t_sim, 1)
        print(f"    Similarity: {sim.shape} in {t_sim:.1f}s")

    elapsed = time.time() - t0
    timing[f"{alg_key}_total_sec"] = round(elapsed, 1)
    print(f"    Total: {elapsed:.1f}s")

    row_names = [info["stem"] for info in originals]
    df = pd.DataFrame(sim, index=row_names, columns=VARIANT_NAMES)
    path = EX3_TABLES / f"{alg_key}_similarity.csv"
    df.to_csv(path, float_format="%.6f")
    print(f"    Saved: {path}")
    return df

def main():
    print("=" * 65)
    print("Ex3 Resume: ViT + 5 Hashes (CLIP already done)")
    print(f"Device: {DEVICE}")
    print("=" * 65)

    originals, variant_map = gather_state()

    # Build timing from scratch (CLIP was done before crash)
    timing_path = TIME_DIR / "ex3_timing.json"
    timing = {
        "experiment": "ex3",
        "num_originals": 1000, "num_variants_per_image": 32,
        "num_transformed_total": 32000, "num_algorithms": 7,
        "device": DEVICE, "batch_size": BATCH_SIZE,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {"generate": {"description": "1000→32000 transformed", "elapsed_sec": 198, "elapsed_fmt": "3m 18s"},
                   "similarity": {}}
    }
    timing["steps"]["generate"] = {"description": "1000→32000 transformed", "elapsed_sec": 190.7, "elapsed_fmt": "3m 10.7s"}

    remaining = [
        ("vit", "ViT (ViT-B/16)", True),
        ("ahash", "aHash (Average)", False),
        ("phash", "pHash (DCT)", False),
        ("dhash", "dHash (Diff)", False),
        ("whash", "wHash (Wavelet)", False),
        ("pdq", "PDQ Hash (256-bit)", False),
    ]

    all_dfs = {}
    # Load existing CLIP df
    all_dfs["clip"] = pd.read_csv(EX3_TABLES / "clip_similarity.csv", index_col=0)

    for alg_key, alg_label, is_embed in remaining:
        at = {}
        df = run_algo(originals, variant_map, alg_key, alg_label, is_embed, at)
        all_dfs[alg_key] = df
        timing["steps"]["similarity"][alg_key] = at

    # Summary
    print("\n" + "─" * 65)
    print("Summary Statistics")
    print("─" * 65)
    algos_all = [("clip","CLIP (ViT-B/32)"),("vit","ViT (ViT-B/16)"),
                 ("ahash","aHash"),("phash","pHash"),("dhash","dHash"),
                 ("whash","wHash"),("pdq","PDQ")]
    rows = []
    for k, lbl in algos_all:
        df = all_dfs[k]
        v = df.values.flatten()
        rows.append({"algorithm":lbl,"mean":round(float(np.mean(v)),6),
                     "std":round(float(np.std(v)),6),"min":round(float(np.min(v)),6),
                     "max":round(float(np.max(v)),6),"median":round(float(np.median(v)),6)})
    sdf = pd.DataFrame(rows)
    sdf.to_csv(EX3_TABLES / "summary_stats.csv", index=False, float_format="%.6f")
    print(sdf.to_string(index=False))

    total = sum(s["total_sec"] for s in timing["steps"]["similarity"].values()) + timing["steps"]["generate"]["elapsed_sec"]
    timing["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timing["total_elapsed_sec"] = round(total, 1)
    timing["total_elapsed_fmt"] = f"{int(total//60)}m {total%60:.1f}s"
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    print(f"\nTiming updated: {timing_path}")
    print(f"Total: {timing['total_elapsed_fmt']}")
    print("Done!")

if __name__ == "__main__":
    main()
