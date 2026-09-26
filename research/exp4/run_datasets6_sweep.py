#!/usr/bin/env python3
"""
Compute CLIP/ViT similarities for Datasets6 and run full threshold sweep (0-1, 0.01).
Requires transforms already generated in Transform/datasets6/.
"""

import csv, json, time, sys
from pathlib import Path
import numpy as np
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

VIT_THRESHOLDS = np.arange(0.00, 1.01, 0.01)
CLIP_THRESHOLDS = np.arange(0.00, 1.01, 0.01)


def main():
    t_total = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Setup device ──
    import torch
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"  # Force CPU — MPS hangs on model loading
    BATCH_SIZE = 64  # Smaller batch for CPU
    print(f"Device: {device}, Batch: {BATCH_SIZE}")

    # ── Load models ──
    from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
    from PIL import Image

    MODEL_DIR = BASE_DIR / "models"
    CLIP_DIR = MODEL_DIR / "clip-vit-base-patch32"
    VIT_DIR = MODEL_DIR / "vit-base-patch16-224"

    clip_path = str(CLIP_DIR) if CLIP_DIR.exists() else "openai/clip-vit-base-patch32"
    vit_path = str(VIT_DIR) if VIT_DIR.exists() else "google/vit-base-patch16-224"

    print(f"CLIP: {clip_path}")
    print(f"ViT:  {vit_path}")

    lp = CLIP_DIR.exists()
    clip_proc = CLIPProcessor.from_pretrained(clip_path, local_files_only=lp)
    clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=lp).to(device).eval()
    vp = VIT_DIR.exists()
    vit_proc = ViTImageProcessor.from_pretrained(vit_path, local_files_only=vp)
    vit_model = ViTModel.from_pretrained(vit_path, local_files_only=vp).to(device).eval()

    # ── Gather originals ──
    originals = []
    for style in ART_STYLES:
        d = DATASETS6 / style
        if d.exists():
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in IMG_EXTS:
                    originals.append(p)
    n_orig = len(originals)
    print(f"\nOriginals: {n_orig}")

    folder_to_idx = {}
    for i, p in enumerate(originals):
        folder_to_idx[f"{p.parent.name}_{p.stem}"] = i

    # Check transforms
    tf_missing = sum(1 for p in originals
                     if not (TRANSFORM_DIR / f"{p.parent.name}_{p.stem}").exists()
                     or len(list((TRANSFORM_DIR / f"{p.parent.name}_{p.stem}").glob("*.jpg"))) < 32)
    if tf_missing > 0:
        print(f"WARNING: {tf_missing} images missing transforms. Run generate_datasets6_transforms.py first.")
        if tf_missing > 100:
            print("Too many missing transforms, aborting.")
            sys.exit(1)

    # ── Embedding helpers ──
    @torch.no_grad()
    def get_embs(paths, proc, model, desc, model_type="clip"):
        embs = []
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc):
            batch = paths[i:i+BATCH_SIZE]
            imgs = []
            valid_paths = []
            for p in batch:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                except Exception:
                    pass  # Skip corrupt/truncated images
            if not imgs:
                continue
            if model_type == "clip":
                inputs = proc(images=imgs, return_tensors="pt", padding=True)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats = model.get_image_features(**inputs)
                if hasattr(feats, "pooler_output"): feats = feats.pooler_output
                if isinstance(feats, tuple): feats = feats[0]
            else:
                inputs = proc(images=imgs, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                feats = model(**inputs).last_hidden_state[:, 0, :]
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu().numpy())
        return np.concatenate(embs, axis=0).astype(np.float32)

    # ── CLIP: originals + positive similarities ──
    clip_cache = EMBED_DIR / "clip_embeddings.npz"
    if clip_cache.exists():
        print("\nLoading cached CLIP embeddings...")
        data = np.load(clip_cache, allow_pickle=True)
        pos_clip = data["pos_sims"].astype(np.float32)
        orig_clip = data["orig_clip"].astype(np.float32)
        pos_orig_idx = data["pos_orig_idx"]
    else:
        print("\nComputing CLIP embeddings for originals...")
        orig_paths = [str(p) for p in originals]
        orig_clip = get_embs(orig_paths, clip_proc, clip_model, "CLIP originals", "clip")

        print("Collecting transform paths...")
        pos_orig_idx_list = []
        all_tf_paths = []
        for p in tqdm(originals, desc="  Scan transforms"):
            folder_name = f"{p.parent.name}_{p.stem}"
            tf_dir = TRANSFORM_DIR / folder_name
            oi = folder_to_idx.get(folder_name)
            if oi is None or not tf_dir.exists():
                continue
            for vp in sorted(tf_dir.glob("*.jpg")):
                if vp.stem.startswith("F2"): continue
                all_tf_paths.append(str(vp))
                pos_orig_idx_list.append(oi)

        pos_orig_idx = np.array(pos_orig_idx_list, dtype=np.int32)
        print(f"Transforms: {len(all_tf_paths)}")

        print("Computing CLIP embeddings for transforms...")
        transform_clip = get_embs(all_tf_paths, clip_proc, clip_model, "CLIP transforms", "clip")
        pos_clip = np.sum(orig_clip[pos_orig_idx] * transform_clip, axis=1).astype(np.float32)

        np.savez_compressed(clip_cache, orig_clip=orig_clip, pos_sims=pos_clip,
                           transform_paths=np.array(all_tf_paths),
                           pos_orig_idx=pos_orig_idx)

    print(f"  CLIP pos sims: {pos_clip.shape}, mean={pos_clip.mean():.4f}")

    # ── ViT: originals + positive + negative similarities ──
    vit_cache = EMBED_DIR / "vit_embeddings.npz"
    if vit_cache.exists():
        print("\nLoading cached ViT embeddings...")
        orig_vit = np.load(vit_cache)["orig_vit"].astype(np.float32)
    else:
        print("\nComputing ViT embeddings for originals...")
        orig_paths = [str(p) for p in originals]
        orig_vit = get_embs(orig_paths, vit_proc, vit_model, "ViT originals", "vit")
        np.savez_compressed(vit_cache, orig_vit=orig_vit)

    # ViT positive similarities
    vit_pos_cache = EMBED_DIR / "vit_pos_sims.npy"
    if vit_pos_cache.exists():
        print("Loading cached ViT positive similarities...")
        pos_vit = np.load(vit_pos_cache).astype(np.float32)
    else:
        data = np.load(clip_cache, allow_pickle=True)
        all_tf_paths = list(data["transform_paths"])
        pos_orig_idx = data["pos_orig_idx"]

        print("Computing ViT embeddings for transforms...")
        transform_vit = get_embs(all_tf_paths, vit_proc, vit_model, "ViT transforms", "vit")
        pos_vit = np.sum(orig_vit[pos_orig_idx] * transform_vit, axis=1).astype(np.float32)
        np.save(vit_pos_cache, pos_vit)

    print(f"  ViT pos sims: {pos_vit.shape}, mean={pos_vit.mean():.4f}")

    # ViT negative similarities
    vit_neg_cache = EMBED_DIR / "vit_neg_sims.npy"
    if vit_neg_cache.exists():
        print("\nLoading cached ViT negative similarities...")
        vit_neg = np.load(vit_neg_cache).astype(np.float32)
    else:
        print("\nComputing ViT negative similarities (all pairs)...")
        t0 = time.time()
        sim_matrix = orig_vit @ orig_vit.T
        tri_i, tri_j = np.triu_indices(n_orig, k=1)
        vit_neg = sim_matrix[tri_i, tri_j].astype(np.float32)
        print(f"  Done in {time.time()-t0:.1f}s")
        np.save(vit_neg_cache, vit_neg)

    print(f"  ViT neg sims: {vit_neg.shape}, mean={vit_neg.mean():.4f}")

    # CLIP negative similarities
    print("\nComputing CLIP negative similarities...")
    orig_clip_norm = orig_clip / np.linalg.norm(orig_clip, axis=1, keepdims=True)
    tri_i, tri_j = np.triu_indices(n_orig, k=1)
    neg_clip = np.sum(orig_clip_norm[tri_i] * orig_clip_norm[tri_j], axis=1).astype(np.float32)
    print(f"  CLIP neg sims: {neg_clip.shape}, mean={neg_clip.mean():.4f}")

    print(f"\nPositive pairs: {len(pos_clip):,}")
    print(f"Negative pairs: {len(neg_clip):,}")

    # ── Full Sweep ──
    print(f"\n{'='*60}")
    print(f"Running full sweep: ViT {len(VIT_THRESHOLDS)} × CLIP {len(CLIP_THRESHOLDS)} = {len(VIT_THRESHOLDS)*len(CLIP_THRESHOLDS):,} combinations")
    print(f"{'='*60}")

    def metrics(tp, fp, tn, fn):
        total = tp + fp + tn + fn
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        return {"TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn),
                "total_pairs":int(total),
                "accuracy":round((tp+tn)/total,6) if total else 0.0,
                "precision":round(p,6),"recall":round(r,6),"f1":round(f1,6),
                "specificity":round(spec,6),
                "false_positive_rate":round(1-spec,6)}

    t0 = time.time()
    rows = []
    for i, vt in enumerate(VIT_THRESHOLDS):
        for j, ct in enumerate(CLIP_THRESHOLDS):
            pos_pred = (pos_clip > ct) & (pos_vit > vt)
            neg_pred = (neg_clip > ct) & (vit_neg > vt)
            tp, fn = int(pos_pred.sum()), int((~pos_pred).sum())
            fp, tn = int(neg_pred.sum()), int((~neg_pred).sum())
            m = metrics(tp, fp, tn, fn)
            rows.append({"vit_threshold":round(vt,2),"clip_threshold":round(ct,2),
                        "rule":f"CLIP>{ct:.2f} AND ViT>{vt:.2f}",**m})
        if i % 20 == 0:
            print(f"  ViT>{vt:.2f} done ({i+1}/{len(VIT_THRESHOLDS)}), {time.time()-t0:.1f}s")

    elapsed = time.time() - t0
    print(f"  Sweep done in {elapsed:.1f}s")

    # ── Save results ──
    csv_path = OUT_DIR / "threshold_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vit_threshold","clip_threshold","rule",
               "TP","FP","TN","FN","total_pairs","accuracy","precision","recall","f1",
               "specificity","false_positive_rate"])
        w.writeheader(); w.writerows(rows)

    sorted_rows = sorted(rows, key=lambda r: (r["f1"], r["accuracy"]), reverse=True)
    with (OUT_DIR / "threshold_sweep_summary_sorted.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(sorted_rows)

    best_f1 = max(rows, key=lambda r: r["f1"])
    best_acc = max(rows, key=lambda r: r["accuracy"])

    print(f"\n  🏆 Best F1:  {best_f1['rule']}  →  F1={best_f1['f1']:.4f}  Acc={best_f1['accuracy']:.4f}  FP={best_f1['FP']}")
    print(f"  🏆 Best Acc: {best_acc['rule']}  →  Acc={best_acc['accuracy']:.4f}  F1={best_acc['f1']:.4f}")

    print("\n  Top 10 by F1:")
    for i, r in enumerate(sorted_rows[:10]):
        print(f"  {i+1}. {r['rule']:35s} F1={r['f1']:.4f} FP={r['FP']:,}")

    # Best results JSON
    import json as _json
    best_results = {
        "dataset": "Datasets6 (wikiart 1000 images)",
        "positive_pairs": int(len(pos_clip)),
        "negative_pairs": int(len(neg_clip)),
        "combinations": len(rows),
        "best_f1": {k: best_f1[k] for k in ["vit_threshold","clip_threshold","rule","f1","accuracy","precision","recall","TP","FP","TN","FN"]},
        "best_accuracy": {k: best_acc[k] for k in ["vit_threshold","clip_threshold","rule","f1","accuracy","precision","recall","TP","FP","TN","FN"]},
        "top10_f1": [{k: r[k] for k in ["vit_threshold","clip_threshold","rule","f1","accuracy"]} for r in sorted_rows[:10]],
        "runtime_seconds": round(elapsed, 1),
    }
    with (OUT_DIR / "best_results.json").open("w", encoding="utf-8") as f:
        _json.dump(best_results, f, indent=2)

    # ── Generate charts ──
    print("\nGenerating charts...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vit_vals = sorted(set(float(r["vit_threshold"]) for r in rows))
    clip_vals = sorted(set(float(r["clip_threshold"]) for r in rows))
    n_vit, n_clip = len(vit_vals), len(clip_vals)
    f1_mat = np.zeros((n_vit, n_clip))
    for r in rows:
        i = vit_vals.index(float(r["vit_threshold"]))
        j = clip_vals.index(float(r["clip_threshold"]))
        f1_mat[i, j] = float(r["f1"])

    # F1 heatmap
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    im1 = axes[0].imshow(f1_mat, aspect='auto', origin='lower', extent=[0,1,0,1], cmap='RdYlGn', vmin=0, vmax=1)
    axes[0].set_xlabel("CLIP Threshold"); axes[0].set_ylabel("ViT Threshold")
    axes[0].set_title("F1 Score: Datasets6 Full Sweep")
    axes[0].plot(float(best_f1["clip_threshold"]), float(best_f1["vit_threshold"]),
                 'r*', markersize=15, markeredgecolor='white')
    axes[0].annotate(f"Best: {best_f1['rule']}\nF1={best_f1['f1']}",
                     (float(best_f1["clip_threshold"])+.02, float(best_f1["vit_threshold"])),
                     fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=.8))
    plt.colorbar(im1, ax=axes[0], label="F1", shrink=.8)

    f1z = f1_mat[:71, 75:]
    im2 = axes[1].imshow(f1z, aspect='auto', origin='lower', extent=[.75,1,0,.7], cmap='RdYlGn', vmin=.95, vmax=1)
    axes[1].set_xlabel("CLIP Threshold"); axes[1].set_ylabel("ViT Threshold")
    axes[1].set_title("F1 Zoom: CLIP 0.75-1.0, ViT 0-0.7")
    axes[1].plot(float(best_f1["clip_threshold"]), float(best_f1["vit_threshold"]),
                 'r*', markersize=15, markeredgecolor='white')
    plt.colorbar(im2, ax=axes[1], label="F1", shrink=.8)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"f1_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved f1_heatmap.png")

    # Top 10 ViT curves
    vit_best = {}
    for r in rows:
        vt = float(r["vit_threshold"]); f1 = float(r["f1"])
        if vt not in vit_best or f1 > vit_best[vt]["f1"]:
            vit_best[vt] = {"vit":vt,"clip":float(r["clip_threshold"]),"f1":f1}
    top_vit = sorted(vit_best.values(), key=lambda x: x["f1"], reverse=True)[:10]

    fig, ax = plt.subplots(figsize=(16, 8))
    colors = plt.cm.plasma(np.linspace(.1,.95,10))
    for vi, v in enumerate(top_vit):
        vi_idx = min(range(len(vit_vals)), key=lambda x: abs(vit_vals[x]-v["vit"]))
        ax.plot(clip_vals, f1_mat[vi_idx,:], color=colors[vi], lw=2,
                label=f"#{vi+1} ViT>{v['vit']:.2f} (best F1={v['f1']:.4f} @ CLIP>{v['clip']:.2f})")
    ax.plot(float(best_f1["clip_threshold"]), float(best_f1["f1"]), 'r*', ms=25,
            markeredgecolor='darkred', mew=2, zorder=10)
    ax.annotate(f"GLOBAL BEST: {best_f1['rule']}\nF1={best_f1['f1']}",
                (float(best_f1["clip_threshold"])+.01, float(best_f1["f1"])-.003),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='red', alpha=.95))
    ax.set_xlabel("CLIP Threshold", fontsize=14); ax.set_ylabel("F1 Score", fontsize=14)
    ax.set_title("F1 vs CLIP — Top 10 ViT Thresholds (Datasets6)", fontsize=15)
    ax.legend(loc='lower left', fontsize=9); ax.set_xlim(.7, 1); ax.set_ylim(.97, 1)
    ax.grid(True, alpha=.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR/"f1_vs_clip_top10.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved f1_vs_clip_top10.png")

    total_t = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"✓ Done in {total_t:.0f}s ({total_t/60:.1f} min)")
    print(f"✓ Results → {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
