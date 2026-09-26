#!/usr/bin/env python3
"""Full sweep: ViT 0.00–1.00 × CLIP 0.00–1.00, step 0.01 (10,201 combinations).
Reuses precomputed similarities — no model reloading."""

from __future__ import annotations
import csv, json, time, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
EMB_DIR = BASE / "analysis/threshold_eval_datasets5/embeddings"
EXP4_RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "full_sweep"

STYLES = ["art_nouveau", "baroque", "expressionism", "impressionism",
          "post_impressionism", "realism", "renaissance", "romanticism",
          "surrealism", "ukiyo_e"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

VIT_THRESHOLDS = np.arange(0.00, 1.01, 0.01)
CLIP_THRESHOLDS = np.arange(0.00, 1.01, 0.01)


def gather_originals() -> list[Path]:
    paths = []
    for s in STYLES:
        d = BASE / "Datasets5" / s
        paths.extend(p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS)
    return paths


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    total = tp + fp + tn + fn
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "total_pairs": int(total),
        "positive_pairs": int(tp + fn),
        "negative_pairs": int(tn + fp),
        "accuracy": round((tp + tn) / total, 6) if total else 0.0,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "specificity": round(specificity, 6),
        "false_positive_rate": round(1 - specificity, 6),
    }


def evaluate_combination(clip_thr: float, vit_thr: float,
                         pos_clip: np.ndarray, pos_vit: np.ndarray,
                         neg_clip: np.ndarray, vit_neg: np.ndarray) -> dict:
    pos_pred = (pos_clip > clip_thr) & (pos_vit > vit_thr)
    neg_pred = (neg_clip > clip_thr) & (vit_neg > vit_thr)
    tp = int(pos_pred.sum())
    fn = int((~pos_pred).sum())
    fp = int(neg_pred.sum())
    tn = int((~neg_pred).sum())
    return metrics(tp, fp, tn, fn)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ── Load precomputed similarities ──
    print("Loading precomputed similarities...")
    clip_d = np.load(EMB_DIR / "clip_embeddings.npz", allow_pickle=False)
    vit_d = np.load(EMB_DIR / "vit_embeddings.npz", allow_pickle=False)
    pos_clip = clip_d["pos_sims"].astype(np.float32)
    pos_vit = np.load(EXP4_RES / "positive_vit_similarities.npy").astype(np.float32)
    pos_orig_idx = clip_d["pos_orig_idx"].astype(np.int32)
    transform_paths = clip_d["transform_paths"]
    orig_clip = clip_d["orig_clip"].astype(np.float32)
    vit_neg = np.load(EMB_DIR / "vit_neg_sims.npy").astype(np.float32)
    originals = gather_originals()

    # Compute negative CLIP similarities
    print("Computing negative CLIP similarities...")
    orig_clip = orig_clip / np.linalg.norm(orig_clip, axis=1, keepdims=True)
    tri_i, tri_j = np.triu_indices(len(originals), k=1)
    neg_clip = np.sum(orig_clip[tri_i] * orig_clip[tri_j], axis=1).astype(np.float32)

    n_pos = len(pos_clip)
    n_neg = len(neg_clip)
    n_combos = len(VIT_THRESHOLDS) * len(CLIP_THRESHOLDS)
    print(f"Positive pairs: {n_pos:,}  Negative pairs: {n_neg:,}")
    print(f"ViT: {len(VIT_THRESHOLDS)} thresholds [0.00–1.00]")
    print(f"CLIP: {len(CLIP_THRESHOLDS)} thresholds [0.00–1.00]")
    print(f"Total combinations: {n_combos:,}")

    # ── Pre-compute TP/FN/FP/TN for each single-model threshold ──
    print("\nPre-computing per-threshold results...")
    # For each vit threshold, how many pos/neg pass
    vit_pos_pass = np.array([(pos_vit > t).sum() for t in VIT_THRESHOLDS], dtype=np.int32)
    vit_neg_pass = np.array([(vit_neg > t).sum() for t in VIT_THRESHOLDS], dtype=np.int32)
    clip_pos_pass = np.array([(pos_clip > t).sum() for t in CLIP_THRESHOLDS], dtype=np.int32)
    clip_neg_pass = np.array([(neg_clip > t).sum() for t in CLIP_THRESHOLDS], dtype=np.int32)

    # ── Evaluate all combinations (fast: just combining precomputed counts per AND logic) ──
    print("Evaluating all 10,201 combinations...")
    rows = []
    best_f1 = None
    best_acc = None

    for i, vit_thr in enumerate(VIT_THRESHOLDS):
        for j, clip_thr in enumerate(CLIP_THRESHOLDS):
            # For AND logic: both must pass
            pos_pred = (pos_clip > clip_thr) & (pos_vit > vit_thr)
            neg_pred = (neg_clip > clip_thr) & (vit_neg > vit_thr)
            tp = int(pos_pred.sum())
            fn = int((~pos_pred).sum())
            fp = int(neg_pred.sum())
            tn = int((~neg_pred).sum())
            m = metrics(tp, fp, tn, fn)

            row = {
                "vit_threshold": round(vit_thr, 2),
                "clip_threshold": round(clip_thr, 2),
                "rule": f"CLIP>{clip_thr:.2f} AND ViT>{vit_thr:.2f}",
                **m,
            }
            rows.append(row)

            if best_f1 is None or m["f1"] > best_f1["f1"]:
                best_f1 = row
            if best_acc is None or m["accuracy"] > best_acc["accuracy"]:
                best_acc = row

        # Progress per ViT row
        if i % 10 == 0:
            elapsed = time.time() - t0
            print(f"  ViT>{vit_thr:.2f} done ({i+1}/{len(VIT_THRESHOLDS)}), {elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    print(f"Evaluation done in {elapsed:.1f}s")

    # ── Save summary CSV ──
    csv_path = OUT / "threshold_sweep_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "vit_threshold", "clip_threshold", "rule",
            "TP", "FP", "TN", "FN",
            "total_pairs", "positive_pairs", "negative_pairs",
            "accuracy", "precision", "recall", "f1",
            "specificity", "false_positive_rate",
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV → {csv_path}")

    # ── Best results ──
    print(f"\n{'='*70}")
    print(f"🏆 Best by F1:  {best_f1['rule']}")
    print(f"   F1={best_f1['f1']:.4f}  Acc={best_f1['accuracy']:.4f}  "
          f"Prec={best_f1['precision']:.4f}  Rec={best_f1['recall']:.4f}")
    print(f"   TP={best_f1['TP']:,}  FP={best_f1['FP']:,}  "
          f"TN={best_f1['TN']:,}  FN={best_f1['FN']:,}")
    print(f"\n🏆 Best by Acc: {best_acc['rule']}")
    print(f"   Acc={best_acc['accuracy']:.4f}  F1={best_acc['f1']:.4f}  "
          f"Prec={best_acc['precision']:.4f}  Rec={best_acc['recall']:.4f}")

    # ── Compare with original ViT 0.50-0.55 / CLIP 0.80-0.85 region ──
    print(f"\n{'='*70}")
    print("Comparison: ViT 0.50-0.55 vs CLIP 0.80-0.85 region")
    print("=" * 70)
    region_rows = [r for r in rows
                   if 0.50 <= r["vit_threshold"] <= 0.55
                   and 0.80 <= r["clip_threshold"] <= 0.85]
    for r in region_rows:
        print(f"  {r['rule']:35s} F1={r['f1']:.4f} Acc={r['accuracy']:.4f} "
              f"FP={r['FP']:,}")

    # ── Find good operating points across the full range ──
    print(f"\n{'='*70}")
    print("Top 10 by F1 across full range:")
    print("=" * 70)
    top10_f1 = sorted(rows, key=lambda r: r["f1"], reverse=True)[:10]
    for r in top10_f1:
        print(f"  {r['rule']:35s} F1={r['f1']:.4f} Acc={r['accuracy']:.4f} "
              f"FP={r['FP']:,}  Prec={r['precision']:.4f}")

    # ── Save sorted summary for easy browsing ──
    sorted_csv = OUT / "threshold_sweep_summary_sorted.csv"
    sorted_rows = sorted(rows, key=lambda r: (r["f1"], r["accuracy"]), reverse=True)
    with sorted_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted_rows)
    print(f"\nSorted CSV → {sorted_csv}")

    # ── Save best combinations per metric ──
    best_results = {
        "best_f1": {k: best_f1[k] for k in ["vit_threshold", "clip_threshold", "rule", "f1", "accuracy", "precision", "recall", "specificity", "TP", "FP", "TN", "FN"]},
        "best_accuracy": {k: best_acc[k] for k in ["vit_threshold", "clip_threshold", "rule", "f1", "accuracy", "precision", "recall", "specificity", "TP", "FP", "TN", "FN"]},
        "top10_f1": [{k: r[k] for k in ["vit_threshold", "clip_threshold", "rule", "f1", "accuracy"]} for r in top10_f1],
        "sweep_params": {"vit_range": "0.00–1.00", "clip_range": "0.00–1.00", "step": 0.01, "combinations": n_combos},
        "runtime_seconds": round(elapsed, 1),
    }
    with (OUT / "best_results.json").open("w", encoding="utf-8") as f:
        json.dump(best_results, f, indent=2)
    print(f"Best results JSON → {OUT / 'best_results.json'}")

    print(f"\n✓ Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
