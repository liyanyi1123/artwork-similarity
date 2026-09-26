#!/usr/bin/env python3
"""Sweep CLIP and ViT thresholds with AND logic, comparing all combinations.

ViT thresholds:  0.50, 0.51, 0.52, 0.53, 0.54, 0.55
CLIP thresholds: 0.80, 0.81, 0.82, 0.83, 0.84, 0.85
Logic: CLIP > clip_thr AND ViT > vit_thr → similar

Reuses precomputed similarities from exp4 — no model reloading.
"""

from __future__ import annotations
import csv, json, time, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
EMB_DIR = BASE / "analysis/threshold_eval_datasets5/embeddings"
EXP4_RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent

STYLES = ["art_nouveau", "baroque", "expressionism", "impressionism",
          "post_impressionism", "realism", "renaissance", "romanticism",
          "surrealism", "ukiyo_e"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Threshold ranges
VIT_THRESHOLDS = np.arange(0.50, 0.555, 0.01)   # 0.50 … 0.55
CLIP_THRESHOLDS = np.arange(0.80, 0.855, 0.01)   # 0.80 … 0.85


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
    """Return metrics for one threshold pair under AND logic."""
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
    print(f"Positive pairs: {n_pos:,}  Negative pairs: {n_neg:,}")
    print(f"ViT thresholds:  {[f'{v:.2f}' for v in VIT_THRESHOLDS]}")
    print(f"CLIP thresholds: {[f'{v:.2f}' for v in CLIP_THRESHOLDS]}")
    print(f"Total combinations: {len(VIT_THRESHOLDS) * len(CLIP_THRESHOLDS)}")

    # ── Sweep all combinations ──
    print("\n" + "=" * 70)
    print("Evaluating threshold combinations (AND logic)...")
    print("=" * 70)

    rows = []
    # Also save each combination's detailed results
    all_results = {}

    for vit_thr in VIT_THRESHOLDS:
        for clip_thr in CLIP_THRESHOLDS:
            label = f"ViT>{vit_thr:.2f}_AND_CLIP>{clip_thr:.2f}"
            m = evaluate_combination(
                clip_thr, vit_thr,
                pos_clip, pos_vit, neg_clip, vit_neg,
            )
            rows.append({
                "vit_threshold": round(vit_thr, 2),
                "clip_threshold": round(clip_thr, 2),
                "rule": f"CLIP>{clip_thr:.2f} AND ViT>{vit_thr:.2f}",
                **m,
            })
            all_results[label] = {
                "vit_threshold": round(vit_thr, 2),
                "clip_threshold": round(clip_thr, 2),
                "rule": f"CLIP>{clip_thr:.2f} AND ViT>{vit_thr:.2f}",
                "metrics": m,
            }
            print(f"  {label:35s}  F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"TP={m['TP']:,}  FP={m['FP']:,}  Prec={m['precision']:.4f}  Spec={m['specificity']:.4f}")

    # ── Save detailed results per combination ──
    print("\nSaving per-combination results...")
    for vit_thr in VIT_THRESHOLDS:
        for clip_thr in CLIP_THRESHOLDS:
            label = f"ViT>{vit_thr:.2f}_AND_CLIP>{clip_thr:.2f}"
            combo_dir = OUT / label
            combo_dir.mkdir(parents=True, exist_ok=True)

            pos_pred = (pos_clip > clip_thr) & (pos_vit > vit_thr)
            neg_pred = (neg_clip > clip_thr) & (vit_neg > vit_thr)

            # Positive results
            with (combo_dir / "positive_results.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["original_path", "transform_path", "label", "clip_similarity",
                            "vit_similarity", "predicted_similar", "outcome"])
                for idx in range(len(pos_clip)):
                    w.writerow([
                        originals[int(pos_orig_idx[idx])], transform_paths[idx], 1,
                        f"{pos_clip[idx]:.6f}", f"{pos_vit[idx]:.6f}",
                        int(pos_pred[idx]), "TP" if pos_pred[idx] else "FN",
                    ])

            # Negative results
            with (combo_dir / "negative_results.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["image_a_path", "image_b_path", "label", "clip_similarity",
                            "vit_similarity", "predicted_similar", "outcome"])
                for idx, (a, b) in enumerate(zip(tri_i, tri_j)):
                    w.writerow([
                        originals[int(a)], originals[int(b)], 0,
                        f"{neg_clip[idx]:.6f}", f"{vit_neg[idx]:.6f}",
                        int(neg_pred[idx]), "FP" if neg_pred[idx] else "TN",
                    ])

            with (combo_dir / "summary.json").open("w", encoding="utf-8") as f:
                json.dump(all_results[label], f, indent=2)

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

    # ── Save summary JSON ──
    json_path = OUT / "threshold_sweep_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Print F1 grid for quick comparison ──
    print("\n" + "=" * 70)
    print("F1 Score Matrix (rows=ViT threshold, cols=CLIP threshold)")
    print("=" * 70)
    print(f"{'ViT ↓ / CLIP →':<14}", end="")
    for ct in CLIP_THRESHOLDS:
        print(f"{ct:.2f}     ", end=" ")
    print()
    for vt in VIT_THRESHOLDS:
        print(f"{vt:.2f}           ", end="")
        for ct in CLIP_THRESHOLDS:
            r = next(row for row in rows
                     if row["vit_threshold"] == round(vt, 2)
                     and row["clip_threshold"] == round(ct, 2))
            print(f"{r['f1']:.4f}  ", end=" ")
        print()

    # ── Print Accuracy grid ──
    print("\n" + "=" * 70)
    print("Accuracy Matrix (rows=ViT threshold, cols=CLIP threshold)")
    print("=" * 70)
    print(f"{'ViT ↓ / CLIP →':<14}", end="")
    for ct in CLIP_THRESHOLDS:
        print(f"{ct:.2f}     ", end=" ")
    print()
    for vt in VIT_THRESHOLDS:
        print(f"{vt:.2f}           ", end="")
        for ct in CLIP_THRESHOLDS:
            r = next(row for row in rows
                     if row["vit_threshold"] == round(vt, 2)
                     and row["clip_threshold"] == round(ct, 2))
            print(f"{r['accuracy']:.4f}  ", end=" ")
        print()

    # ── Print FP (false positives) grid ──
    print("\n" + "=" * 70)
    print("False Positives Matrix (rows=ViT threshold, cols=CLIP threshold)")
    print("=" * 70)
    print(f"{'ViT ↓ / CLIP →':<14}", end="")
    for ct in CLIP_THRESHOLDS:
        print(f"{ct:.2f}      ", end=" ")
    print()
    for vt in VIT_THRESHOLDS:
        print(f"{vt:.2f}           ", end="")
        for ct in CLIP_THRESHOLDS:
            r = next(row for row in rows
                     if row["vit_threshold"] == round(vt, 2)
                     and row["clip_threshold"] == round(ct, 2))
            print(f"{r['FP']:>6,}  ", end=" ")
        print()

    # ── Best by F1 ──
    best_f1 = max(rows, key=lambda r: r["f1"])
    print(f"\n🏆 Best by F1:  {best_f1['rule']}  →  F1={best_f1['f1']:.4f}  Acc={best_f1['accuracy']:.4f}")
    best_acc = max(rows, key=lambda r: r["accuracy"])
    print(f"🏆 Best by Acc: {best_acc['rule']}  →  Acc={best_acc['accuracy']:.4f}  F1={best_acc['f1']:.4f}")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Summary CSV → {csv_path}")
    print(f"Summary JSON → {json_path}")


if __name__ == "__main__":
    main()
