#!/usr/bin/env python3
"""Compare 4 decision logic rules on all Datasets5 pairs.

Rules:
  1. AND:       CLIP > 0.8 AND ViT > 0.5
  2. OR:        CLIP > 0.8 OR  ViT > 0.5
  3. CLIP-only: CLIP > 0.8
  4. ViT-only:  ViT > 0.5

Reuses precomputed similarities from exp4 without reloading models.
"""
from __future__ import annotations
import csv, json, time, numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
EMB_DIR = BASE / "analysis/threshold_eval_datasets5/embeddings"
EXP4_RES = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent

CLIP_THR, VIT_THR = 0.8, 0.5
STYLES = ["art_nouveau","baroque","expressionism","impressionism",
          "post_impressionism","realism","renaissance","romanticism",
          "surrealism","ukiyo_e"]
IMG_EXTS = {".jpg",".jpeg",".png",".bmp"}

def gather_originals():
    paths = []
    for s in STYLES:
        d = BASE / "Datasets5" / s
        paths.extend(p for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS)
    return paths

def metrics(tp, fp, tn, fn):
    total = tp+fp+tn+fn
    p = tp/(tp+fp) if tp+fp else 0.0
    r = tp/(tp+fn) if tp+fn else 0.0
    return {"total_pairs":total,"positive_pairs":tp+fn,"negative_pairs":tn+fp,
            "TP":int(tp),"FP":int(fp),"TN":int(tn),"FN":int(fn),
            "accuracy":round((tp+tn)/total,6) if total else 0,
            "precision":round(p,6),"recall":round(r,6),
            "f1":round(2*p*r/(p+r),6) if p+r else 0,
            "specificity":round(tn/(tn+fp),6) if tn+fp else 0}

def save_rule(rule_name, pos_pred, neg_pred, pos_clip, pos_vit, neg_clip, neg_vit,
              originals, transform_paths, pos_orig_idx, tri_i, tri_j):
    d = OUT / rule_name; d.mkdir(parents=True, exist_ok=True)

    # Positive results
    tp, fn = int(pos_pred.sum()), int((~pos_pred).sum())
    with (d/"positive_results.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["original_path","transform_path","label","clip_similarity",
                     "vit_similarity","predicted_similar","outcome"])
        for idx in range(len(pos_clip)):
            w.writerow([originals[int(pos_orig_idx[idx])],transform_paths[idx],1,
                        f"{pos_clip[idx]:.6f}",f"{pos_vit[idx]:.6f}",
                        int(pos_pred[idx]),"TP" if pos_pred[idx] else "FN"])

    # Negative results
    fp, tn = int(neg_pred.sum()), int((~neg_pred).sum())
    with (d/"negative_results.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_a_path","image_b_path","label","clip_similarity",
                     "vit_similarity","predicted_similar","outcome"])
        for idx, (a,b) in enumerate(zip(tri_i, tri_j)):
            w.writerow([originals[int(a)],originals[int(b)],0,
                        f"{neg_clip[idx]:.6f}",f"{neg_vit[idx]:.6f}",
                        int(neg_pred[idx]),"FP" if neg_pred[idx] else "TN"])

    m = metrics(tp, fp, tn, fn)
    with (d/"summary.json").open("w",encoding="utf-8") as f:
        json.dump({"rule":rule_name,"clip_threshold":CLIP_THR,"vit_threshold":VIT_THR,
                   "description":RULE_DESC[rule_name],"metrics":m},f,indent=2)
    return m

RULE_DESC = {
    "rule1_and": "CLIP > 0.8 AND ViT > 0.5 → similar",
    "rule2_or": "CLIP > 0.8 OR ViT > 0.5 → similar",
    "rule3_clip_only": "CLIP > 0.8 → similar",
    "rule4_vit_only": "ViT > 0.5 → similar",
}

def main():
    t0 = time.time()
    print("Loading data...")
    clip_d = np.load(EMB_DIR/"clip_embeddings.npz", allow_pickle=False)
    vit_d = np.load(EMB_DIR/"vit_embeddings.npz", allow_pickle=False)
    pos_clip = clip_d["pos_sims"].astype(np.float32)
    pos_vit = np.load(EXP4_RES/"positive_vit_similarities.npy").astype(np.float32)
    transform_paths = clip_d["transform_paths"]
    pos_orig_idx = clip_d["pos_orig_idx"].astype(np.int32)
    orig_clip = clip_d["orig_clip"].astype(np.float32)
    orig_vit = vit_d["orig_vit"].astype(np.float32)
    vit_neg = np.load(EMB_DIR/"vit_neg_sims.npy").astype(np.float32)
    originals = gather_originals()

    # Normalize + compute negative CLIP similarities
    print("Computing negative CLIP similarities...")
    orig_clip = orig_clip / np.linalg.norm(orig_clip, axis=1, keepdims=True)
    tri_i, tri_j = np.triu_indices(len(originals), k=1)
    neg_clip = np.sum(orig_clip[tri_i] * orig_clip[tri_j], axis=1).astype(np.float32)
    print(f"  Negative CLIP: mean={neg_clip.mean():.4f}, >0.8={(neg_clip>0.8).sum()}")

    # ── Apply 4 rules ──
    print("\n" + "="*60)
    print("Evaluating 4 rules...")
    print("="*60)

    # Rule 1: AND
    pos1 = (pos_clip > CLIP_THR) & (pos_vit > VIT_THR)
    neg1 = (neg_clip > CLIP_THR) & (vit_neg > VIT_THR)
    m1 = save_rule("rule1_and", pos1, neg1, pos_clip, pos_vit, neg_clip, vit_neg,
                   originals, transform_paths, pos_orig_idx, tri_i, tri_j)
    print(f"  Rule1 AND:    F1={m1['f1']:.4f}  Acc={m1['accuracy']:.4f}  TP={m1['TP']}  FP={m1['FP']}  TN={m1['TN']}  FN={m1['FN']}")

    # Rule 2: OR
    pos2 = (pos_clip > CLIP_THR) | (pos_vit > VIT_THR)
    neg2 = (neg_clip > CLIP_THR) | (vit_neg > VIT_THR)
    m2 = save_rule("rule2_or", pos2, neg2, pos_clip, pos_vit, neg_clip, vit_neg,
                   originals, transform_paths, pos_orig_idx, tri_i, tri_j)
    print(f"  Rule2 OR:     F1={m2['f1']:.4f}  Acc={m2['accuracy']:.4f}  TP={m2['TP']}  FP={m2['FP']}  TN={m2['TN']}  FN={m2['FN']}")

    # Rule 3: CLIP-only
    pos3 = (pos_clip > CLIP_THR)
    neg3 = (neg_clip > CLIP_THR)
    m3 = save_rule("rule3_clip_only", pos3, neg3, pos_clip, pos_vit, neg_clip, vit_neg,
                   originals, transform_paths, pos_orig_idx, tri_i, tri_j)
    print(f"  Rule3 CLIP:   F1={m3['f1']:.4f}  Acc={m3['accuracy']:.4f}  TP={m3['TP']}  FP={m3['FP']}  TN={m3['TN']}  FN={m3['FN']}")

    # Rule 4: ViT-only
    pos4 = (pos_vit > VIT_THR)
    neg4 = (vit_neg > VIT_THR)
    m4 = save_rule("rule4_vit_only", pos4, neg4, pos_clip, pos_vit, neg_clip, vit_neg,
                   originals, transform_paths, pos_orig_idx, tri_i, tri_j)
    print(f"  Rule4 ViT:    F1={m4['f1']:.4f}  Acc={m4['accuracy']:.4f}  TP={m4['TP']}  FP={m4['FP']}  TN={m4['TN']}  FN={m4['FN']}")

    # ── Summary comparison ──
    print("\n" + "="*60)
    print("Cross-rule comparison")
    print("="*60)
    rules = [("Rule1 AND", m1), ("Rule2 OR", m2), ("Rule3 CLIP-only", m3), ("Rule4 ViT-only", m4)]
    header = ["Rule","TP","FP","TN","FN","Accuracy","Precision","Recall","F1","Specificity","FPR"]
    rows = []
    for name, m in rules:
        rows.append([name, m["TP"], m["FP"], m["TN"], m["FN"],
                     m["accuracy"], m["precision"], m["recall"], m["f1"],
                     m["specificity"], round(1-m["specificity"],6)])
        print(f"  {name:<18s} F1={m['f1']:.4f}  Prec={m['precision']:.4f}  "
              f"Rec={m['recall']:.4f}  Acc={m['accuracy']:.4f}  FP={m['FP']:,}")

    # Save comparison CSV
    with (OUT/"summary_comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows: w.writerow(r)

    # Save comparison JSON
    key_map = {"Rule1 AND":"rule1_and","Rule2 OR":"rule2_or",
               "Rule3 CLIP-only":"rule3_clip_only","Rule4 ViT-only":"rule4_vit_only"}
    comp = {}
    for name, m in rules:
        comp[key_map[name]] = {"description": RULE_DESC[key_map[name]],
                               "thresholds": {"clip": CLIP_THR, "vit": VIT_THR}, "metrics": m}
    with (OUT/"summary_comparison.json").open("w",encoding="utf-8") as f:
        json.dump(comp, f, indent=2, ensure_ascii=False)

    print(f"\nDone in {time.time()-t0:.1f}s → {OUT}")

if __name__ == "__main__":
    main()
