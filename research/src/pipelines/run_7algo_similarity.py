#!/usr/bin/env python3
"""
Run all 7 similarity algorithms (CLIP, ViT, 5 hashes) on each original→transform pair.
Creates per-algorithm similarity tables + combined summary chart.

Datasets handled:
  - Datasets2_2 only (100 images: 10/class)
  - Datasets2_2 + Archive (200 images combined)
"""

from __future__ import annotations
import json, time, sys, numpy as np, pandas as pd
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import imagehash
from pdqhash import compute as pdq_compute

BASE = Path(__file__).resolve().parent.parent.parent
OUT_BASE = BASE / "analysis" / "similarity_7algo"
CLIP_DIR = BASE / "models" / "clip-vit-base-patch32"
VIT_DIR = BASE / "models" / "vit-base-patch16-224"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 16
CLASSES = ["airplane","bird","car","cat","chair","dog","fish","flower","house","tree"]
N_PER_CLASS = 10

# ── Transforms ──
def op_F1(img): return cv2.flip(img, 1)
def op_F2(img): return cv2.flip(img, 0)
def op_C(img, s=1.0):
    h,w=img.shape[:2]; r=min(0.05*s,0.95); cr=1.0-r
    nh,nw=int(h*cr),int(w*cr); y0,x0=(h-nh)//2,(w-nw)//2
    return cv2.resize(img[y0:y0+nh,x0:x0+nw],(w,h),interpolation=cv2.INTER_LANCZOS4)
def op_S1(img,s=1.0): h,w=img.shape[:2]; return cv2.warpAffine(img,np.float32([[1,0,int(0.05*w*s)],[0,1,0]]),(w,h),borderMode=cv2.BORDER_REFLECT_101)
def op_S2(img,s=1.0): h,w=img.shape[:2]; return cv2.warpAffine(img,np.float32([[1,0,0],[0,1,int(0.05*h*s)]]),(w,h),borderMode=cv2.BORDER_REFLECT_101)
def op_R(img,s=1.0): h,w=img.shape[:2]; M=cv2.getRotationMatrix2D((w/2,h/2),2.0*s,1.0); return cv2.warpAffine(img,M,(w,h),borderMode=cv2.BORDER_REFLECT_101)
def op_B(img,s=1.0): return cv2.convertScaleAbs(img,alpha=1.0+0.04*s,beta=8.0*s)
def op_V(img,s=1.0):
    h,w=img.shape[:2]; y,x=np.ogrid[:h,:w]; cx,cy=w/2,h/2; dx,dy=(x-cx)/(w/2),(y-cy)/(h/2)
    d=np.sqrt(dx**2+dy**2); mask=np.clip(1.0-d*(0.25*s),0,1); mask=np.power(mask,1.5*s)
    return (img.astype(np.float32)*mask[:,:,np.newaxis]).astype(np.uint8)
def op_T(img,s=1.0):
    r=img.astype(np.float32); r[:,:,2]=np.clip(r[:,:,2]*(1.0+0.06*s),0,255)
    r[:,:,1]=np.clip(r[:,:,1]*(1.0+0.02*s),0,255); r[:,:,0]=np.clip(r[:,:,0]*(1.0-0.05*s),0,255)
    return r.astype(np.uint8)
def op_Sh(img,s=1.0):
    sk=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
    ik=np.array([[0,0,0],[0,1,0],[0,0,0]],dtype=np.float32)
    w=min(0.4*s,0.95); return cv2.filter2D(img,-1,w*sk+(1-w)*ik)
def op_K(img,s=1.0):
    h,w=img.shape[:2]; m=0.03*s; src=np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst=np.float32([[0,0],[w,h*m],[w,h*(1-m)],[0,h]])
    return cv2.warpPerspective(img,cv2.getPerspectiveTransform(src,dst),(w,h),borderMode=cv2.BORDER_REFLECT_101)
def op_N(img,s=1.0):
    rng=np.random.default_rng(42); return np.clip(img.astype(np.float32)+rng.normal(0,3.0*s,img.shape),0,255).astype(np.uint8)

OPS=[("F1",op_F1,None),("F2",op_F2,None),
     ("C",op_C,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
     ("S1",op_S1,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
     ("S2",op_S2,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
     ("R",op_R,[(1.5,"low"),(3.0,"mid"),(6.0,"high")]),
     ("B",op_B,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
     ("V",op_V,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
     ("T",op_T,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
     ("Sh",op_Sh,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
     ("K",op_K,[(0.33,"low"),(1.0,"mid"),(1.67,"high")]),
     ("N",op_N,[(0.5,"low"),(1.0,"mid"),(1.5,"high")])]

VARIANT_NAMES = []
for code, func, settings in OPS:
    if settings is None: VARIANT_NAMES.append(f"{code}")
    else:
        for s,lbl in settings: VARIANT_NAMES.append(f"{code}_{lbl}")

NON_F2_IDX = [i for i, vn in enumerate(VARIANT_NAMES) if not vn.startswith("F2")]
NON_F2_NAMES = [VARIANT_NAMES[i] for i in NON_F2_IDX]

# ── Gather originals ──
def gather_dataset(name, src_dir, limit=N_PER_CLASS):
    imgs = []
    for cls in CLASSES:
        d = src_dir / cls
        if not d.is_dir(): continue
        for fp in sorted(d.glob("*.jpg"))[:limit]:
            imgs.append({"path": fp, "class": cls, "stem": f"{name}_{fp.stem}", "source": name})
    return imgs

# ── Generate transforms ──
def gen_transforms(originals, tdir):
    tdir.mkdir(parents=True, exist_ok=True)
    for info in tqdm(originals, desc="  Transforms"):
        od = tdir / info["stem"]; od.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(info["path"]))
        if img is None: continue
        for code, func, settings in OPS:
            if settings is None:
                cv2.imwrite(str(od/f"{code}.jpg"), func(img))
            else:
                for s,lbl in settings: cv2.imwrite(str(od/f"{code}_{lbl}.jpg"), func(img,s=s))

# ── Load models once ──
def load_models():
    cp = str(CLIP_DIR) if CLIP_DIR.exists() else "openai/clip-vit-base-patch32"
    vp = str(VIT_DIR) if VIT_DIR.exists() else "google/vit-base-patch16-224"
    print(f"Loading CLIP+ViT...")
    return ((CLIPProcessor.from_pretrained(cp), CLIPModel.from_pretrained(cp).to(DEVICE).eval()),
            (ViTImageProcessor.from_pretrained(vp), ViTModel.from_pretrained(vp).to(DEVICE).eval()))

# ── Embedding helpers ──
@torch.no_grad()
def extract_all_embeddings(paths, proc, model, is_clip):
    embs = []
    for i in range(0, len(paths), BATCH):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i+BATCH]]
        if is_clip:
            inp = proc(images=imgs, return_tensors="pt", padding=True)
        else:
            inp = proc(images=imgs, return_tensors="pt")
        inp = {k:v.to(DEVICE) for k,v in inp.items()}
        if is_clip:
            feats = model.get_image_features(**inp)
            feats = feats.pooler_output if hasattr(feats,"pooler_output") else feats
        else:
            feats = model(**inp).last_hidden_state[:,0,:]
        if isinstance(feats, tuple): feats = feats[0]
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embs.append(feats.cpu().numpy())
    return np.concatenate(embs, axis=0)

# ── Hash helpers ──
def h_ahash(pil): return np.array(imagehash.average_hash(pil, hash_size=8).hash.flatten(), dtype=np.int8)
def h_phash(pil): return np.array(imagehash.phash(pil, hash_size=8).hash.flatten(), dtype=np.int8)
def h_dhash(pil): return np.array(imagehash.dhash(pil, hash_size=8).hash.flatten(), dtype=np.int8)
def h_whash(pil): return np.array(imagehash.whash(pil, hash_size=8).hash.flatten(), dtype=np.int8)
def h_pdq(pil): arr=np.array(pil); b,_=pdq_compute(arr); return b.astype(np.int8)

HASH_FNS = {"ahash":(h_ahash,64),"phash":(h_phash,64),"dhash":(h_dhash,64),"whash":(h_whash,64),"pdq":(h_pdq,256)}

# ── Run experiment ──
def run_experiment(label, originals, tdir, clip_proc, clip_model, vit_proc, vit_model):
    print(f"\n{'='*50}\n{label}\n{'='*50}")
    out = OUT_BASE / label; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Transforms
    print("\n[1/4] Generating transforms...")
    gen_transforms(originals, tdir)
    print(f"  {time.time()-t0:.0f}s")

    # 2. Build paths: originals + all variants
    print("\n[2/4] Collecting paths...")
    N = len(originals); M = len(VARIANT_NAMES)
    all_paths = []
    for info in originals:
        all_paths.append(info["path"])
        for vn in VARIANT_NAMES:
            all_paths.append(tdir / info["stem"] / f"{vn}.jpg")

    # 3. Compute all similarities
    print(f"\n[3/4] Computing {N}×{M} similarities for 7 algorithms...")
    timing = {}
    sims = {}  # algo_key -> (N, M) matrix

    # ── CLIP ──
    t1 = time.time()
    emb_clip = extract_all_embeddings(all_paths, clip_proc, clip_model, is_clip=True)
    clip_mat = np.zeros((N, M), dtype=np.float32)
    oi = 0
    for i in range(N):
        oe = emb_clip[oi]; ve = emb_clip[oi+1:oi+M+1]
        clip_mat[i] = np.clip(np.dot(ve, oe), 0, 1); oi += M + 1
    sims["clip"] = clip_mat
    timing["clip"] = round(time.time()-t1, 1)
    print(f"  CLIP: {timing['clip']}s")

    # ── ViT ──
    t1 = time.time()
    emb_vit = extract_all_embeddings(all_paths, vit_proc, vit_model, is_clip=False)
    vit_mat = np.zeros((N, M), dtype=np.float32)
    oi = 0
    for i in range(N):
        oe = emb_vit[oi]; ve = emb_vit[oi+1:oi+M+1]
        vit_mat[i] = np.clip(np.dot(ve, oe), 0, 1); oi += M + 1
    sims["vit"] = vit_mat
    timing["vit"] = round(time.time()-t1, 1)
    print(f"  ViT:  {timing['vit']}s")

    # ── 5 Hashes ──
    for hk, (hfn, bits) in HASH_FNS.items():
        t1 = time.time()
        mat = np.zeros((N, M), dtype=np.float32)
        # Pre-compute all hashes
        all_h = []
        for p in tqdm(all_paths, desc=f"    {hk}"):
            all_h.append(hfn(Image.open(p).convert("RGB")))
        for i, info in enumerate(originals):
            oh = all_h[i*(M+1)]
            for j in range(M):
                vh = all_h[i*(M+1)+1+j]
                mat[i,j] = 1.0 - np.sum(oh != vh) / bits
        sims[hk] = mat
        timing[hk] = round(time.time()-t1, 1)
        print(f"  {hk}: {timing[hk]}s")

    # 4. Save tables
    print(f"\n[4/4] Saving...")
    row_names = [info["stem"] for info in originals]
    for alg in ["clip","vit","ahash","phash","dhash","whash","pdq"]:
        df = pd.DataFrame(sims[alg], index=row_names, columns=VARIANT_NAMES)
        df.to_csv(out / f"{alg}_similarity.csv", float_format="%.6f")

    # Summary
    algos = [("clip","CLIP"),("vit","ViT"),("ahash","aHash"),("phash","pHash"),
             ("dhash","dHash"),("whash","wHash"),("pdq","PDQ")]
    sum_rows = []
    for ak, al in algos:
        v = sims[ak].flatten()
        sum_rows.append({"algorithm":al,"mean":np.mean(v),"std":np.std(v),
                         "min":np.min(v),"max":np.max(v),"median":np.median(v)})
    pdf = pd.DataFrame(sum_rows)
    pdf.to_csv(out/"summary_stats.csv",index=False,float_format="%.6f")
    print(pdf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    total = time.time()-t0
    with open(out/"timing.json","w") as f: json.dump(timing,f,indent=2)

    print(f"\n  Total: {int(total//60)}m{total%60:.0f}s → {out}")
    return sims, pdf

# ═══════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════
def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)

    datasets = []

    # Dataset 1: Datasets2_2 (100 images)
    ds2_2 = gather_dataset("ds2_2", BASE/"Datasets2_2", N_PER_CLASS)
    datasets.append(("datasets2_2_100", ds2_2))
    print(f"Datasets2_2: {len(ds2_2)} images")

    # Dataset 2: Datasets2_2 + Archive (100 + 100)
    archive = []
    # Get 10 images per Archive category
    ar_cats = ["ex1_ex2_inputs","ex2_ai_art","ex2_car","ex2_cat","ex2_china_art",
               "ex2_football","ex2_house","ex2_nba","ex2_painting","ex2_view"]
    for cat in ar_cats:
        d = BASE/"Archive"/cat
        if not d.is_dir(): continue
        for fp in sorted(d.glob("*"))[:N_PER_CLASS]:
            if fp.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}:
                archive.append({"path": fp, "class": cat, "stem": f"ar_{cat}_{fp.stem}", "source": "archive"})
    # Use first 10 categories × 10 = 100
    archive = archive[:100]

    ds_combined = ds2_2 + archive
    datasets.append(("datasets2_2_plus_archive_200", ds_combined))
    print(f"Archive: {len(archive)} images → Combined: {len(ds_combined)}")

    # Load models once
    (clip_proc, clip_model), (vit_proc, vit_model) = load_models()

    all_results = {}
    for name, originals in datasets:
        tdir = OUT_BASE / name / "transforms"
        sims, summary = run_experiment(name, originals, tdir, clip_proc, clip_model, vit_proc, vit_model)
        all_results[name] = {"sims": sims, "summary": summary, "originals": originals}

    # ── Generate combined charts ──
    print(f"\n{'='*50}\nGenerating summary charts...\n{'='*50}")
    generate_charts(all_results)

    print(f"\nDone! Results in {OUT_BASE}")


def generate_charts(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    algos = ["clip","vit","ahash","phash","dhash","whash","pdq"]
    algo_labels = ["CLIP","ViT","aHash","pHash","dHash","wHash","PDQ"]

    for name, data in all_results.items():
        summary = data["summary"]
        sims = data["sims"]

        # Use NON_F2 only for cleaner stats
        non_f2_means = {}
        for ak in algos:
            mat = sims[ak][:, NON_F2_IDX]  # only non-F2 columns
            non_f2_means[ak] = np.mean(mat, axis=0)  # mean across images per variant

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))
        fig.suptitle(f"7-Algorithm Similarity Analysis — {name}", fontsize=16, fontweight="bold")

        # 1. Bar: mean similarity per algorithm (across all pairs)
        ax = axes[0,0]
        means = [summary[summary["algorithm"]==al]["mean"].values[0] for al in algo_labels]
        colors = ["#5DA5E0","#2A6F97","#F5853F","#E45756","#F9A03F","#D85040","#A23E48"]
        bars = ax.bar(algo_labels, means, color=colors, edgecolor="white")
        for b,v in zip(bars, means): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.15); ax.set_ylabel("Mean Similarity")
        ax.set_title("Mean Similarity by Algorithm", fontweight="bold"); ax.grid(alpha=0.3, axis="y")

        # 2. Per-transform line chart
        ax = axes[0,1]
        x = range(len(NON_F2_NAMES))
        for ak, al, c in zip(algos, algo_labels, colors):
            ax.plot(x, non_f2_means[ak], linewidth=1.5, color=c, label=al, alpha=0.85)
        ax.set_xticks(x[::3]); ax.set_xticklabels([NON_F2_NAMES[i] for i in x[::3]], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Mean Similarity"); ax.set_title("By Transform Variant (non-F2)", fontweight="bold")
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

        # 3. Stats table
        ax = axes[0,2]; ax.axis("off")
        tbl = summary.copy()
        tbl_text = f"{'Algorithm':<12s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s}\n{'─'*60}\n"
        for _, row in tbl.iterrows():
            tbl_text += f"{row['algorithm']:<12s} {row['mean']:>8.4f} {row['std']:>8.4f} {row['median']:>8.4f} {row['min']:>8.4f} {row['max']:>8.4f}\n"
        ax.text(0.02, 0.98, tbl_text, transform=ax.transAxes, fontsize=9, va="top", fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.7))

        # 4. Per-level boxplot-style
        ax = axes[1,0]
        levels = {"low":[],"mid":[],"high":[]}
        for i, vn in enumerate(VARIANT_NAMES):
            parts = vn.split("_")
            if len(parts) >= 2 and parts[1] in levels:
                lv = parts[1]
                for ak in algos:
                    levels[lv].append(np.mean(sims[ak][:,i]))
        xp = np.arange(3); w = 0.12
        for idx, (ak, al) in enumerate(zip(algos, algo_labels)):
            vals = [np.mean(levels[lv][idx::7]) for lv in ["low","mid","high"]]
            ax.bar(xp + idx*w - 3*w, vals, w, color=colors[idx], label=al, alpha=0.85)
        ax.set_xticks(xp); ax.set_xticklabels(["Low","Mid","High"])
        ax.set_ylabel("Mean Similarity"); ax.set_title("By Transform Level", fontweight="bold")
        ax.legend(fontsize=7, ncol=4); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.1)

        # 5. Distribution violin-like (strip plot)
        ax = axes[1,1]
        positions = range(len(algos))
        for idx, ak in enumerate(algos):
            vals = sims[ak].flatten()
            # Show mean + range
            m, s = np.mean(vals), np.std(vals)
            ax.bar(idx, m, 0.5, color=colors[idx], alpha=0.7, yerr=s, capsize=3)
            ax.text(idx, m+s+0.02, f"{m:.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(positions); ax.set_xticklabels(algo_labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Similarity"); ax.set_title("Mean ± Std by Algorithm", fontweight="bold")
        ax.set_ylim(0, 1.15); ax.grid(alpha=0.3, axis="y")

        # 6. Correlation heatmap
        ax = axes[1,2]
        # Flatten all values per algorithm
        flat_vals = np.column_stack([sims[ak].flatten() for ak in algos])
        corr = np.corrcoef(flat_vals.T)
        im = ax.imshow(corr, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(algos)):
            for j in range(len(algos)):
                ax.text(j, i, f"{corr[i,j]:.3f}", ha="center", va="center", fontsize=8, fontweight="bold",
                        color="white" if corr[i,j] < 0.5 else "black")
        ax.set_xticks(range(len(algos))); ax.set_xticklabels(algo_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(algos))); ax.set_yticklabels(algo_labels, fontsize=8)
        ax.set_title("Cross-Algorithm Correlation", fontweight="bold")
        plt.colorbar(im, ax=ax, shrink=0.8)

        plt.tight_layout()
        out_path = OUT_BASE / name / "all_7_metrics_combined.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Chart saved: {out_path}")


if __name__ == "__main__":
    main()
