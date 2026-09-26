#!/usr/bin/env python3
"""Combined 200-image threshold sweep: Archive(100) + Datasets2_1(100)."""

from __future__ import annotations
import json, time, sys, numpy as np, pandas as pd
from pathlib import Path
import cv2
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "analysis" / "threshold_eval_clip08_05_200images"
TRANSFORM_DIR = OUT_DIR / "transforms"
CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"
ARCHIVE_DIR = BASE_DIR / "Archive"
DATASETS_DIR = BASE_DIR / "Datasets2_1"
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
BATCH_SIZE = 32

CLIP_THRESHOLDS = np.round(np.arange(0.80, 0.85 + 0.001, 0.01), 2)
VIT_THRESHOLDS = np.round(np.arange(0.30, 0.50 + 0.001, 0.01), 2)

# ── Transforms (same as before) ──
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

OPERATIONS=[
    ("F1",op_F1,None),("F2",op_F2,None),
    ("C",op_C,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("S1",op_S1,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("S2",op_S2,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("R",op_R,[(1.5,"low"),(3.0,"mid"),(6.0,"high")]),
    ("B",op_B,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("V",op_V,[(0.4,"low"),(1.0,"mid"),(1.6,"high")]),
    ("T",op_T,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("Sh",op_Sh,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
    ("K",op_K,[(0.33,"low"),(1.0,"mid"),(1.67,"high")]),
    ("N",op_N,[(0.5,"low"),(1.0,"mid"),(1.5,"high")]),
]

def gather_originals():
    """Gather 200 originals: 100 from Archive + 100 from Datasets2_1."""
    archives = []
    # Read from existing positive_pairs to get the exact 100 Archive images
    pos_ar = pd.read_csv(BASE_DIR / "analysis/threshold_eval_clip08_vit05_archieve/positive_pairs_20260707.csv")
    seen = set()
    for name in pos_ar["name_A"].unique():
        if len(archives) >= 100: break
        # name format: "ex1_ex2_inputs/Images_boat.png"
        parts = name.split("/", 1)
        if len(parts) == 2:
            fpath = ARCHIVE_DIR / parts[0] / parts[1]
            if fpath.exists():
                stem = f"ar_{parts[0]}_{Path(parts[1]).stem}"
                if stem not in seen:
                    archives.append({"path": fpath, "stem": stem, "source": "archive"})
                    seen.add(stem)

    datasets2 = []
    # Read from positive_pairs to get the exact 100 Datasets2_1 images
    pos_ds = pd.read_csv(BASE_DIR / "analysis/threshold_eval_clip08_vit05_datasets2_2/positive_pairs.csv")
    for name in pos_ds["name_A"].unique():
        if len(datasets2) >= 100: break
        stem = name  # e.g. "airplane01"
        # Find in Datasets2_1
        for cls_dir in DATASETS_DIR.iterdir():
            if not cls_dir.is_dir(): continue
            fpath = cls_dir / f"{stem}.jpg"
            if fpath.exists():
                datasets2.append({"path": fpath, "stem": f"ds_{stem}", "source": "datasets2_1"})
                break

    print(f"Archive: {len(archives)}, Datasets2_1: {len(datasets2)}")
    return archives + datasets2

def generate_transforms(originals):
    TRANSFORM_DIR.mkdir(parents=True, exist_ok=True)
    for info in tqdm(originals, desc="  Transforms"):
        od = TRANSFORM_DIR / info["stem"]
        od.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(info["path"]))
        if img is None: continue
        for code, func, settings in OPERATIONS:
            if settings is None:
                cv2.imwrite(str(od/f"{code}.jpg"), func(img))
            else:
                for strength, label in settings:
                    cv2.imwrite(str(od/f"{code}_{label}.jpg"), func(img, s=strength))

def build_pairs(originals):
    pos, neg = [], []
    for info in originals:
        vd = TRANSFORM_DIR / info["stem"]
        for vp in sorted(vd.glob("*.jpg")):
            if vp.stem.startswith("F2"): continue
            pos.append({"path_A": info["path"], "path_B": vp, "name_A": info["stem"], "name_B": f"{info['stem']}/{vp.name}", "label": 1})
    for i in range(len(originals)):
        for j in range(i+1, len(originals)):
            neg.append({"path_A": originals[i]["path"], "path_B": originals[j]["path"], "name_A": originals[i]["stem"], "name_B": originals[j]["stem"], "label": 0})
    return pos, neg

def load_models():
    cp = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
    vp = str(VIT_MODEL_DIR) if VIT_MODEL_DIR.exists() else "google/vit-base-patch16-224"
    print(f"CLIP: {cp}\nViT:  {vp}")
    return ((CLIPProcessor.from_pretrained(cp), CLIPModel.from_pretrained(cp).to(DEVICE).eval()),
            (ViTImageProcessor.from_pretrained(vp), ViTModel.from_pretrained(vp).to(DEVICE).eval()))

@torch.no_grad()
def compute_clip(pa, pb, proc, model):
    sims = []
    for i in range(0, len(pa), BATCH_SIZE):
        ba = [Image.open(p).convert("RGB") for p in pa[i:i+BATCH_SIZE]]
        bb = [Image.open(p).convert("RGB") for p in pb[i:i+BATCH_SIZE]]
        ia = {k:v.to(DEVICE) for k,v in proc(images=ba, return_tensors="pt", padding=True).items()}
        ib = {k:v.to(DEVICE) for k,v in proc(images=bb, return_tensors="pt", padding=True).items()}
        fa = model.get_image_features(**ia); fb = model.get_image_features(**ib)
        fa = fa.pooler_output if hasattr(fa,"pooler_output") else fa
        fb = fb.pooler_output if hasattr(fb,"pooler_output") else fb
        if isinstance(fa,tuple): fa=fa[0]
        if isinstance(fb,tuple): fb=fb[0]
        fa, fb = fa/fa.norm(dim=-1,keepdim=True), fb/fb.norm(dim=-1,keepdim=True)
        sims.append((fa*fb).sum(dim=-1).cpu().numpy())
    return np.concatenate(sims)

@torch.no_grad()
def compute_vit(pa, pb, proc, model):
    sims = []
    for i in range(0, len(pa), BATCH_SIZE):
        ba = [Image.open(p).convert("RGB") for p in pa[i:i+BATCH_SIZE]]
        bb = [Image.open(p).convert("RGB") for p in pb[i:i+BATCH_SIZE]]
        ia = {k:v.to(DEVICE) for k,v in proc(images=ba, return_tensors="pt").items()}
        ib = {k:v.to(DEVICE) for k,v in proc(images=bb, return_tensors="pt").items()}
        ca = model(**ia).last_hidden_state[:,0,:]; cb = model(**ib).last_hidden_state[:,0,:]
        ca, cb = ca/ca.norm(dim=-1,keepdim=True), cb/cb.norm(dim=-1,keepdim=True)
        sims.append((ca*cb).sum(dim=-1).cpu().numpy())
    return np.concatenate(sims)

def evaluate(y_true, y_pred):
    tp=int(np.sum((y_true==1)&(y_pred==1))); tn=int(np.sum((y_true==0)&(y_pred==0)))
    fp=int(np.sum((y_true==0)&(y_pred==1))); fn=int(np.sum((y_true==1)&(y_pred==0)))
    total=tp+tn+fp+fn
    return {"TP":tp,"TN":tn,"FP":fp,"FN":fn,
            "accuracy":round((tp+tn)/total,4) if total else 0,
            "recall":round(tp/(tp+fn),4) if(tp+fn)else 0,
            "precision":round(tp/(tp+fp),4) if(tp+fp)else 0,
            "f1":round(2*tp/(2*tp+fp+fn),4) if(2*tp+fp+fn)else 0}

# ═══════════════════════════════════════════════════════
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print("="*60)
    print("Combined 200-image Threshold Sweep")
    print("="*60)

    # 1. Gather
    print("\n[1/5] Gathering 200 originals...")
    originals = gather_originals()
    print(f"  Total: {len(originals)}")

    # 2. Transforms
    print(f"\n[2/5] Generating transforms ({len(originals)}×32)...")
    t0 = time.time()
    generate_transforms(originals)
    print(f"  Done: {time.time()-t0:.1f}s")

    # 3. Pairs
    print("\n[3/5] Building pairs...")
    pos_pairs, neg_pairs = build_pairs(originals)
    print(f"  Positive: {len(pos_pairs)}, Negative: {len(neg_pairs)}")

    # 4. Similarity
    print("\n[4/5] Computing similarities...")
    (clip_proc, clip_model), (vit_proc, vit_model) = load_models()

    t1 = time.time()
    clip_pos_sims = compute_clip([p["path_A"] for p in pos_pairs], [p["path_B"] for p in pos_pairs], clip_proc, clip_model)
    print(f"  CLIP: {time.time()-t1:.1f}s")

    t2 = time.time()
    vit_neg_sims = compute_vit([p["path_A"] for p in neg_pairs], [p["path_B"] for p in neg_pairs], vit_proc, vit_model)
    print(f"  ViT:  {time.time()-t2:.1f}s")

    # 5. Evaluate
    print("\n[5/5] Evaluating thresholds...")
    clip_true = np.ones(len(clip_pos_sims), dtype=int)
    vit_true = np.zeros(len(vit_neg_sims), dtype=int)

    # Summary at base threshold
    cp = (clip_pos_sims >= 0.8).astype(int)
    vp = (vit_neg_sims >= 0.5).astype(int)
    ov = evaluate(np.concatenate([clip_true, vit_true]), np.concatenate([cp, vp]))
    print(f"  Base (CLIP=0.8,ViT=0.5): Acc={ov['accuracy']:.4f}, F1={ov['f1']:.4f}")

    # Grid sweep
    grid_rows = []
    for ct in CLIP_THRESHOLDS:
        cpred = (clip_pos_sims >= ct).astype(int)
        cm = evaluate(clip_true, cpred)
        for vt in VIT_THRESHOLDS:
            vpred = (vit_neg_sims >= vt).astype(int)
            vm = evaluate(vit_true, vpred)
            ovm = evaluate(np.concatenate([clip_true, vit_true]), np.concatenate([cpred, vpred]))
            grid_rows.append({"clip_threshold":round(float(ct),2),"vit_threshold":round(float(vt),2),
                "clip_TP":cm["TP"],"clip_FN":cm["FN"],"vit_TN":vm["TN"],"vit_FP":vm["FP"],
                "overall_TP":ovm["TP"],"overall_TN":ovm["TN"],"overall_FP":ovm["FP"],"overall_FN":ovm["FN"],
                "accuracy":ovm["accuracy"],"recall":ovm["recall"],"precision":ovm["precision"],"f1":ovm["f1"]})
    grid_df = pd.DataFrame(grid_rows)

    # Save CSVs
    pd.DataFrame({"name_A":[p["name_A"] for p in pos_pairs],"name_B":[p["name_B"] for p in pos_pairs],
                  "label":1,"clip_similarity":clip_pos_sims}).to_csv(OUT_DIR/"positive_pairs.csv",index=False,float_format="%.6f")
    pd.DataFrame({"name_A":[p["name_A"] for p in neg_pairs],"name_B":[p["name_B"] for p in neg_pairs],
                  "label":0,"vit_similarity":vit_neg_sims}).to_csv(OUT_DIR/"negative_pairs.csv",index=False,float_format="%.6f")
    grid_df.to_csv(OUT_DIR/"threshold_grid.csv",index=False,float_format="%.4f")

    best = grid_df.loc[grid_df["f1"].idxmax()]
    print(f"\n  Best: CLIP={best['clip_threshold']}, ViT={best['vit_threshold']} -> F1={best['f1']:.4f}, Acc={best['accuracy']:.4f}")

    # Timing
    total = time.time() - t_total
    with open(OUT_DIR/"timing.json","w") as f:
        json.dump({"num_originals":len(originals),"positive_pairs":len(pos_pairs),"negative_pairs":len(neg_pairs),
                   "total_sec":round(total,1),"total_fmt":f"{int(total//60)}m{total%60:.0f}s"},f,indent=2)

    # ── Excel (matching existing format) ──
    print("\n  Generating Excel...")
    with pd.ExcelWriter(OUT_DIR/"overall_summary_excel_chart_200images.xlsx", engine="xlsxwriter") as writer:
        wb = writer.book
        c_ws = wb.add_worksheet("03_threshold_curves"); f1_ws = wb.add_worksheet("04_f1_heatmap"); fp_ws = wb.add_worksheet("05_fp_heatmap")
        writer.sheets["03_threshold_curves"]=c_ws; writer.sheets["04_f1_heatmap"]=f1_ws; writer.sheets["05_fp_heatmap"]=fp_ws

        # Build sweep + overview data
        CLIP_THRESHOLD = 0.8
        cb = evaluate(clip_true, cp)
        sweep_rows = []
        for vt in VIT_THRESHOLDS:
            vpr = (vit_neg_sims >= vt).astype(int)
            vm = evaluate(vit_true, vpr)
            ovs = evaluate(np.concatenate([clip_true, vit_true]), np.concatenate([cp, vpr]))
            sweep_rows.append({"clip_threshold":CLIP_THRESHOLD,"vit_threshold":round(float(vt),2),
                "clip_TP":cb["TP"],"clip_FN":cb["FN"],"vit_TN":vm["TN"],"vit_FP":vm["FP"],
                "overall_TP":ovs["TP"],"overall_TN":ovs["TN"],"overall_FP":ovs["FP"],"overall_FN":ovs["FN"],
                "accuracy":ovs["accuracy"],"recall":ovs["recall"],"precision":ovs["precision"],"f1":ovs["f1"]})
        sweep_df = pd.DataFrame(sweep_rows)

        overall_summary = {"model":"Overall (CLIP + ViT)","threshold":"0.8/0.5",**ov}
        overall_metrics_df = pd.DataFrame([{"metric":k,"value":v} for k,v in ov.items() if k in ["TP","TN","FP","FN","accuracy","recall","precision","f1"]])

        best_f1 = grid_df.sort_values(by=["f1","precision","accuracy","overall_FP"],ascending=[False,False,False,True]).head(5).copy()
        best_f1.insert(0,"ranking",[f"top_f1_{i}" for i in range(1,6)])
        low_fp = grid_df.sort_values(by=["overall_FP","f1","accuracy"],ascending=[True,False,False]).head(5).copy()
        low_fp.insert(0,"ranking",[f"low_fp_{i}" for i in range(1,6)])
        best_df = pd.concat([best_f1,low_fp],ignore_index=True)

        f1_hm = grid_df.pivot(index="clip_threshold",columns="vit_threshold",values="f1")
        f1_hm = f1_hm.reindex(index=CLIP_THRESHOLDS,columns=VIT_THRESHOLDS).reset_index()
        fp_hm = grid_df.pivot(index="clip_threshold",columns="vit_threshold",values="overall_FP")
        fp_hm = fp_hm.reindex(index=CLIP_THRESHOLDS,columns=VIT_THRESHOLDS).reset_index()

        pd.DataFrame([{"model":overall_summary["model"],"threshold":overall_summary["threshold"]}]).to_excel(writer,sheet_name="00_overview",index=False,startrow=1)
        overall_metrics_df.to_excel(writer,sheet_name="00_overview",index=False,startrow=5)
        best_df.to_excel(writer,sheet_name="00_overview",index=False,startrow=17)
        sweep_df.to_excel(writer,sheet_name="01_vit_sweep",index=False)
        grid_df.to_excel(writer,sheet_name="02_clip_vit_grid",index=False)

        hdr=wb.add_format({"bold":True,"bg_color":"#D9EAF7","border":1}); title=wb.add_format({"bold":True,"font_size":14})
        note=wb.add_format({"italic":True,"font_color":"#666666"}); txt=wb.add_format({"border":1})
        ifmt=wb.add_format({"border":1,"num_format":"0"}); ffmt=wb.add_format({"border":1,"num_format":"0.0000"})
        pfmt=wb.add_format({"border":1,"num_format":"0.00%"})
        int_cols={"clip_TP","clip_FN","vit_TN","vit_FP","overall_TP","overall_TN","overall_FP","overall_FN"}
        pct_cols={"accuracy","recall","precision","f1"}

        def fmt_write(ws,df,freeze="A2"):
            ws.freeze_panes(freeze); ws.autofilter(0,0,len(df),len(df.columns)-1)
            for ci,cn in enumerate(df.columns):
                ws.write(0,ci,cn,hdr)
                if cn in int_cols: cf,w_=ifmt,12
                elif cn in pct_cols: cf,w_=pfmt,11
                elif cn in {"clip_threshold","vit_threshold"}: cf,w_=ffmt,12
                else: cf,w_=txt,max(12,len(str(cn))+2)
                ws.set_column(ci,ci,w_,cf)
            for ri,(_,row) in enumerate(df.iterrows(),start=1):
                for ci,cn in enumerate(df.columns):
                    if cn in int_cols: f=ifmt
                    elif cn in pct_cols: f=pfmt
                    elif cn in {"clip_threshold","vit_threshold"}: f=ffmt
                    else: f=txt
                    ws.write(ri,ci,row[cn],f)

        # 00_overview
        ow=writer.sheets["00_overview"]
        ow.set_column("A:A",18); ow.set_column("B:B",14); ow.set_column("C:N",12)
        ow.write("A1","Threshold Experiment — Combined 200 Images (Archive + Datasets2_1)",title)
        ow.write("A4","Baseline overall summary",title)
        ow.write(4,0,"model",hdr); ow.write(4,1,"threshold",hdr)
        ow.write(5,0,overall_summary["model"],txt); ow.write(5,1,overall_summary["threshold"],txt)
        ow.write("A7","Overall metrics",title)
        for ci,cn in enumerate(["metric","value"]): ow.write(7,ci,cn,hdr)
        for ri,(_,row) in enumerate(overall_metrics_df.iterrows(),start=8):
            ow.write(ri,0,row["metric"],txt); ow.write(ri,1,row["value"],ifmt if row["metric"] in {"TP","TN","FP","FN"} else ffmt)
        ow.write("A17","Best threshold combinations",title)
        for ci,cn in enumerate(best_df.columns): ow.write(17,ci,cn,hdr)
        for ri,(_,row) in enumerate(best_df.iterrows(),start=18):
            for ci,cn in enumerate(best_df.columns):
                if cn in int_cols: f=ifmt
                elif cn in pct_cols: f=pfmt
                elif cn in {"clip_threshold","vit_threshold"}: f=ffmt
                else: f=txt
                ow.write(ri,ci,row[cn],f)
        ow.write("A30","01_vit_sweep: 固定CLIP=0.80 ViT扫描",note)
        ow.write("A31","02_clip_vit_grid: 126组CLIP×ViT",note)
        ow.write("A32","03_threshold_curves: 曲线图",note)
        ow.write("A33","04_f1_heatmap: F1热力图",note)
        ow.write("A34","05_fp_heatmap: FP热力图",note)
        tf=grid_df.sort_values(by=["f1","precision"],ascending=[False,False]).iloc[0]
        bf=grid_df.sort_values(by=["overall_FP","f1"],ascending=[True,False]).iloc[0]
        ow.write("F1","Best F1",title); ow.write("F2",f"CLIP={tf['clip_threshold']:.2f}, ViT={tf['vit_threshold']:.2f}",txt); ow.write("F3",tf["f1"],pfmt)
        ow.write("H1","Lowest FP",title); ow.write("H2",f"CLIP={bf['clip_threshold']:.2f}, ViT={bf['vit_threshold']:.2f}",txt); ow.write("H3",int(bf["overall_FP"]),ifmt)

        # 01_vit_sweep
        sw=writer.sheets["01_vit_sweep"]; fmt_write(sw,sweep_df)
        ch=wb.add_chart({"type":"line"})
        for col,color in [(10,"#1f77b4"),(12,"#ff7f0e"),(13,"#2ca02c")]:
            ch.add_series({"name":["01_vit_sweep",0,col],"categories":["01_vit_sweep",1,1,len(sweep_df),1],
                "values":["01_vit_sweep",1,col,len(sweep_df),col],"line":{"color":color,"width":2.0},
                "marker":{"type":"circle","size":5,"border":{"color":color},"fill":{"color":color}}})
        ch.set_title({"name":"ViT threshold vs Accuracy / Precision / F1"}); ch.set_x_axis({"name":"ViT threshold"})
        ch.set_y_axis({"name":"Percentage","num_format":"0%","min":0,"max":1}); ch.set_legend({"position":"bottom"})
        ch.set_size({"width":880,"height":420}); sw.insert_chart("P2",ch)

        # 02 grid
        gw=writer.sheets["02_clip_vit_grid"]; fmt_write(gw,grid_df)

        # 03 curves
        c_ws.set_column("A:A",60); c_ws.write("A1","Threshold Curves — 200 Images Combined",title)
        c_ws.write("A2","每条线=一个CLIP threshold；横轴=ViT threshold",note)
        colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b"]; bs=len(VIT_THRESHOLDS)
        for vcol,name_,ylabel,cell,ymin,ymax in [(10,"Accuracy","Percentage","A4",0,1),(12,"Precision","Percentage","J4",0,1),(13,"F1","Percentage","A24",0,1),(8,"Overall FP","Count","J24",None,None)]:
            ch=wb.add_chart({"type":"line"})
            for idx,ct in enumerate(CLIP_THRESHOLDS):
                sr,er=1+idx*bs,idx*bs+bs; c=colors[idx%6]
                ch.add_series({"name":f"CLIP={ct:.2f}","categories":["02_clip_vit_grid",sr,1,er,1],
                    "values":["02_clip_vit_grid",sr,vcol,er,vcol],"line":{"color":c,"width":2.0},
                    "marker":{"type":"circle","size":4,"border":{"color":c},"fill":{"color":c}}})
            ch.set_title({"name":f"ViT threshold vs {name_}"}); ch.set_x_axis({"name":"ViT threshold"})
            ya={"name":ylabel,"num_format":"0%" if ylabel=="Percentage" else "0"}
            if ymin is not None: ya["min"]=ymin
            if ymax is not None: ya["max"]=ymax
            ch.set_y_axis(ya); ch.set_legend({"position":"bottom"}); ch.set_size({"width":720,"height":360})
            c_ws.insert_chart(cell,ch)

        # 04/05 heatmaps
        for ws_,name_,df_,is_f1 in [(f1_ws,"F1 Heatmap",f1_hm,True),(fp_ws,"FP Heatmap",fp_hm,False)]:
            ws_.write("A1",name_,title); ws_.write("A2","行=CLIP threshold, 列=ViT threshold",note)
            df_.to_excel(writer,sheet_name=ws_.name,index=False,startrow=3)
            for ci,cn in enumerate(df_.columns):
                ws_.write(3,ci,cn,hdr); ws_.set_column(ci,ci,12,ffmt if ci==0 else (pfmt if is_f1 else ifmt))
            for ri in range(len(df_)):
                ws_.write(ri+4,0,df_.iloc[ri,0],ffmt)
                for ci in range(1,len(df_.columns)):
                    ws_.write(ri+4,ci,df_.iloc[ri,ci],pfmt if is_f1 else ifmt)
            ws_.freeze_panes("B5")
            ws_.conditional_format(4,1,len(df_)+3,len(df_.columns)-1,
                {"type":"3_color_scale","min_color":"#D73027","mid_color":"#FFFFBF","max_color":"#1A9850"})

    print(f"\nDone! Saved to {OUT_DIR}")
    print(f"Total time: {int(total//60)}m{total%60:.0f}s")

if __name__=="__main__":
    main()
