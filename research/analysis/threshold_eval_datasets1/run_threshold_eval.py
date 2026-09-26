#!/usr/bin/env python3
"""
新实验：
- CLIP：检测 3100 张图片（100张原图×31种变换，排除F2），阈值 0.8，高于判定为相似（正样本）
- ViT：检测 100 张原图之间的 4950 对，阈值 0.5，低于判定为不相似（负样本）
- 统计 TP, TN, FP, FN
"""

import time
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel
from transformers import ViTForImageClassification, ViTImageProcessor

warnings.filterwarnings("ignore")

# 添加项目根目录到路径
project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.file_manager import ProjectPaths, save_dataframe, save_json

# 使用路径管理器
paths = ProjectPaths(project_root)
BASE_DIR = paths.base_dir
ARCHIVE_DIR = BASE_DIR / "Archive"  # 原图
OUTPUT_IMAGE_DIR = BASE_DIR / "data" / "processed"  # 变换后的图
RESULTS_DIR = Path(__file__).parent

# 模型路径
CLIP_MODEL_DIR = BASE_DIR / "models" / "clip-vit-base-patch32"
VIT_MODEL_NAME = "google/vit-base-patch16-224"

# 阈值配置
CLIP_THRESHOLD = 0.8
VIT_THRESHOLD = 0.5

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# 确保结果目录存在
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===================================================================
# 1. 构建数据集
# ===================================================================
def build_evaluation_pairs():
    """
    Returns:
      positive_pairs: 原图和它的变换图（排除F2），共约3100对，标签为1
      negative_pairs: 100张原图之间两两比较，共4950对，标签为0
    """
    # 从 Archive 下所有子文件夹收集原图
    archive_images = {}
    for dir_name in ["ex1_ex2_inputs", "ex2_ai_art", "ex2_car", "ex2_cat", "ex2_china_art", 
                    "ex2_football", "ex2_house", "ex2_nba", "ex2_painting", "ex2_view"]:
        images_subdir = ARCHIVE_DIR / dir_name
        if not images_subdir.exists():
            continue
        for fpath in images_subdir.rglob("*"):
            if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
                stem = fpath.stem
                archive_images[stem] = fpath

    # --- 正样本对：原图 vs 它的变换图（排除F2）---
    positive_pairs = []
    skipped = 0

    for stem, original_path in sorted(archive_images.items()):
        # 查找对应的变换图文件夹
        var_folder = OUTPUT_IMAGE_DIR / stem
        if not var_folder.exists():
            skipped += 1
            continue

        # 查找所有变换图，排除F2
        for var_path in sorted(var_folder.iterdir()):
            if var_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            if var_path.stem.startswith("F2_"):  # 排除垂直翻转
                continue
            positive_pairs.append({
                "path_A": original_path,
                "path_B": var_path,
                "name_A": f"{original_path.parent.name}/{original_path.name}",
                "name_B": f"{var_folder.name}/{var_path.name}",
                "label": 1,  # 是同一张图的变体
            })

    # --- 负样本对：不同原图之间---
    originals = sorted(archive_images.values(), key=lambda p: (p.parent.name, p.name))
    negative_pairs = []
    for i in range(len(originals)):
        for j in range(i + 1, len(originals)):
            negative_pairs.append({
                "path_A": originals[i],
                "path_B": originals[j],
                "name_A": f"{originals[i].parent.name}/{originals[i].name}",
                "name_B": f"{originals[j].parent.name}/{originals[j].name}",
                "label": 0,  # 是不同的图
            })

    return positive_pairs, negative_pairs, skipped


# ===================================================================
# 2. 加载模型
# ===================================================================
def load_models():
    """加载 CLIP 和 ViT 模型"""
    # 加载 CLIP
    clip_path = str(CLIP_MODEL_DIR) if CLIP_MODEL_DIR.exists() else "openai/clip-vit-base-patch32"
    print(f"Loading CLIP from: {clip_path}")
    clip_proc = CLIPProcessor.from_pretrained(clip_path)
    clip_model = CLIPModel.from_pretrained(clip_path).to(DEVICE).eval()

    # 加载 ViT
    print(f"Loading ViT from: {VIT_MODEL_NAME}")
    vit_proc = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()

    return clip_proc, clip_model, vit_proc, vit_model


# ===================================================================
# 3. 计算相似度
# ===================================================================
@torch.no_grad()
def compute_clip_similarity(image_paths_A: list[Path], image_paths_B: list[Path],
                            processor, model) -> np.ndarray:
    """计算 CLIP 余弦相似度"""
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


@torch.no_grad()
def compute_vit_similarity(image_paths_A: list[Path], image_paths_B: list[Path],
                           processor, model) -> np.ndarray:
    """计算 ViT 余弦相似度（使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token）"""
    sims = []
    for i in tqdm(range(0, len(image_paths_A), BATCH_SIZE), desc="  ViT pairs"):
        batch_A = [Image.open(p).convert("RGB") for p in image_paths_A[i:i+BATCH_SIZE]]
        batch_B = [Image.open(p).convert("RGB") for p in image_paths_B[i:i+BATCH_SIZE]]
        inputs_A = processor(images=batch_A, return_tensors="pt")
        inputs_B = processor(images=batch_B, return_tensors="pt")
        inputs_A = {k: v.to(DEVICE) for k, v in inputs_A.items()}
        inputs_B = {k: v.to(DEVICE) for k, v in inputs_B.items()}

        # 使用 ViT 提取特征（取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token）
        outputs_A = model.vit(**inputs_A)
        outputs_B = model.vit(**inputs_B)
        feat_A = outputs_A.last_hidden_state[:, 0, :]
        feat_B = outputs_B.last_hidden_state[:, 0, :]

        fa = torch.nn.functional.normalize(feat_A, p=2, dim=1)
        fb = torch.nn.functional.normalize(feat_B, p=2, dim=1)

        cos_sim = (fa * fb).sum(dim=-1).cpu().numpy()
        sims.append(cos_sim)
    return np.concatenate(sims)


# ===================================================================
# 4. 评估指标
# ===================================================================
def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, threshold: float):
    """计算 TP, TN, FP, FN 和准确率等指标"""
    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = TP + TN + FP + FN
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    accuracy = (TP + TN) / total if total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"{model_name} - Threshold = {threshold}")
    print(f"{'='*60}")
    print(f"  TP: {TP}")
    print(f"  TN: {TN}")
    print(f"  FP: {FP}")
    print(f"  FN: {FN}")
    print(f"  Total: {total}")
    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    return {
        "model": model_name,
        "threshold": threshold,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
    }


# ===================================================================
# 主函数
# ===================================================================
def main():
    print("="*60)
    print("新实验：CLIP(0.8) + ViT(0.5)")
    print("="*60)

    # 1. 构建数据集
    print("\n[1/4] 构建评估数据集...")
    positive_pairs, negative_pairs, skipped = build_evaluation_pairs()
    print(f"  正样本对：{len(positive_pairs)}（原图×31种变换，排除F2）")
    print(f"  负样本对：{len(negative_pairs)}（100张原图两两比较）")
    if skipped:
        print(f"  跳过文件夹：{skipped}")

    # 2. 加载模型
    print("\n[2/4] 加载模型...")
    clip_proc, clip_model, vit_proc, vit_model = load_models()

    # 3. 计算相似度
    print("\n[3/4] 计算相似度...")

    # --- 正样本对 ---
    print(f"\n  --- 正样本对 ({len(positive_pairs)}) ---")
    pos_A = [p["path_A"] for p in positive_pairs]
    pos_B = [p["path_B"] for p in positive_pairs]

    t0 = time.time()
    clip_pos_sims = compute_clip_similarity(pos_A, pos_B, clip_proc, clip_model)
    print(f"  CLIP 计算完成：{time.time() - t0:.1f}s")

    # --- 负样本对 ---
    print(f"\n  --- 负样本对 ({len(negative_pairs)}) ---")
    neg_A = [p["path_A"] for p in negative_pairs]
    neg_B = [p["path_B"] for p in negative_pairs]

    t0 = time.time()
    vit_neg_sims = compute_vit_similarity(neg_A, neg_B, vit_proc, vit_model)
    print(f"  ViT 计算完成：{time.time() - t0:.1f}s")

    # 4. 评估
    print("\n[4/4] 评估...")

    # --- CLIP 正样本评估 ---
    # 正样本：所有都是 label 1
    clip_y_true_pos = np.ones(len(positive_pairs), dtype=int)
    # CLIP: 相似度 >= 0.8 判定为 1
    clip_y_pred_pos = (clip_pos_sims >= CLIP_THRESHOLD).astype(int)

    # --- ViT 负样本评估 ---
    # 负样本：所有都是 label 0
    vit_y_true_neg = np.zeros(len(negative_pairs), dtype=int)
    # ViT: 相似度 < 0.5 判定为 0；否则判定为 1
    vit_y_pred_neg = (vit_neg_sims >= VIT_THRESHOLD).astype(int)

    # --- 合并评估 ---
    # 把所有样本合并在一起
    all_y_true = np.concatenate([clip_y_true_pos, vit_y_true_neg])
    all_y_pred = np.concatenate([clip_y_pred_pos, vit_y_pred_neg])

    # 分别评估和总体评估
    print(f"\n{'='*60}")
    print("CLIP 正样本评估")
    clip_pos_result = evaluate(clip_y_true_pos, clip_y_pred_pos, "CLIP (Positives Only)", CLIP_THRESHOLD)

    print(f"\n{'='*60}")
    print("ViT 负样本评估")
    vit_neg_result = evaluate(vit_y_true_neg, vit_y_pred_neg, "ViT (Negatives Only)", VIT_THRESHOLD)

    print(f"\n{'='*60}")
    print("总体评估（正样本+负样本）")
    overall_result = evaluate(all_y_true, all_y_pred, "Overall (CLIP + ViT)", f"{CLIP_THRESHOLD}/{VIT_THRESHOLD}")

    # 5. 保存结果
    print(f"\n{'='*60}")
    print("保存结果...")

    # 保存总体指标
    save_json([clip_pos_result, vit_neg_result, overall_result], "experiment_summary", RESULTS_DIR)

    # 保存正样本详细数据
    df_pos = pd.DataFrame({
        "name_A": [p["name_A"] for p in positive_pairs],
        "name_B": [p["name_B"] for p in positive_pairs],
        "label": [p["label"] for p in positive_pairs],
        "clip_similarity": clip_pos_sims,
        "prediction": clip_y_pred_pos,
    })
    save_dataframe(df_pos, "positive_pairs", RESULTS_DIR, "csv")

    # 保存负样本详细数据
    df_neg = pd.DataFrame({
        "name_A": [p["name_A"] for p in negative_pairs],
        "name_B": [p["name_B"] for p in negative_pairs],
        "label": [p["label"] for p in negative_pairs],
        "vit_similarity": vit_neg_sims,
        "prediction": vit_y_pred_neg,
    })
    save_dataframe(df_neg, "negative_pairs", RESULTS_DIR, "csv")

    print(f"\n{'='*60}")
    print(f"结果保存到: {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
