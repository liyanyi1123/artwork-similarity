# Image Similarity Comparison — 实验数据集与目录说明

## 数据集命名对照

| 新名称 | 描述 | 原始图片数 | 类别 | 来源 |
|---|---|---|---|---|
| **datasets1** | Archive 100 (原始数据集) | 100 张 | 10 类 (AI art, car, cat, china_art, …) | 手动收集，含多种增强变换 |
| **datasets2** | Datasets2_2 | 100 张 | 10 类 (airplane, bird, car, cat, chair, dog, fish, flower, house, tree) | OpenImages，每类 10 张 |
| **datasets3** | Combined 200 (datasets1 + datasets2) | 200 张 | 混合 | datasets1 ∪ datasets2 |
| **datasets4** | Datasets2_1 (1000) | 1000 张 | 10 类 (同上) | OpenImages，每类 100 张 |

## 正负样本构造

- **正样本对 (Positive Pairs)**：每张原始图片与其变换变体（flip, crop, shift, rotate, brightness, vignette, tint, sharpen, keystone, noise 等，共 31 种非 F2 变换）
- **负样本对 (Negative Pairs)**：不同原始图片之间的所有两两组合 C(N, 2)
- 每张原始图片生成 32 种变换（含 F1 + F2 + 10 种操作 × 3 强度），其中 F2 仅用于扩充，不参与正样本对

| 数据集 | 原始图片数 | 正样本对数 | 负样本对数 | 总对数 |
|---|---|---|---|---|
| datasets1 | 100 | 3,100 | 4,950 | 8,050 |
| datasets2 | 100 | 3,100 | 4,950 | 8,050 |
| datasets3 | 200 | 6,200 | 19,900 | 26,100 |
| datasets4 | 1,000 | 31,000 | 499,500 | 530,500 |

## 子目录重命名

| 旧名称 | 新名称 |
|---|---|
| `threshold_eval_clip08_vit05_archieve_100images` | `threshold_eval_datasets1` |
| `threshold_eval_clip08_vit05_datasets2_2_100images` | `threshold_eval_datasets2` |
| `threshold_eval_clip08_05_200images` | `threshold_eval_datasets3` |
| `threshold_eval_clip08-vit05_datasets2_1_1000images` | `threshold_eval_datasets4` |

其他目录保持不变：`similarity_matrices/`、`similarity_7algo/`、`summary_plots_archieve_100images/`、`summary_all_experiments/`、`threshold_evaluation/`

## 实验方法：双阈值分类

核心策略：**CLIP 检测正样本（高阈值判相似）+ ViT 检测负样本（低阈值判不相似）**

```
Stage 1 (CLIP): 对正样本对，sim >= clip_threshold → TP, sim < clip_threshold → FN
Stage 2 (ViT):  对负样本对，sim <  vit_threshold → TN, sim >= vit_threshold → FP
```

### 各数据集最佳阈值与结果

| 数据集 | CLIP 阈值 | ViT 阈值 | F1 | 准确率 | 假阳性 |
|---|---|---|---|---|---|
| datasets1 | 0.80 | 0.50 | 0.9778 | 98.25% | 140 |
| datasets2 | 0.75 | 0.70 | 0.9995 | 99.96% | 21 |
| datasets3 | 0.80 | 0.50 | 0.9857 | 99.31% | 176 |
| datasets4 | 0.80 | 0.50 | 0.9756 | 99.71% | 1,532 |

### 跨数据集对比

| 实验 | 正样本对 | 负样本对 | CLIP mean sim | CLIP min sim | CLIP <0.8 | ViT mean sim | ViT max sim | ViT >0.5 | F1 | Acc | FP | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| datasets1 | 3100 | 4950 | 0.9793 | 0.7289 | 1 | 0.0991 | 0.7712 | 140 | 0.9568 | 0.9825 | 140 | 3099 |
| datasets2 | 3100 | 4950 | 0.9812 | 0.7236 | 3 | 0.0436 | 0.7301 | 21 | 0.9961 | 0.9970 | 21 | 3097 |
| datasets3 | 6200 | 19900 | 0.9803 | 0.7236 | 4 | 0.0572 | 0.7712 | 176 | 0.9857 | 0.9931 | 176 | 6196 |
| datasets4 | 31000 | 499500 | 0.9810 | 0.5342 | 20 | 0.0476 | 0.9231 | 1532 | 0.9756 | 0.9971 | 1532 | 30980 |

## `record/` 目录

保存 4 个数据集的 CLIP 正样本检测结果和 ViT 负样本检测结果，共 8 个 CSV 文件：

| 文件名 | 内容 | 行数 | 列 |
|---|---|---|---|
| `positive_pairs_datasets1.csv` | datasets1 CLIP 正样本检测 | 3,100 | name_A, name_B, label, clip_similarity, prediction |
| `negative_pairs_datasets1.csv` | datasets1 ViT 负样本检测 | 4,950 | name_A, name_B, label, vit_similarity, prediction |
| `positive_pairs_datasets2.csv` | datasets2 CLIP 正样本检测 | 3,100 | name_A, name_B, label, clip_similarity |
| `negative_pairs_datasets2.csv` | datasets2 ViT 负样本检测 | 4,950 | name_A, name_B, label, vit_similarity |
| `positive_pairs_datasets3.csv` | datasets3 CLIP 正样本检测 | 6,200 | name_A, name_B, label, clip_similarity |
| `negative_pairs_datasets3.csv` | datasets3 ViT 负样本检测 | 19,900 | name_A, name_B, label, vit_similarity |
| `positive_pairs_datasets4.csv` | datasets4 CLIP 正样本检测 | 31,000 | name_A, name_B, label, clip_similarity |
| `negative_pairs_datasets4.csv` | datasets4 ViT 负样本检测 | 499,500 | name_A, name_B, label, vit_similarity |

## 涉及的算法

| 类别 | 算法 | 用途 |
|---|---|---|
| 深度学习 | CLIP (ViT-B/32) | 正样本检测（高阈值） |
| 深度学习 | ViT (DINO/MAE) | 负样本检测（低阈值） |
| 感知哈希 | aHash, pHash, dHash, wHash | 相似度矩阵对比 |
| 工业级 | PDQ (Facebook) | 相似度矩阵对比 |

## 7 种算法相关性

CLIP 与 ViT 中度相关 (r=0.60)；传统哈希方法 (aHash/pHash/dHash/wHash) 彼此高度相关 (r=0.28-0.88)，但与深度学习方法几乎不相关 (r<0.03)；PDQ 与其他方法均弱相关。

---

*生成日期: 2026-08-03*
