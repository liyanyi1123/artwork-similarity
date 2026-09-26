# ComprehensiveAnalysisAug3 — 综合实验分析结果

> 生成日期：2026-08-03  
> 数据集范围：datasets1–5（共 5 个数据集，2,300 张原始图片，1,093,900 对正负样本）

---

## 目录结构总览

```
ComprehensiveAnalysisAug3/
├── README.md                                      ← 本文件
├── all_raw_data.xlsx                              # 全部正负样本对原始数据
├── group_vit_similarity_comparison.xlsx           # datasets4/5 按类别分组的 ViT 相似度对比
├── threshold_evaluation_4datasets.xlsx            # datasets1-4 双阈值网格搜索结果
├── generate_threshold_excel.py                    # 生成阈值评估表格的 Python 脚本
├── vit_distribution/                              # ViT 负样本相似度分布直方图
├── clip_distribution/                             # CLIP 正样本相似度分布直方图
├── top10_vit_negative_pairs/                      # ViT 相似度最高 10 对负样本纵向图
├── bottom10_clip_positive_pairs/                  # CLIP 相似度最低 10 对正样本纵向图
└── transforms_cache/                              # 按需生成的 transform 图像缓存，top10的vit最高相似度生成时用到
```

---

## 文件详解

### 📊 Excel 数据表

#### `all_raw_data.xlsx`（34 MB）
全部 5 个数据集的完整正负样本对数据，共 **11 个 sheet**：

| Sheet | 内容 | 行数 |
|---|---|---|
| `overview` | 5 个数据集概览统计 | 5 |
| `positive_datasets1` | Archive 100 — CLIP 正样本对 | 3,100 |
| `negative_datasets1` | Archive 100 — ViT 负样本对 | 4,950 |
| `positive_datasets2` | Datasets2_2 (100) — CLIP 正样本对 | 3,100 |
| `negative_datasets2` | Datasets2_2 (100) — ViT 负样本对 | 4,950 |
| `positive_datasets3` | Combined 200 — CLIP 正样本对 | 6,200 |
| `negative_datasets3` | Combined 200 — ViT 负样本对 | 19,900 |
| `positive_datasets4` | Datasets2_1 (1000) — CLIP 正样本对 | 31,000 |
| `negative_datasets4` | Datasets2_1 (1000) — ViT 负样本对 | 499,500 |
| `positive_datasets5` | ArtBench-10 (1000) — CLIP 正样本对 | 31,000 |
| `negative_datasets5` | ArtBench-10 (1000) — ViT 负样本对 | 499,500 |

**总数据量：1,103,250 对**（43,400 正样本 + 1,059,850 负样本）

#### `group_vit_similarity_comparison.xlsx`（14 KB）
datasets4 和 datasets5 按 **55 个对比组**（10 组内 + 45 组间）统计 ViT 负样本相似度，各含均值/中位数/最大/最小/标准差：

| Sheet | 数据集 |
|---|---|
| `datasets4 (自然照片)` | 10 类自然物体 → 55 组 |
| `datasets5 (艺术画作)` | 10 类艺术风格 → 55 组 |

组内行以粉色高亮，按平均相似度降序排列。

#### `threshold_evaluation_4datasets.xlsx`
datasets1-4 的 **CLIP + ViT 双阈值网格搜索**结果，4 个 sheet：

| Sheet | 数据集 | CLIP 阈值范围 | ViT 阈值范围 |
|---|---|---|---|
| `Datasets1 (Archive 100)` | 100 张 | 0.80–0.85 | 0.30–0.70 |
| `Datasets2 (OpenImages 100)` | 100 张 | 0.70–0.90 | 0.50–0.85 |
| `Datasets3 (Combined 200)` | 200 张 | 0.80–0.85 | 0.35–0.65 |
| `Datasets4 (OpenImages 1000)` | 1,000 张 | 0.80–0.85 | 0.30–0.60 |

每行记录一个阈值组合下的 TP/TN/FP/FN/Accuracy/Recall/Precision/F1。

---

### 📈 相似度分布直方图

#### `vit_distribution/`
| 文件 | 内容 |
|---|---|
| `negative_pairs_similarity_histograms.png` | datasets1-4 **ViT 负样本**相似度分布四合一图，高度右偏 L 型 |
| `datasets5_vit_negative_histogram.png` | datasets5 **ViT 负样本**分布图，类正态，峰值 0.1–0.2 |

**关键发现**：datasets1-4 约 80% 负样本集中在 [0, 0.1)，而 datasets5 仅 17%，均值从 0.062 跃升至 0.215（3.5×）。

#### `clip_distribution/`
| 文件 | 内容 |
|---|---|
| `positive_pairs_similarity_histograms.png` | datasets1-4 **CLIP 正样本**相似度分布四合一图 |
| `datasets5_clip_positive_histogram.png` | datasets5 **CLIP 正样本**分布图 |

**关键发现**：CLIP 在各数据集上表现一致（均值 0.977–0.981），对自然照片和艺术画作的 transform 检测能力相当。

---

### 🖼️ 极端样本对可视化

#### `top10_vit_negative_pairs/`
每个数据集纵向排列 ViT 相似度**最高**的 10 对负样本（每行 1 组，图片 A/B + 文件名 + 相似度）：

| 文件 | 数据集 | 最高 sim |
|---|---|---|
| `top10_vit_negative_pairs_datasets1.png` | Archive 100 | cat3↔cat9: 0.7712 |
| `top10_vit_negative_pairs_datasets2.png` | Datasets2_2 (100) | airplane06↔airplane08: 0.7301 |
| `top10_vit_negative_pairs_datasets3.png` | Combined 200 | cat3↔cat9: 0.7712 |
| `top10_vit_negative_pairs_datasets4.png` | Datasets2_1 (1000) | fish04↔fish71: **0.9231** |
| `top10_vit_negative_pairs_datasets5.png` | ArtBench-10 (1000) | renaissance↔renaissance: **0.8581** |

#### `bottom10_clip_positive_pairs/`
每个数据集纵向排列 CLIP 相似度**最低**的 10 对正样本（原图 vs transform 变体，CLIP 未能正确识别为相似）：

| 文件 | 数据集 | 最低 sim | 缺失 Transform |
|---|---|---|---|
| `bottom10_clip_positive_pairs_datasets1.png` | Archive 100 | 0.7289 | 部分（已补） |
| `bottom10_clip_positive_pairs_datasets2.png` | Datasets2_2 (100) | 0.7236 | ✅ 已全部补全 |
| `bottom10_clip_positive_pairs_datasets3.png` | Combined 200 | 0.7236 | 部分（已补） |
| `bottom10_clip_positive_pairs_datasets4.png` | Datasets2_1 (1000) | 0.5342 | ✅ 全部存在 |

---

### 🛠️ 其他

#### `transforms_cache/`
为 datasets1-3 的 `bottom10_clip_positive_pairs` 按需生成的 transform 图像缓存。datasets1-2 原始实验中 transform 为动态生成未持久化，此处仅缓存了 bottom-10 所需的变体图片。

#### `generate_threshold_excel.py`
生成 `threshold_evaluation_4datasets.xlsx` 的 Python 脚本，从 `analysis/threshold_eval_datasets{1-4}/threshold_grid.csv` 读取网格搜索结果并汇入一个 Excel 文件。

---

## 数据集速查

| 代号 | 名称 | 原始数 | 类别 | 领域 |
|---|---|---|---|---|
| datasets1 | Archive 100 | 100 | 10 类混合 | 自然照片 |
| datasets2 | Datasets2_2 | 100 | 10 类 OpenImages | 自然照片 |
| datasets3 | Combined 200 | 200 | datasets1 + datasets2 | 自然照片 |
| datasets4 | Datasets2_1 | 1,000 | 10 类 OpenImages | 自然照片 |
| datasets5 | ArtBench-10 | 1,000 | 10 类艺术风格 | 绘画作品 |
