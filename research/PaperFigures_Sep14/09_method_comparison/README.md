# 方法对比图 —— "我们的方法 vs 以往方法"

生成脚本：`make_method_comparison.py`（一条命令重出全部三张图）

```bash
<local-path> make_method_comparison.py
```

---

## 一、数据来源与可信度

全部为**本地真实数据**，不是示意值：

| 项 | 值 |
|---|---|
| 数据集 | `analysis/threshold_eval_datasets5/`（1000 件作品） |
| 配对总数 | 530,500（正 31,000 / 负 499,500） |
| 相似度来源 | `clip_embeddings.npz`、`vit_embeddings.npz`、`vit_neg_sims.npy`、`exp4/results/positive_vit_similarities.npy` |
| 阈值 | CLIP > 0.87 AND ViT > 0.50（论文冻结值） |

**交叉校验通过**：脚本算出的联合规则在 (0.87, 0.50) 下 FP=121、FN=221、F1=0.994475，与论文 Table 4 的 ArtBench-1K 行**完全一致**。这也反证了本地 Datasets5 就是论文里的 ArtBench-1K。

---

## 二、三张图各自的作用

### 图 A `figA_rule_comparison.png` —— 结果对比
- (a) 同一冻结操作点下，四种判定规则的 Precision / Recall / F1
- (b) 误判数量对比（对数刻度），FP 与 FN 分开

**看点**：联合规则的 FP 是 121，而 ViT 单路 10,972、OR 规则 11,187 —— 相差约 90 倍，在对数刻度上一眼可见。

### 图 B `figB_decision_space.png` —— 机制解释（最有说服力）
横轴 CLIP 相似度、纵轴 ViT 相似度，把每个配对画成一个点：
- 灰色：负样本（不同作品），集中在左下
- 绿色：正样本（同作品的变换版本），集中在右上
- 橙色：**215 个负样本，CLIP 单路会误收，但被 ViT 条件挡住**
- 深红：**121 个两路都越线的误报**，全部紧贴边界
- 蓝色矩形：联合匹配区

**看点**：这张图把"为什么 AND 有效"变成几何事实 —— 橙色那 215 个点全部落在 CLIP 边界右侧、ViT 边界下方，视觉上直接说明 ViT 条件在替 CLIP 兜底。

### 图 C `figC_precision_recall.png` —— 公平对比
- (a) 三条 PR 曲线（各自扫遍自己的阈值），放大到高精度区
- (b) 各方法在自己最优阈值下的最佳 F1

**看点**：蓝色（联合规则）曲线全程压在橙色（CLIP 单路）之上，橙色又压着红色（ViT 单路）—— 标准的"支配关系"表达。

---

## 三、关键数字（可直接引用）

### 同一操作点 (0.87, 0.50)

| 规则 | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| **AND（我们）** | **0.9961** | 0.9929 | **0.9945** | **121** | 221 |
| CLIP 单路 | 0.9892 | 0.9947 | 0.9919 | 336 | 165 |
| ViT 单路 | 0.7382 | 0.9978 | 0.8486 | 10,972 | 67 |
| OR | 0.7348 | 0.9996 | 0.8470 | 11,187 | 11 |

### 各方法在各自最优阈值下

| 方法 | 最佳 F1 | 最优阈值 |
|---|---|---|
| **联合 AND（我们）** | **0.9945** | t_C=0.87, t_V=0.50 |
| CLIP 单路 | 0.9931 | t=0.88 |
| ViT 单路 | 0.9845 | t=0.686 |

### 误差拆解（可直接写进正文）

- CLIP 单路的 336 个误报 = **215 个被 ViT 条件挡掉** + 121 个两路都越线
- 联合规则的 221 个漏检 = **154 个被 CLIP 条件挡掉** + 67 个 ViT 单路本来也会漏

---

## 四、必须注意的两件事

**1. 相对 CLIP 单路的优势很薄，措辞要小心。**
在各自最优阈值下，联合规则 F1 = 0.9945 vs CLIP 单路 0.9931，只高 **0.0014**（相对提升 0.14%）。论文摘要说 "improves the precision–recall trade-off over either encoder alone"，字面没错，但审稿人一定会追问这个差距是否显著。
**更稳的表述**：联合规则相对 ViT 单路把误报从 10,972 降到 121（约 90 倍），同时相对 CLIP 单路把误报从 336 降到 121（约 2.8 倍），召回率仅下降 0.0018 —— 即"以可忽略的召回损失换取数量级级别的误报削减"。这个说法既准确又更有力。

**2. 目前只有 Datasets5（ArtBench-1K 口径）一套。**
论文 Table 3 的基线对比要用 ArtBench-1K + WikiArt-1K 的 pooled 结果，并且要加上 pHash / PDQ 两个哈希基线。投稿前需要：
- 把 `DATASETS` 换成两套数据集并做 pooled；
- 补 pHash / PDQ 的相似度数组（目前哈希基线数据在 `analysis/similarity_matrices/`，是另一批图，不能直接混用）。

---

## 五、投稿版图注建议

- 图 A/B：**Figure 5. Decision-rule comparison at the frozen operating point.** ...
- 图 C：**Figure 6. Precision–recall behaviour when each method selects its own threshold.** ...

三张图都按 300 dpi 输出，配色为蓝（本文方法）/ 橙 / 红（基线），黑白打印下靠线型和填充区分，不依赖颜色。
