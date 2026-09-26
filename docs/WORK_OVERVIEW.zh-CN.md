# 工作全景：艺术品图像相似性与登记前核验

这份文件将工作目录中的代码、实验结果和应用原型串联起来。它可以作为 GitHub 项目的工作说明，也可以用于介绍研究经历。工作主线是：**构建变换与配对基准，比较图像表征，形成双模型核验规则，再探索登记前相似性检查的应用流程。**

## 1. 数据采集与实验集构建

工作包含多种图像来源和不同规模的实验设置。

| 数据类别 | 代码位置（相对于 `research/`） | 完成内容 |
| --- | --- | --- |
| Open Images | `download_openimages_100.py`、`downloader.py` | 按类别下载、从每类 10 张扩展至 100 张、记录来源 |
| ArtBench 风格采样 | `Datasets5/download_100_per_style.py`、`Datasets5/fix_ukiyoe.py` | 按 10 种风格采集，校验并替换损坏图片 |
| WikiArt URL 图像 | `Datasets6/download_wikiart_1000.py` | 从 ArtBench CSV 中的 WikiArt URL 采集并记录来源 |
| Pixiv 插画 | `download_pixiv_1000.py`、`download_pixiv_supplement.py`、`pixiv/download_pixiv_first_1000.py` | 下载、补齐、去除已有 ID；使用 HTTP Range 读取远程 ZIP 的早期文件 |

早期工作使用 Archive 100、Open Images 100、两者合并的 200 张和 1,000 张实验；后续扩展到艺术风格图像与插画域。具体映射见 [DATA.md](DATA.md)。本次未重新下载或扩充任何数据。

`opendata/` 是 National Gallery of Art 的独立外部仓库。其数据处理代码在盘点中登记，但没有混入本项目贡献列表。Google Open Images 下载器保留其原作者和许可声明。

## 2. 图像变换基准

建立了 12 种图像操作，包括水平/垂直翻转、中心裁剪与缩放、水平/垂直平移、旋转、亮度与对比度、暗角、色温、锐化、透视扰动和噪声。

形成两类互补实验：

- **组合变换**：选择 1–4 个操作，组合数量为 12 + 66 + 220 + 495 = 793。对应 `src/core/generate_ex1.py` 与 `ex1_2/generate_ex1.py`。
- **强度分级**：两个翻转各生成一个版本，其余十个操作各生成 low/mid/high 三档，共 32 个版本。对应 `generate_ex2.py`、`generate_datasets5_transforms.py` 与 `exp4/generate_datasets6_transforms.py`、`generate_datasets7_transforms.py`。

这提供了从单因素扰动到多操作叠加的鲁棒性评估素材。不同实验脚本的强度与文件命名存在版本差异，因此保留各版实现。本次新增 `scripts/generate_transforms.py` 复用 Datasets5 操作，增加输入输出参数和变换清单。

后期正样本协议排除 F2 垂直翻转，保留每张原图的 31 个变体。32 是生成数量，31 是该协议中的评估数量。

## 3. 七种相似度方法的系统比较

比较了深度特征 CLIP、ViT，以及 aHash、pHash、dHash、wHash、PDQ 五种感知哈希。

对应实现包括：

- `src/core/auto_perception_ex1.py`、`auto_perception_ex2.py`：原图与变换图的分数计算、报告和缩略图。
- `src/core/similarity.py`：七种方法的原图两两相似度矩阵、类别统计、相似样本对。
- `src/pipelines/run_ex3.py`、`resume_ex3.py`、`run_7algo_similarity.py`：批量计算、断点延续、方法比较。
- `src/plotting/`：分操作、分强度、分模型和汇总图表。

本项目的特征提取使用预训练模型。已有代码没有实现新的模型训练过程。ViT 的具体版本需要区别：主实验本地配置是 `google/vit-base-patch16-224`；`final/compare.py` 的默认值则是 `google/vit-base-patch16-224-in21k`，因此不能直接把该脚本的输出当作主实验复现。

## 4. 从诊断实验到联合判定规则

阈值相关工作经历了不同阶段，必须按实际代码区分。

| 阶段 | 代表代码 | 实际计算方式 |
| --- | --- | --- |
| 早期单模型阈值 | `src/core/threshold.py` | 分别考察 CLIP、ViT 对正负配对的判定 |
| 正负类别分别计分 | `run_threshold_eval_1000*.py`、`run_datasets5_threshold.py`、`run_threshold_sweep.py` | 正样本用 CLIP 计 TP/FN，负样本用 ViT 计 FP/TN |
| fallback 补救 | `clip_all_fallback_comparison.py`、`vit_all_fallback_4thresholds.py` | 主模型未通过时由另一方法补救，包含针对变换正样本的召回统计 |
| 每对联合 AND | `exp4/run_exp4.py`、`threshold_eval/full_sweep_0_1.py`、`run_datasets6_sweep.py`、`run_datasets7_sweep.py` | 每个正负图像对均计算两种分数，同时超过阈值才判相似 |

早期“已知正负类别后选模型”的统计可以描述两种表征的诊断性能，但它本身不能直接作为未知图像对的分类器。新 README 以最后一类联合规则介绍主要方法，保留早期工作作为实验演进。

还实现了 AND、OR、CLIP-only、ViT-only 的比较，以及各模型阈值扫描和决策空间可视化。对应 `exp4/logic_compare/run_comparison.py`、`run_single_model_comparison_sep14.py` 与 `PaperFigures_Sep14/09_method_comparison/make_method_comparison.py`。

## 5. 跨数据集阈值搜索与统一阈值选择

后期实验在 CLIP、ViT 各 0.00–1.00、步长 0.01 的网格上搜索，共 10,201 组。每个 1,000 张实验使用 31,000 个正样本对和 499,500 个负样本对。

原工作还提出并展示了基于高性能区域大小的权重：统计各数据集 F1 > 0.99 的阈值对数量，以数量占比为权重，再最大化加权 F1。

| 数据集 | F1 > 0.99 的网格点 | 权重 | 统一阈值下 F1 |
| --- | ---: | ---: | ---: |
| ArtBench-1K / Datasets5 | 374 | 0.606159 | 0.994475 |
| WikiArt-1K / Datasets6 | 243 | 0.393841 | 0.992661 |

已有表格复核得到统一阈值 CLIP > 0.87、ViT > 0.50，加权 F1 = 0.993761（按原表六位小数计算）。相关工作原来分布在幻灯片源代码与阈值 CSV 中；本次增加 `scripts/select_threshold.py`，把已有计算方案整理为可运行入口。

Pixiv-1K 的自身网格最优点为 (0.96, 0.76)，F1 = 0.947858。它也记录了艺术域统一阈值在插画域上的变化，不能与前两行合并成同一阈值结果。

## 6. 错误案例与实验解释

对误报、漏报的整理不仅包含统计，也包含具体图像对的导出和来源校验。

- `analysis/Papersep11/export_fn_fp.py`：按冻结阈值导出 FP/FN 图像对，检查配对顺序、数目和 SHA-256。
- ArtBench-1K：FP = 121，FN = 221。
- WikiArt-1K：FP = 225，FN = 230。
- 合计：346 个 FP 图像对、451 个 FN 图像对。
- `gen_top10_vit_ds5.py`、`exp4/plot_datasets7_*`：高相似负样本和低相似配对的可视化。
- 方法对比绘图：PR 曲线、决策空间、误判构成。

本次保留错误导出代码及统计，未将大量作品图像打入发布包。部分历史图使用绝对余弦分数、不同阈值或分箱近似；[PROTOCOLS.md](PROTOCOLS.md) 标明其口径。

## 7. 性能优化与运行工程

已有实现包括批量特征提取、向量缓存、归一化、矩阵/向量运算、变换生成并行处理、重启恢复，以及逐阶段计时。

代表文件为 `run_threshold_eval_1000_v2.py`、`src/pipelines/benchmark_pipeline.py`、`resume_ex3.py`、`Aug27/extract_clip_vit_vectors.py`。计时记录和代表图保存在 `results/reported/experiments/time/` 与 `docs/figures/pipeline_timing.png`。

计时是当时环境下的历史记录。本次只运行小规模验证，未把原始计时重新标为当前硬件结果。

## 8. 应用代码与登记原型

工作已从实验脚本扩展到面向应用的接口和交互。

| 实现 | 已有能力 | 当前边界 |
| --- | --- | --- |
| `Aug27/extract_clip_vit_vectors.py` | 图片/文件夹输入，导出归一化 CLIP 与 ViT 向量 | 本次已整理为参数化公开模型入口 |
| `final/compare.py` | 两个文件夹的笛卡尔积比较 | 默认 ViT 模型和阈值与主实验不同 |
| `app.py` | Flask `/predict`、`/update`、`/health`；检索及参考向量更新 | 按 CLIP top-3 再判阈值，模型加载和数据布局固定，属于历史服务原型 |
| `prototype/index.html` | 上传、预览、资料输入、模拟登记成功/失败 | 无真实推理、身份认证或数据库连接 |
| PRD 与工作流文档 | 提出 DID、登记记录、模型核验等应用结构 | DID/区块链流程属于设计说明，未找到相应服务实现 |

本次保留原有界面交互，将默认展示图片换成代码生成的几何图案。模拟登记编号与判定逻辑仍是原型行为。

## 9. 本次整理新增了什么

新增内容是对已有工作的发布整理，未增加新的研究实验结果：

1. 中英文 README、本工作总览、代码与数据索引、实验口径说明。
2. 原始代码清单、重复文件记录、原文件与打包文件的 SHA-256 映射。
3. 便携的结果评估、网格搜索、加权选阈值、变换生成和向量比较入口。
4. 从原缓存派生的三个精简分数包，去除绝对文件路径和原图。
5. 分层依赖、运行参数、Git 忽略规则和可执行回归检查。

历史代码中几处个人绝对路径在副本中改为相对路径。其余实验逻辑保留。完全相同的代码没有直接删除，因为不同路径仍被历史脚本引用；重复关系见 [SOURCE_GUIDE.md](SOURCE_GUIDE.md)。

## 10. 项目包的阅读顺序

对外展示从 `README.md` 进入；核对本人工作从本文件进入；运行与复核从 [REPRODUCIBILITY.md](REPRODUCIBILITY.md) 进入。希望找具体脚本时看 [SOURCE_GUIDE.md](SOURCE_GUIDE.md)，希望确认数字出处时看 `score_provenance.json` 和 `results/reported/`。

范围覆盖当前工作区可见的 85 个 Python/JavaScript 源文件，以及 HTML 原型、图集和关键实验文档；没有恢复 Git 历史中已删除的旧文件。压缩包 `Aug27.zip` 中的旧提取脚本已有较新的同名版本，以当前文件为准。原研究目录及其已有 Git 状态均未改动。
