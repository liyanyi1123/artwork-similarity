# 艺术品图像相似性研究

这个项目整理了从图像采集、变换实验、相似度算法比较、阈值标定，到艺术品登记原型的完整工作链条。

**先看：[完整工作总览](docs/WORK_OVERVIEW.zh-CN.md)。** 其中按研究内容列出对应代码、已有结果、工作阶段和整理说明。

## 项目包含什么

| 工作 | 内容 |
| --- | --- |
| 数据准备 | Open Images、ArtBench/WikiArt、Pixiv 的采集、校验、补充与整理脚本 |
| 图像变换 | 12 种操作；32 种单操作变体；1–4 个操作的 793 种组合 |
| 算法比较 | CLIP、ViT、aHash、pHash、dHash、wHash、PDQ |
| 阈值实验 | 早期诊断、fallback 补救策略、双模型 AND 规则、完整阈值网格 |
| 统一阈值选择 | 基于 F1 > 0.99 区域大小的跨数据集加权 |
| 分析与展示 | 单模型与联合规则比较、误报漏报、鲁棒性、相关性、计时和实验图 |
| 应用探索 | 向量导出、文件夹比较、Flask API、艺术品登记网页原型 |

## 主要方法和结果

后期联合规则对每一个图像对同时计算两种分数：

```text
CLIP similarity > 0.87 且 ViT similarity > 0.50 → 判为相似
```

模型采用 OpenAI CLIP ViT-B/32，以及 Google ViT-B/16-224 的 CLS 表征。已有代码主要使用预训练特征，没有另行训练模型。

统一阈值在 ArtBench-1K 与 WikiArt-1K 上的结果分别为：

| 本地数据集 | 正样本对 | 负样本对 | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Datasets5` / ArtBench-1K | 31,000 | 499,500 | 121 | 221 | 0.994475 |
| `Datasets6` / WikiArt-1K | 31,000 | 499,500 | 225 | 230 | 0.992661 |

Pixiv-1K 自身网格最优点为 CLIP > 0.96、ViT > 0.76，F1 = 0.947858；直接使用前述统一阈值时 F1 = 0.364273。项目将不同数据域的结果分别标明。

这些数字来自已有配对与缓存分数。阈值搜索和指标计算使用相同配对集合，不能当作独立测试集的泛化结果。ArtBench/WikiArt 名称沿用本地记录，两套采集代码都利用了 ArtBench URL 元数据，名称并不代表已经验证数据集互不重叠。

## 最快运行方式

建议 Python 3.11+。只复核已有结果时，只需要 NumPy，不需要下载图片或模型。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/evaluate.py results/scores/artbench1k.npz
python scripts/evaluate.py results/scores/wikiart1k.npz
python scripts/sweep.py results/scores/pixiv1k.npz --output outputs/pixiv_sweep.csv
python -m unittest discover -s tests -v
```

用自己的图片运行特征提取与比较：

```bash
python -m pip install -r requirements-inference.txt
python scripts/extract_vectors.py examples outputs/reference_vectors --recursive
python scripts/extract_vectors.py examples/synthetic.png outputs/query_vectors
python scripts/compare_vectors.py outputs/query_vectors outputs/reference_vectors \
  --output outputs/comparison.csv
```

首次使用会下载模型；已有本地模型时可通过 `--clip-model`、`--vit-model` 指定目录。

## 怎么找到代码

- `scripts/`：这次整理出的便携运行入口，支持参数化输入输出。
- `research/`：既有研究脚本，保持相对目录结构，方便追溯不同阶段。
- `results/`：精简后的数值分数缓存、原始阈值表、代表性统计。
- `prototype/`：原有网页交互原型；内置示例模拟失败，上传图片模拟成功。
- `docs/`：完整工作总览、实验口径、代码索引、结果来源和验证记录。

`research/` 的历史脚本保留了当时的实验配置，部分还依赖原始数据布局。它们与本次新增入口的运行条件在 [SOURCE_GUIDE.md](docs/SOURCE_GUIDE.md) 中分别说明。网页没有连接模型服务，Flask API 与 DID、区块链架构的实现程度也在文档中明确标注。

## 本次整理的范围

原工作目录未改动。新项目保留研究代码和结果证据，排除大规模图片、变换图片、模型权重、第三方论文和稿件草稿。网页示例替换为可由代码生成的几何图案。

扫描到的 85 个 Python/JavaScript 源文件均有去向记录；其中 4 个属于单独克隆的 National Gallery of Art `opendata` 项目，以外部来源说明保留，不作为本项目原创代码。HTML 原型另行检查。源文件哈希和修改记录位于 `docs/source_manifest.json`。

本项目由本地研究工作区独立整理，发布记录以仓库提交历史为准。项目作者的代码许可证尚未指定；第三方来源及原有许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
