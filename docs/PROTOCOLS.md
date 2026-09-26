# Experiment protocols

The workspace contains several distinct experiments. Their results must be interpreted with their actual prediction rules, score conventions and model versions.

## Prediction rules

| Protocol | Rule | Representative original implementation |
| --- | --- | --- |
| Single-model threshold | Apply the same model to both positive and negative pairs | `research/src/core/threshold.py` |
| Class-specific diagnostic | CLIP on known positive pairs, ViT on known negative pairs | `research/run_threshold_eval_1000_v2.py`, `research/run_datasets5_threshold.py` |
| Fallback analysis | Accept with the main method; try a fallback after failure | `research/src/core/clip_all_fallback_comparison.py` and `vit_all_fallback_4thresholds.py` |
| Joint AND | Both CLIP and ViT must exceed their thresholds on every pair | `research/exp4/run_exp4.py`, `research/exp4/*sweep.py` |
| Joint OR | Either model can exceed its threshold | `research/exp4/logic_compare/run_comparison.py` |

Class-specific diagnostic counts depend on knowing the label before choosing the model. They are not an executable two-stage classifier for an unknown pair. Likewise, positive-transformation fallback success rates alone do not establish a false-positive rate on different originals. The release README leads with the later label-independent AND rule.

## Pair construction

For the three included 1,000-image caches:

- Positive pairs: each original versus 31 of its own transformations, for 31,000 pairs.
- Negative pairs: all unordered pairs of distinct originals, for 1,000 × 999 / 2 = 499,500 pairs.
- F2 vertical flips are generated in the 32-variant transformation set and excluded from these positive pairs.
- Negative labels derive from distinct original-image records. This is not an independently adjudicated copyright-infringement dataset.

The combined ArtBench and WikiArt count is the sum of two within-dataset evaluations. It does not include all cross-dataset original pairs.

## Models and representations

| Model | Primary experiments | Representation |
| --- | --- | --- |
| CLIP | `openai/clip-vit-base-patch32` | Projected image feature, 512 dimensions, L2-normalized |
| ViT | `google/vit-base-patch16-224` | Final hidden-state CLS token, 768 dimensions, L2-normalized |

The local model configuration files confirm those identifiers. The historical `research/final/compare.py` instead defaults to `google/vit-base-patch16-224-in21k`, and `research/app.py` uses the separate OpenAI CLIP loader. Those scripts are preserved as application experiments, not used to generate the release's reported metrics.

The ViT classifier head and pooler output are unused. Loading an image-classification checkpoint into `ViTModel` may report unused classifier weights and newly initialized pooler weights; the extractor reads `last_hidden_state[:, 0, :]`.

## Signed and absolute scores

The original `exp4` sweeps and FP/FN export use signed cosine similarities and strict `>` comparisons. Later `run_single_model_comparison_sep14.py` and method-comparison plotting use absolute cosine similarities. Workflow drawings sometimes use `>=`.

The release keeps signed scores in its archives. All numerical entry points use strict `>`; `--score-mode absolute` explicitly applies an absolute-value transformation. At `(0.87, 0.50)`, the joint-rule counts happen to agree between signed and absolute modes on the included caches. This does not establish equivalence at other thresholds or for single-model baselines.

Stored scores are float32. The original grid uses float64 `np.arange(0.0, 1.01, 0.01)` thresholds. The portable sweep promotes the stored scores for comparison and preserves that grid, including strict boundary behavior. Casting thresholds down to float32 changes a small number of grid rows; the tests compare all 30,603 rows against the original tables.

## Selection and evaluation scope

The high-performance-region procedure counts grid points where each dataset has F1 > 0.99, normalizes the counts into weights, and maximizes the weighted F1 over the common grid. The supplied tables give ArtBench 374 points and WikiArt 243 points; their weighted optimum is CLIP 0.87, ViT 0.50.

The source presentation contains this procedure and its result. `scripts/select_threshold.py` is a new reusable implementation of that existing calculation, using the recorded six-decimal F1 values. It does not create a held-out test split. Per-dataset sweep optima, shared-threshold evaluations and fixed-threshold single-model comparisons are labeled separately in this release.

Historical PR plots use a two-dimensional histogram approximation at bin edges for the AND frontier. Their source and figures are retained; the numerical `scripts/sweep.py` counts threshold passes exactly and is the regression-tested reproduction entry point.

## Application scope

The historical API retrieves CLIP top-3 candidates before checking the two thresholds. This can differ from checking all reference images, as done by `scripts/compare_vectors.py`. The API is not connected to the static prototype. Artist DID checks, ownership reasoning, persistent registration services and blockchain components appear in design documents, without corresponding working implementations in the inspected source tree.
