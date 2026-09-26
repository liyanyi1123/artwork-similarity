# Results index

| Location | Contents |
| --- | --- |
| `scores/` | Three paired CLIP/ViT numeric archives, sufficient for offline decision-stage reproduction |
| `reproduced_summary.json` | Shared-threshold results, dataset-specific optima and weighting recomputed during packaging |
| `reported/exp4/threshold_eval/full_sweep/` | Original ArtBench-1K 10,201-point AND sweep |
| `reported/exp4/datasets6_sweep/` | Original WikiArt-1K 10,201-point AND sweep |
| `reported/exp4/datasets7_sweep/` | Original Pixiv-1K 10,201-point AND sweep |
| `reported/exp4/logic_compare/` | Historical fixed-threshold AND/OR/single-model comparison at (0.80, 0.50) |
| `reported/analysis/ComparisonSep14/` | Later single-model comparison using absolute scores at (0.87, 0.50) |
| `reported/analysis/threshold_eval_datasets*/` | Selected early class-specific diagnostic tables |
| `reported/analysis/similarity_7algo/` | Small seven-method comparison tables |
| `reported/analysis/similarity_matrices/` | Method-correlation table |
| `reported/analysis/Papersep11/FNnFP/` | FP/FN count record; images omitted |
| `reported/experiments/time/` | Historical runtime measurements |

Preserved outputs represent different experiment stages. They should not be merged into a single performance table without consulting [the protocol guide](../docs/PROTOCOLS.md). Pair-score provenance is documented in [score_provenance.json](../docs/score_provenance.json).
