# Source guide

The inventory covers 85 Python/JavaScript files in the original workspace, plus the HTML prototype and figure gallery. All files were read for structural inventory, imports, paths and provenance; core decision, cache, extraction and application code received targeted review. This is a packaging audit, not a claim of line-by-line formal verification.

## Release entry points

| Entry point | Purpose | Dependencies |
| --- | --- | --- |
| `scripts/evaluate.py` | Fixed thresholds, AND/OR/single models | NumPy |
| `scripts/sweep.py` | Strict-AND 101 × 101 grid | NumPy |
| `scripts/select_threshold.py` | Weighting from existing grid tables | Python standard library |
| `scripts/extract_vectors.py` | Existing Aug27 extractor with portable model defaults | Inference requirements |
| `scripts/compare_vectors.py` | Cartesian comparison of exported vectors | NumPy |
| `scripts/generate_transforms.py` | Original Datasets5 operations with configurable I/O | NumPy, OpenCV |

## Historical source inventory

Paths below are original workspace paths. Included sources are under `research/` at the same relative path, except the separately packaged prototype. See `source_manifest.json` for checksums and exact destinations.

| Original path | Lines | Role from source |
| --- | ---: | --- |
| `.ppt_build_weighting/build_slide.mjs` | 255 | Presentation construction |
| `Aug27/extract_clip_vit_vectors.py` | 205 | Extract normalized CLIP and ViT image vectors from an image directory. |
| `Aug27/legacy_code/run_datasets6_sweep.py` | 361 | Compute CLIP/ViT similarities for Datasets6 and run full threshold sweep (0-1, 0.01). |
| `Aug27/legacy_code/run_datasets7_sweep.py` | 374 | Compute CLIP/ViT similarities for Datasets7 (pixiv1000) and run full threshold sweep (0-1, 0.01). |
| `Datasets5/download_100_per_style.py` | 113 | Download 100 images per art style from ArtBench-10 CSV. |
| `Datasets5/fix_ukiyoe.py` | 110 | Parallel download + validate ukiyo_e images, replacing corrupted ones. |
| `Datasets6/download_wikiart_1000.py` | 163 | Download 1000 images from wikiart CDN using ArtBench-10.csv URLs. |
| `PaperFigures_Sep14/02_fig2_workflow/system_implementation_flowchart.py` | 261 | rounded_box, panel, arrow |
| `PaperFigures_Sep14/09_method_comparison/make_method_comparison.py` | 295 | 方法对比图 v2 —— 补充 PR 曲线与公平对比 |
| `analysis/ComprehensiveAnalysisAug3/generate_threshold_excel.py` | 263 | Generate a 4-sheet Excel file: threshold evaluation summary for datasets1-4. |
| `analysis/Papersep11/export_fn_fp.py` | 237 | Export the paper's FP/FN image pairs at CLIP > 0.87 AND ViT > 0.50. |
| `analysis/threshold_eval_datasets1/export_overall_summary_to_excel.py` | 532 | 导出阈值实验结果到 Excel，包括： |
| `analysis/threshold_eval_datasets1/plot_vit_similarity_ai_art1_vs_others.py` | 138 | Plot ViT similarity between ai_art1.jpg and the other 99 original archive images. |
| `analysis/threshold_eval_datasets1/run_threshold_eval.py` | 339 | 新实验： |
| `analysis/threshold_evaluation/src/clip_all_fallback_comparison.py` | 470 | Compare CLIP baseline and CLIP+6 fallback strategies in one table and one chart. |
| `analysis/threshold_evaluation/src/threshold_evaluation.py` | 349 | Threshold Evaluation @ 0.87 — CLIP + pHash fallback for positives |
| `analysis/threshold_evaluation/src/vit_all_fallback_4thresholds.py` | 352 | Use ViT as the baseline and compare six fallback strategies across four thresholds. |
| `app.py` | 216 | get_clip_embedding, get_vit_embedding, compute_reference_embeddings, load_reference_embeddings, append_embedding |
| `docs/ppt-output/runs/20260629-image-perception/generate_ppt.py` | 437 | Generate a 4-slide academic PPTX about image perception comparison experiments. |
| `download_openimages_100.py` | 287 | Expand existing 10-class Open Images dataset from 10 to 100 images per class. |
| `download_pixiv_1000.py` | 129 | Download 1000 Pixiv images using parallel downloads via pixiv.cat proxy. |
| `download_pixiv_supplement.py` | 177 | Supplement datasets7 to 1000 Pixiv images. |
| `downloader.py` | 129 | Open Images image downloader. |
| `ex1_2/Similarity_compare/clip_similarity.py` | 469 | CLIP, ViT & Perceptual Hash Image Similarity Comparison |
| `ex1_2/Similarity_compare/threshold_evaluation.py` | 309 | Threshold Evaluation @ 0.9 — CLIP & ViT |
| `ex1_2/auto_perception_ex1.py` | 277 | get_image_files, get_hashers, precompute_hashes, compute_distances, get_vit_embedding |
| `ex1_2/auto_perception_ex2.py` | 322 | get_image_files, get_hashers, precompute_hashes, compute_distances, get_vit_embedding |
| `ex1_2/benchmark_pipeline.py` | 137 | Run the ex1_2 pipeline for multiple batch sizes with precise per-step timing. |
| `ex1_2/generate_ex1.py` | 185 | 12 base image operations — generate all combinations of 1, 2, 3, and 4 operations |
| `ex1_2/generate_ex2.py` | 250 | Generate image variations using 12 base operations. |
| `ex1_2/plot_ex1.py` | 266 | Plot charts from test_report.xlsx — ViT/CLIP + hash similarities for 793 combinations. |
| `ex1_2/plot_ex2.py` | 629 | Plot charts from all *_report.xlsx files in the table directory. |
| `ex1_2/plot_summary.py` | 375 | Generate summary charts from ex1_2 report Excel files. |
| `ex1_2/run_pipeline.py` | 102 | Run the full ex1_2 pipeline end to end. |
| `exp4/generate_datasets6_transforms.py` | 179 | Generate 32 image transforms for Datasets6 (reuses same operations as Datasets5). |
| `exp4/generate_datasets7_transforms.py` | 173 | Generate 32 image transforms for datasets7 (pixiv1000 anime images). |
| `exp4/logic_compare/run_comparison.py` | 170 | Compare 4 decision logic rules on all Datasets5 pairs. |
| `exp4/plot_datasets7_least_similar.py` | 152 | Render the 10 least similar original image pairs from Datasets7 in one image. |
| `exp4/plot_datasets7_top10_most_similar_negative_pairs.py` | 164 | Render the 10 most similar negative original-image pairs in Datasets7. |
| `exp4/plot_f1_vs_clip_top10_large.py` | 142 | Plot WikiArt F1 against CLIP for the ten strongest ViT thresholds. |
| `exp4/plot_f1_vs_vit_top10_large.py` | 141 | Plot WikiArt F1 against ViT for the ten strongest CLIP thresholds. |
| `exp4/replot_top10.py` | 136 | Re-render f1_vs_clip_top10.png from an existing threshold_sweep_summary.csv. |
| `exp4/run_datasets6_pipeline.py` | 649 | Complete pipeline for Datasets6 threshold sweep: |
| `exp4/run_datasets6_sweep.py` | 361 | Compute CLIP/ViT similarities for Datasets6 and run full threshold sweep (0-1, 0.01). |
| `exp4/run_datasets7_sweep.py` | 374 | Compute CLIP/ViT similarities for Datasets7 (pixiv1000) and run full threshold sweep (0-1, 0.01). |
| `exp4/run_exp4.py` | 248 | Evaluate the fixed CLIP+ViT rule on every Datasets5 pair. |
| `exp4/run_single_model_comparison_sep14.py` | 187 | Evaluate CLIP-only and ViT-only rules on Datasets5 and Datasets6. |
| `exp4/summaryAug25/.pptx_build_threshold/build_weighted_threshold_slide.mjs` | 240 | Presentation construction |
| `exp4/threshold_eval/full_sweep/plot_comparison.py` | 201 | Generate comparison charts for full threshold sweep. |
| `exp4/threshold_eval/full_sweep_0_1.py` | 209 | Full sweep: ViT 0.00–1.00 × CLIP 0.00–1.00, step 0.01 (10,201 combinations). |
| `exp4/threshold_eval/plot_heatmaps.py` | 85 | Generate two heatmaps: F1 score and False Positives for threshold sweep. |
| `exp4/threshold_eval/sweep_thresholds.py` | 250 | Sweep CLIP and ViT thresholds with AND logic, comparing all combinations. |
| `final/compare.py` | 183 | Image Similarity Comparator using CLIP and ViT |
| `gen_top10_vit_ds5.py` | 133 | Generate top-10 ViT negative pair visualization for Datasets5 — single-column layout. |
| `generate_datasets5_transforms.py` | 251 | Generate 32 image transforms for each of the 1000 images in Datasets5. |
| `generate_missing_transforms.py` | 37 | Generate transforms for only the missing images in Transform/datasets5. |
| `opendata/scripts/extract_opendata.py` | 350 | External NGA checkout; excluded from project source bundle. Open Data CSV extraction — SQL Server direct. |
| `opendata/tests/conftest.py` | 79 | External NGA checkout; excluded from project source bundle. Shared fixtures for the test suite. |
| `opendata/tests/test_csv_formatting.py` | 164 | External NGA checkout; excluded from project source bundle. Unit tests for CSV formatting functions in extract_opendata.py. |
| `opendata/tests/test_database.py` | 163 | External NGA checkout; excluded from project source bundle. Integration tests requiring a SQL Server connection. |
| `pixiv/download_pixiv_first_1000.py` | 334 | Download a prefix of images from a large remote ZIP using HTTP ranges. |
| `run_datasets5_threshold.py` | 397 | CLIP + ViT two-stage threshold evaluation on Datasets5 (ArtBench-10). |
| `run_threshold_eval_1000.py` | 700 | Threshold Evaluation: CLIP + ViT two-stage pipeline on 1000 images (Datasets2_1). |
| `run_threshold_eval_1000_v2.py` | 496 | Optimized Threshold Evaluation — pre-compute embeddings, then matrix-multiply. |
| `src/core/auto_perception_ex1.py` | 277 | get_image_files, get_hashers, precompute_hashes, compute_distances, get_vit_embedding |
| `src/core/auto_perception_ex2.py` | 322 | get_image_files, get_hashers, precompute_hashes, compute_distances, get_vit_embedding |
| `src/core/clip_all_fallback_comparison.py` | 459 | Compare CLIP baseline and CLIP+6 fallback strategies in one table and one chart. |
| `src/core/generate_ex1.py` | 185 | 12 base image operations — generate all combinations of 1, 2, 3, and 4 operations |
| `src/core/generate_ex2.py` | 250 | Generate image variations using 12 base operations. |
| `src/core/similarity.py` | 469 | CLIP, ViT & Perceptual Hash Image Similarity Comparison |
| `src/core/threshold.py` | 309 | Threshold Evaluation @ 0.9 — CLIP & ViT |
| `src/core/vit_all_fallback_4thresholds.py` | 332 | Use ViT as the baseline and compare six fallback strategies across four thresholds. |
| `src/pipelines/benchmark_pipeline.py` | 137 | Run the ex1_2 pipeline for multiple batch sizes with precise per-step timing. |
| `src/pipelines/resume_ex3.py` | 244 | Resume Ex3 from ViT after CLIP completed. Skips Phase 1 and CLIP. |
| `src/pipelines/run_7algo_similarity.py` | 382 | Run all 7 similarity algorithms (CLIP, ViT, 5 hashes) on each original→transform pair. |
| `src/pipelines/run_combined_200.py` | 402 | Combined 200-image threshold sweep: Archive(100) + Datasets2_1(100). |
| `src/pipelines/run_ex3.py` | 514 | Ex3 Pipeline: 12 transformations × 1000 images × 7 similarity algorithms. |
| `src/pipelines/run_pipeline.py` | 102 | Run the full ex1_2 pipeline end to end. |
| `src/pipelines/run_threshold_sweep.py` | 493 | Threshold Sweep: CLIP (positive) + ViT (negative) with 5×5 threshold grid. |
| `src/plotting/plot_abs_negative_pairs.py` | 57 | 重新绘制 vit_negative_pairs_line.jpg：对 ViT 负样本余弦相似度取绝对值后排序绘图。 |
| `src/plotting/plot_ex1.py` | 266 | Plot charts from test_report.xlsx — ViT/CLIP + hash similarities for 793 combinations. |
| `src/plotting/plot_ex2.py` | 629 | Plot charts from all *_report.xlsx files in the table directory. |
| `src/plotting/plot_summary.py` | 375 | Generate summary charts from ex1_2 report Excel files. |
| `src/utils/__init__.py` | 6 | 工具模块 |
| `src/utils/file_manager.py` | 159 | 文件管理工具模块 |

## Byte-identical source groups

Copies are retained to preserve historical paths and imports. Identical content does not imply an additional research contribution.

- `Aug27/legacy_code/run_datasets6_sweep.py`; `exp4/run_datasets6_sweep.py`
- `Aug27/legacy_code/run_datasets7_sweep.py`; `exp4/run_datasets7_sweep.py`
- `ex1_2/Similarity_compare/threshold_evaluation.py`; `src/core/threshold.py`
- `ex1_2/auto_perception_ex1.py`; `src/core/auto_perception_ex1.py`
- `ex1_2/auto_perception_ex2.py`; `src/core/auto_perception_ex2.py`
- `ex1_2/benchmark_pipeline.py`; `src/pipelines/benchmark_pipeline.py`
- `ex1_2/generate_ex1.py`; `src/core/generate_ex1.py`
- `ex1_2/generate_ex2.py`; `src/core/generate_ex2.py`
- `ex1_2/plot_ex1.py`; `src/plotting/plot_ex1.py`
- `ex1_2/plot_ex2.py`; `src/plotting/plot_ex2.py`
- `ex1_2/plot_summary.py`; `src/plotting/plot_summary.py`
- `ex1_2/run_pipeline.py`; `src/pipelines/run_pipeline.py`

## Known historical execution requirements

- Several `src/` files are copies moved from `ex1_2/`. In particular, `src/pipelines/run_pipeline.py` and `benchmark_pipeline.py` import sibling module names that now live elsewhere. The colocated `research/ex1_2/` versions retain that layout; neither version was advertised as a portable quick-start command.
- `run_ex3.py` / `resume_ex3.py` expect an old `datasets2` name. Some analyses refer to renamed or currently absent dataset/result directories. Restore the relevant inputs or adapt paths before running them.
- Historical generators and plotters often use fixed input names, create directories during import, or execute at module scope. Importing every research file is not a safe dependency check; this release uses syntax inspection and focused execution instead.
- `research/app.py` loads models and data at import time, uses the OpenAI CLIP package and a container-oriented model cache, and checks only the top three CLIP candidates. `/update` writes reference data. It is preserved as a service prototype; it was not started or deployed.
- `research/final/compare.py` defaults to CLIP 0.80 and the in21k ViT model. Its defaults differ from the main cache protocol.
- Historical `.mjs` slide builders need the environment-specific `@oai/artifact-tool` package and fonts. Formula images used by the weighting slide are retained. These builders are documentation sources, not requirements for numerical reproduction.
- The copied historical requirements file records an earlier environment. Root-level requirement groups are the release entry points; clean-environment installation of every optional legacy dependency was not performed.

## Changes limited to the new package

Personal absolute paths were made relative in five source files (the WikiArt downloader, method-comparison plotter, PPT generator, and two JavaScript slide builders). Their hashes and adjustments are recorded in `source_manifest.json`. Original workspace files were not edited. The browser demo image was replaced with a generated geometric fixture, and the portable extractor uses explicit public model IDs. Other new scripts organize existing computations for reuse.
