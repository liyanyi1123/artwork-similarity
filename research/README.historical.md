# Image Similarity Comparison — Two-Stage CLIP + ViT Pipeline

Evaluating a two-stage image similarity verification pipeline: CLIP as a high-recall pre-filter, ViT as a false-positive suppressor. Applied across 12 image transform operations at multiple intensity levels.

## Project Structure

```
.
├── src/                              # Source code
│   ├── core/                         # Core modules
│   │   ├── generate_ex1.py           # Image generation: all operation combos (1–4 ops)
│   │   ├── generate_ex2.py           # Image generation: per-op intensity levels (low/mid/high)
│   │   ├── auto_perception_ex1.py    # Similarity evaluation — Experiment 1
│   │   ├── auto_perception_ex2.py    # Similarity evaluation — Experiment 2
│   │   ├── similarity.py             # Similarity metric computation (hash, CLIP, ViT)
│   │   ├── threshold.py              # Threshold-based classification & metrics
│   │   ├── clip_all_fallback_comparison.py  # CLIP on all images, fallback comparison
│   │   └── vit_all_fallback_4thresholds.py  # ViT on all images, 4-threshold sweep
│   ├── plotting/                     # Visualization
│   │   ├── plot_ex1.py / plot_ex2.py # Per-experiment plots
│   │   └── plot_summary.py           # Summary dashboards
│   ├── pipelines/                    # End-to-end pipeline scripts
│   │   ├── run_pipeline.py           # Full pipeline: generate → evaluate → plot
│   │   ├── run_threshold_sweep.py    # CLIP × ViT threshold grid sweep (Datasets2_1, 100 imgs)
│   │   ├── run_combined_200.py       # Combined 200-image evaluation
│   │   ├── run_ex3.py                # Experiment 3: CLIP+ViT fallback pipeline
│   │   ├── run_7algo_similarity.py   # 7-algorithm similarity benchmark
│   │   ├── resume_ex3.py             # Resume interrupted ex3 run
│   │   └── benchmark_pipeline.py     # Performance benchmarking
│   └── utils/                        # Utilities
│       ├── __init__.py
│       └── file_manager.py           # File I/O helpers
│
├── ex1_2/                            # Legacy Experiment 1 & 2 workspace
│   ├── auto_perception_ex1.py / ex2.py
│   ├── generate_ex1.py / ex2.py
│   ├── plot_ex1.py / ex2.py / plot_summary.py
│   ├── run_pipeline.py / benchmark_pipeline.py
│   ├── Similarity_compare/           # Similarity comparison outputs
│   ├── Plot4AllImages/               # Batch image plots
│   └── OperatedImage/                # Generated transform images
│
├── Datasets2_1/                      # Dataset: 1,000 images (10 classes × 100)
├── Datasets2_2/                      # Dataset: 1,000 images (10 classes × 100)
├── Datasets3/                        # Dataset: 100 images
│
├── experiments/                      # Experiment outputs
│   ├── ex1/                          # Experiment 1 results
│   │   ├── tables/  plots/  results/
│   ├── ex2/                          # Experiment 2 results
│   │   ├── tables/  plots/  results/
│   ├── ex3/                          # Experiment 3: CLIP+ViT fallback
│   │   ├── tables/  results/
│   ├── threshold_sweep/              # Threshold grid sweep results (Datasets2_1, 100 images)
│   │   ├── threshold_grid.csv        # 25 CLIP × ViT combinations
│   │   ├── timing.json               # Runtime breakdown
│   │   └── transforms/               # Generated transform images (32×100)
│   └── time/                         # Benchmark timing reports
│
├── analysis/                         # Analysis results by dataset / config
│   ├── threshold_eval_clip08-vit05_datasets2_1/  # Full 1000-image Datasets2_1 run
│   │   ├── overall_summary_excel_chart_datasets2_1.xlsx  # 6-sheet Excel + 4 charts
│   │   ├── threshold_grid.csv                         # 126 CLIP×ViT combinations
│   │   ├── timing.json                                # Runtime summary
│   │   ├── vit_negative_pairs_line.jpg                # ViT sim ranked chart (all 499,500 pairs)
│   │   ├── clip_positive_pairs_line.jpg               # CLIP sim ranked chart (all 31,000 pairs)
│   │   └── embeddings/                                # Cached CLIP & ViT embeddings (.npz)
│   ├── threshold_eval_clip08_vit05_datasets2_2/       # Datasets2_2 run (100 images)
│   ├── threshold_eval_clip08_vit05_archieve/          # Archive baseline (8,050-image dataset)
│   ├── threshold_eval_clip08_05_200images/            # 200-image CLIP=0.80+ViT=0.50 run
│   ├── threshold_evaluation/                          # Threshold analysis spreadsheets
│   ├── similarity_matrices/                           # 7-algorithm similarity matrices
│   ├── similarity_7algo/                              # 7-algorithm comparison reports
│   └── summary_plots/                                 # Cross-experiment summary charts
│
├── models/                           # Pretrained models (local cache)
│   ├── clip-vit-base-patch32/        # OpenAI CLIP ViT-B/32
│   └── vit-base-patch16-224/         # Google ViT-B/16 @ 224px
│
├── PPT/                              # Presentation files
├── docs/                             # Documentation
├── Archive/  Archive2.zip            # Archived data
│
├── run_threshold_eval_1000.py        # Top-level: threshold sweep on full Datasets2_1
├── run_threshold_eval_1000_v2.py     # Optimized version (pre-computed embeddings)
├── downloader.py                     # OpenImages download utility
├── download_openimages_100.py        # Download 100-image subsets from OpenImages
├── requirements.txt                  # Python dependencies
└── README.md
```

## Pipeline Overview

```
Original Image (1000×)
      │
      ├─→ 12 transform operations × intensity levels → 32 variants each
      │
      ├─→ Positive pairs (original vs transform):  31,000 pairs
      │         ↓ CLIP cosine similarity
      │         ↓ CLIP ≥ threshold → TP  (same image)
      │         ↓ CLIP < threshold → FN  (missed)
      │
      └─→ Negative pairs (different originals):  C(1000,2) = 499,500 pairs
                ↓ ViT cosine similarity
                ↓ ViT < threshold → TN  (correctly rejected)
                ↓ ViT ≥ threshold → FP  (false positive)
```

**Key finding**: Best performance at **CLIP = 0.80, ViT = 0.50** — lenient CLIP preserves near-perfect recall (~0.999), aggressive ViT suppresses false positives.

## 12 Image Transform Operations

| # | Code | Operation | Parameters |
|---|------|-----------|------------|
| 1 | F1 | Horizontal flip | — |
| 2 | F2 | Vertical flip | — |
| 3 | C | Center crop + resize | low / mid / high |
| 4 | S1 | Horizontal shift | low / mid / high |
| 5 | S2 | Vertical shift | low / mid / high |
| 6 | R | Rotation (±2° base) | low / mid / high |
| 7 | B | Brightness / contrast | low / mid / high |
| 8 | V | Vignette | low / mid / high |
| 9 | T | Color temperature shift | low / mid / high |
| 10 | Sh | Sharpen | low / mid / high |
| 11 | K | Perspective distortion | low / mid / high |
| 12 | N | Film grain / noise | low / mid / high |

## Experiments

| Experiment | Description | Key Script |
|---|---|---|
| **Ex1** | All 1–4 operation combinations (793 images) | `src/core/generate_ex1.py` |
| **Ex2** | Per-operation intensity sweep (low/mid/high) | `src/core/generate_ex2.py` |
| **Ex3** | CLIP + ViT two-stage fallback pipeline | `src/pipelines/run_ex3.py` |
| **Threshold Sweep** | CLIP × ViT grid search (25 or 126 combos) | `run_threshold_eval_1000_v2.py` |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full threshold sweep on Datasets2_1 (1000 images)
python run_threshold_eval_1000_v2.py

# Run smaller sweep on 100 images
python src/pipelines/run_threshold_sweep.py

# Run CLIP+ViT fallback pipeline (Experiment 3)
python src/pipelines/run_ex3.py
```

## Key Results (Datasets2_1 — 1,000 images, 530,500 pairs)

| Metric | Value | Thresholds |
|---|---|---|
| Best F1 | **0.9756** | CLIP=0.80, ViT=0.50 |
| Accuracy | 0.9971 | CLIP=0.80, ViT=0.50 |
| Recall | 0.9990 | CLIP=0.80 (30 FN / 31,000) |
| Precision | 0.9533 | ViT=0.50 (1,532 FP / 499,500) |

**Conclusion**: Lower CLIP threshold + higher ViT threshold yields best accuracy. CLIP's role is to catch nearly all positives with minimal FN (0.1%); ViT's role is to filter false positives from the large negative-pair space.
