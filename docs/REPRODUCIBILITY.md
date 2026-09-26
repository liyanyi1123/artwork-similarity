# Reproduction guide

Run commands from the root of `artwork-similarity-github/`. Use Python 3.11+; the verified local interpreter was Python 3.11.15. Dependencies are grouped by task rather than copied from the old environment freeze.

## 1. Recompute recorded metrics

```bash
python -m pip install -r requirements.txt
python scripts/evaluate.py results/scores/artbench1k.npz --output outputs/artbench_metrics.json
python scripts/evaluate.py results/scores/wikiart1k.npz --output outputs/wikiart_metrics.json
python scripts/evaluate.py results/scores/pixiv1k.npz \
  --clip-threshold 0.96 --vit-threshold 0.76 --output outputs/pixiv_metrics.json
```

The default rule setting prints four comparisons: AND, OR, CLIP-only and ViT-only. Use `--rule and` for just the joint result. Use `--score-mode absolute` to select the later absolute-score convention; the default is signed. Every result records its thresholds, comparison operator and score mode.

## 2. Recompute threshold grids

```bash
python scripts/sweep.py results/scores/artbench1k.npz --output outputs/artbench_sweep.csv
python scripts/sweep.py results/scores/wikiart1k.npz --output outputs/wikiart_sweep.csv
python scripts/sweep.py results/scores/pixiv1k.npz --output outputs/pixiv_sweep.csv
```

Each grid has 10,201 combinations. The implementation groups pairs by the number of thresholds passed and uses cumulative counts. It needs neither models nor raw images. The regression suite checks every TP/FP/TN/FN count against the original tables.

## 3. Recompute the shared threshold

```bash
python scripts/select_threshold.py \
  results/reported/exp4/threshold_eval/full_sweep/threshold_sweep_summary.csv \
  results/reported/exp4/datasets6_sweep/threshold_sweep_summary.csv \
  --output outputs/weighted_selection.json
```

This matches the existing slide calculation: counts `[374, 243]`, weights approximately `[0.606159, 0.393841]`, selected thresholds `(0.87, 0.50)`, weighted F1 approximately `0.993761`. The grids must contain identical threshold pairs. Replacing the paths with newly generated grids is supported; full-precision F1 values may change the last displayed digits relative to the six-decimal original tables.

## 4. Generate transformations

```bash
python -m pip install -r requirements-inference.txt
python scripts/generate_transforms.py examples/synthetic.png outputs/transforms
```

This reuses `research/generate_datasets5_transforms.py` operations. It creates 32 JPEG variants and a manifest with the source, operation, intensity and whether the variant belongs to the 31-positive protocol. For a folder input it searches recursively, uses collision-resistant output folder names, and requires output outside the source tree. Choose a fresh output directory for each run.

## 5. Extract and compare image vectors

```bash
python scripts/extract_vectors.py examples outputs/reference_vectors --recursive
python scripts/extract_vectors.py examples/synthetic.png outputs/query_vectors
python scripts/compare_vectors.py outputs/query_vectors outputs/reference_vectors \
  --output outputs/comparison.csv
```

First-use model downloading requires network access. `--device auto` chooses CUDA if available and CPU otherwise; `--device mps` is an explicit option. `--batch-size` defaults to 32.

With local models:

```bash
python scripts/extract_vectors.py examples outputs/reference_vectors \
  --clip-model models/clip-vit-base-patch32 \
  --vit-model models/vit-base-patch16-224 --device cpu
```

Each vector export contains `embeddings`, `image_paths`, `filenames`, `model`, and `normalized`. Query and reference exports must use the same model identifier/path and aligned image rows for CLIP and ViT. `compare_vectors.py` writes all query–reference pairs, one query at a time; runtime and output grow with the Cartesian product. It does not apply the legacy API's top-3 truncation.

## 6. Browse the application prototype

Open `prototype/index.html` in a browser. The page is self-contained apart from the bundled synthetic image. The demo button selects the simulated failure case; uploading an image selects simulated success. There is no server request or persistent registration.

`research/app.py` and `research/final/compare.py` are historical application sources. They need their own model/data setup and the upstream OpenAI CLIP package. Their behavior and defaults differ from the tested entry points; see [SOURCE_GUIDE.md](SOURCE_GUIDE.md). No Flask server was started during packaging.

## 7. Run validation

```bash
python -m unittest discover -s tests -v
```

This checks all three historical grids, shared-threshold counts, later single-model absolute-score metrics, weighting, strict boundaries and input alignment. The tests use NumPy and the standard library; installing pytest is unnecessary.

The local packaging smoke tests also ran the actual CLIP/ViT extraction on the synthetic image, checked unit-length `(1, 512)` and `(1, 768)` vectors, compared the image with itself, and generated 32 transforms. Those small checks establish that the entry points execute in the available local environment. The full research image collection, optional historical dependencies and UI behavior were not exhaustively re-executed. See [VALIDATION.md](VALIDATION.md) for final evidence.
