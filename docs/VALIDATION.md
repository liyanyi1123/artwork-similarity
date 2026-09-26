# Release verification

Verification was performed on 2026-09-24 with Python 3.11.15 and the existing local environment. Package versions are recorded in `validation_environment.json`. No new full-dataset model inference run was performed.

## Numerical checks

`python -m unittest discover -s tests -v` passed all **10 tests**:

- All **30,603** recorded grid rows across ArtBench-1K, WikiArt-1K and Pixiv-1K match TP, FP, TN and FN exactly; F1 matches the six-decimal recorded values.
- Shared-threshold counts match all three cached evaluations.
- ArtBench/WikiArt absolute-score single-model counts match the September 14 report.
- The original weighting result is reproduced: 374 and 243 high-performance points, selecting CLIP 0.87 and ViT 0.50.
- Strict `>` boundaries, negative scores, float32/float64 comparison behavior, identical rules for both classes, pair-array alignment, vector-row alignment and grid consistency are checked.

## Inference and transformation smoke tests

The existing local CLIP and ViT weights were loaded on CPU. The generated geometric image produced:

| Check | Result |
| --- | --- |
| CLIP vector shape | `(1, 512)` |
| ViT vector shape | `(1, 768)` |
| Vector normalization | Finite and approximately unit length |
| Self-comparison CLIP cosine | 1.0 |
| Self-comparison ViT cosine | 0.9999999404 |
| Joint decision | Similar |
| Generated variants | 32 JPEG files |
| Variants retained for positive pairs | 31, excluding F2 |

The model-loading report includes unused ViT classifier weights and an unused initialized pooler. The extractor uses the CLS token directly, as in the source implementation. These small inference checks used local weights; first-use Hugging Face downloading was not tested.

## Source and artifact checks

- All 86 packaged Python files passed syntax parsing.
- Both historical JavaScript slide sources and the prototype's inline JavaScript passed `node --check`.
- Original Python/JavaScript source hashes still match the initial inventory: no changes to those 85 files in the research workspace.
- Copied-file provenance hashes were checked. Personal absolute paths were removed from packaged text; no hardcoded credential assignment was found by the targeted scan.
- Ten existing figure files were opened in a contact sheet to verify readability and content. They preserve historical plotting choices; the Pixiv heatmap's zoom panel uses a high-F1 color range that obscures values below 0.97, so the complete numeric sweep is the authoritative result for that dataset.
- The synthetic demo image exists and is referenced by the copied prototype.
- Model weights, raw artwork datasets, `.git` histories, environment directories and bytecode caches are excluded from the archive.

Final archive integrity, relative documentation links and execution after extraction are recorded in `package_validation.json`.

## Scope of verification

The portable score workflow, one-image extraction/comparison and one-image transform generation were exercised. A fresh installation of every dependency group, complete reruns of historical experiments, legacy Flask startup, interactive browser tests and the environment-specific slide-rendering toolchain were not performed. The legacy code remains available with its original assumptions described in `SOURCE_GUIDE.md`.
