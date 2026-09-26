# Data and model inventory

Dataset names below follow the local project. Counts in an experiment are determined by its cached rows and selection code, not by recursively counting every image currently in the corresponding folder.

| Research label | Original workspace input | Recorded experiment size | Notes |
| --- | --- | ---: | --- |
| datasets1 | `Archive/` | 100 originals | Mixed early collection; the archive also contains derivative material |
| datasets2 | `Datasets2_2/` or early Open Images subset | 100 selected originals | The current folder contains 1,000 images; early experiments selected 10 per class |
| datasets3 | Archive + Open Images subset | 200 originals | Historical combined set, not a physical `Datasets3/` folder in the current workspace |
| datasets4 | `Datasets2_1/` | 1,000 originals | Ten classes × 100; some old analysis directory names are now absent |
| ArtBench-1K / datasets5 | Ten named style subdirectories of `Datasets5/` | 1,000 originals | An additional `datasets6_test/` directory has 10 test images and is outside the cache's selection |
| WikiArt-1K / datasets6 | Ten named style subdirectories of `Datasets6/` | 1,000 originals | The folder also has 1,000 top-level image files; do not count both layouts as 2,000 independent originals |
| Pixiv-1K / datasets7 | `datasets7/` | 1,000 originals | Current folder and Aug27 export contain 1,017; the sweep cache predates the additional 17 |

Counts were inspected in the current workspace during packaging on 2026-09-24. `Datasets2_1/` and `Datasets2_2/` both currently contain 1,000 image files. Their directory names and size alone do not establish disjointness. Similarly, Datasets5 and Datasets6 collection scripts both use ArtBench CSV URL metadata. No dataset-disjointness claim is made here.

## Included score archives

`results/scores/{artbench1k,wikiart1k,pixiv1k}.npz` contain four float32 arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `pos_clip` | `(31000,)` | CLIP scores for original–transformation pairs |
| `pos_vit` | `(31000,)` | ViT scores for the same positive pairs in the same order |
| `neg_clip` | `(499500,)` | CLIP scores for unordered distinct originals |
| `neg_vit` | `(499500,)` | ViT scores for the same negative pairs |

The archives contain numeric scores only, with no artwork images, absolute file paths or pickled objects. Negative order is `np.triu_indices(1000, k=1)`. Positive order matches the original cache. The packaging checks verified 31 retained transformations per original and exclusion of F2. Individual image identity and adjudication cannot be reconstructed from score-only archives; use the historical caches and source dataset if image-level inspection is needed.

Positive CLIP and both ViT arrays were copied from the existing caches. Negative CLIP scores were derived from cached normalized original CLIP vectors with the same float32 row-wise product/sum operation as the original sweep. Source file hashes and derivation are recorded in [score_provenance.json](score_provenance.json).

The original score tables under `results/reported/` preserve their original numerical values. They are evidence for the historical runs and regression targets, not results of a new full image-inference experiment.

## Obtaining images for historical runs

The relevant acquisition code remains under `research/`:

- Open Images: `download_openimages_100.py` and the attributed Google downloader. Existing source ID lists and the initial input layout are needed for exact subsets.
- ArtBench: `Datasets5/download_100_per_style.py` expects `Datasets5/ArtBench-10.csv` from the ArtBench distribution. Sampling, URL availability and image validation affect the resulting subset.
- WikiArt URLs: `Datasets6/download_wikiart_1000.py` uses that ArtBench CSV and generates source records.
- Pixiv: the three collection approaches retain their original remote-source assumptions. Remote availability may differ from the original collection date.

Raw images, source archives, full metadata distributions and all transformed images stay outside this package. Consult the original sources for their access terms and image rights. `opendata/` metadata availability does not grant rights to every image URL referenced by that dataset.

## Model weights

Weights are not included. The portable extractor accepts Hugging Face model IDs or local directories. The default IDs match the main local model configurations. `model_provenance.json` records local configuration/weight hashes used in the smoke test; the original cache does not record an upstream revision ID. For exact future inference, archive the chosen upstream revision and preprocessing configuration alongside the outputs.

`examples/synthetic.png` is a deterministic geometric fixture created for this release. It is used only to demonstrate execution and the static UI, and contributes no research benchmark result.
