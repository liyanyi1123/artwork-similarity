# CLIP / ViT image vectors

This directory contains the complete image-vector exports for the current
`Datasets6` and `datasets7` directories.

## Files

- `Datasets6/clip_vectors.npz`: 1,000 normalized CLIP vectors, shape `(1000, 512)`
- `Datasets6/vit_vectors.npz`: 1,000 normalized ViT vectors, shape `(1000, 768)`
- `datasets7/clip_vectors.npz`: 1,017 normalized CLIP vectors, shape `(1017, 512)`
- `datasets7/vit_vectors.npz`: 1,017 normalized ViT vectors, shape `(1017, 768)`
- `extract_clip_vit_vectors.py`: reusable image-to-vector extraction script
- `legacy_code/`: copies of the original Datasets6/Datasets7 extraction and sweep scripts

The old `datasets7` cache contained 1,000 images. The following 17 images were
added after that cache was created and were extracted with the same local
models before the final 1,017-row files were assembled:

`74681538`, `75619971`, `76899919`, `78142163`, `78308479`, `78368737`,
`78429877`, `79730190`, `79754096`, `80405436`, `81678107`, `82727745`,
`83427982`, `84000253`, `85035406`, `85242286`, `86359999`.

## NPZ structure

Each vector file has the same keys:

- `embeddings`: `float32` matrix, one row per image
- `image_paths`: image paths relative to the source dataset directory
- `filenames`: image filenames in the same row order
- `model`: Hugging Face model identifier
- `normalized`: `True`; every row is L2-normalized

Example:

```python
import numpy as np

data = np.load("Aug27/datasets7/clip_vectors.npz")
vectors = data["embeddings"]
paths = data["image_paths"]

print(vectors.shape)  # (1017, 512)
print(paths[0], vectors[0])
```

## Regenerate vectors

The script defaults to the repository's local models:

- CLIP: `models/clip-vit-base-patch32`
- ViT: `models/vit-base-patch16-224`

Run from the repository root:

```bash
python Aug27/extract_clip_vit_vectors.py Datasets6 Aug27/Datasets6_new
python Aug27/extract_clip_vit_vectors.py datasets7 Aug27/datasets7_new
```

To extract one image directly:

```bash
python Aug27/extract_clip_vit_vectors.py \
  datasets7/72415148.jpg \
  Aug27/single_image_vectors
```

The resulting CLIP and ViT embedding shapes are `(1, 512)` and `(1, 768)`.

Use `--recursive` when images are stored in subdirectories. Run
`python Aug27/extract_clip_vit_vectors.py --help` for all
options. The repository's `python` environment already
contains the packages listed in `requirements.txt`.
