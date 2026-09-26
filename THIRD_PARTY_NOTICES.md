# Third-party sources and project licensing

No new license is assigned by this packaging step to the project author's code, research results or documentation. The project owner can add an appropriate license at any time. Absence of a license should not be interpreted as a grant of unrestricted reuse rights.

## Google Open Images downloader

`research/downloader.py` retains the original notice:

> Copyright 2020 The Google Research Authors.
> Licensed under the Apache License, Version 2.0.

An Apache-2.0 license copy is provided at `licenses/Apache-2.0.txt`. This downloader is a third-party utility used in the workflow, not an original algorithm contribution of this project.

## National Gallery of Art Open Data

The original workspace contains a separate checkout of [NationalGalleryOfArt/opendata](https://github.com/NationalGalleryOfArt/opendata). Its SQL Server extractor and tests were counted in the source inventory but excluded from the project source bundle. The upstream dataset's CC0 statement applies to its collection metadata as stated upstream; its README explicitly excludes image/media files from that data program. Refer to the upstream repository for its code, documentation and current terms.

## Pretrained models and packages

The experiments use [OpenAI CLIP](https://github.com/openai/CLIP), Hugging Face Transformers and Google ViT model artifacts, plus the Python packages named in the requirement files. Model weights and dependency code are not redistributed in this package. Their respective licenses and model cards apply.

## Images and documents

Raw ArtBench/WikiArt/Pixiv/Open Images collections, image-pair panels, manuscript drafts, downloaded papers and presentation binaries are excluded. Included plots are selected from the local experiment outputs and retain source mapping in `docs/source_manifest.json`. `examples/synthetic.png` is a new procedural geometric fixture used for execution checks and the static prototype; it is not a sampled artwork.
