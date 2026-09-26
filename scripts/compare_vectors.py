#!/usr/bin/env python3
"""Compare two image-vector exports from extract_vectors.py and write pair scores."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate import predict, validate_threshold


def load_vectors(directory: Path):
    vectors, names, models = {}, {}, {}
    for kind in ("clip", "vit"):
        with np.load(directory / f"{kind}_vectors.npz", allow_pickle=False) as archive:
            value = archive["embeddings"].astype(np.float32)
            names[kind] = archive["image_paths"].astype(str)
            models[kind] = str(archive["model"].item())
        if value.ndim != 2 or not len(value) or value.shape[0] != len(names[kind]):
            raise ValueError(f"Invalid {kind} vector shape in {directory}")
        norms = np.linalg.norm(value, axis=1, keepdims=True)
        if not np.isfinite(value).all() or np.any(norms == 0):
            raise ValueError(f"Non-finite or zero {kind} vector in {directory}")
        if len(set(names[kind])) != len(names[kind]):
            raise ValueError(f"Duplicate image paths in {directory}")
        vectors[kind] = value / norms
    if not np.array_equal(names["clip"], names["vit"]):
        raise ValueError(f"CLIP/ViT image row order differs in {directory}")
    return vectors, names["clip"], models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", type=Path)
    parser.add_argument("references", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-threshold", type=float, default=0.87)
    parser.add_argument("--vit-threshold", type=float, default=0.50)
    parser.add_argument("--score-mode", choices=("signed", "absolute"), default="signed")
    args = parser.parse_args()
    try:
        validate_threshold(args.clip_threshold)
        validate_threshold(args.vit_threshold)
        q, qnames, qmodels = load_vectors(args.queries)
        r, rnames, rmodels = load_vectors(args.references)
        for kind in ("clip", "vit"):
            if qmodels[kind] != rmodels[kind] or q[kind].shape[1] != r[kind].shape[1]:
                raise ValueError(f"Query and reference {kind} models/dimensions differ")
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    matches = 0
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["query", "reference", "clip_similarity", "vit_similarity", "similar", "score_mode"])
        for qi, name in enumerate(qnames):
            clip = r["clip"] @ q["clip"][qi]
            vit = r["vit"] @ q["vit"][qi]
            if args.score_mode == "absolute":
                clip, vit = np.abs(clip), np.abs(vit)
            similar = predict(clip, vit, args.clip_threshold, args.vit_threshold)
            matches += int(similar.sum())
            for ri, reference in enumerate(rnames):
                writer.writerow([name, reference, float(clip[ri]), float(vit[ri]), int(similar[ri]), args.score_mode])
    print(f"Saved {len(qnames) * len(rnames):,} pairs ({matches:,} similar) to {args.output}")


if __name__ == "__main__":
    main()
