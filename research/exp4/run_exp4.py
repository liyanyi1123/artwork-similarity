#!/usr/bin/env python3
"""Evaluate the fixed CLIP+ViT rule on every Datasets5 pair.

Rule: predict similar only when CLIP cosine similarity > 0.8 AND
ViT cosine similarity > 0.5.  Existing Datasets5 embeddings are reused;
only the missing ViT embeddings for the 31,000 transformed positive images
are computed here.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUT_DIR / "results"
EMBEDDING_DIR = BASE_DIR / "analysis" / "threshold_eval_datasets5" / "embeddings"
CLIP_CACHE = EMBEDDING_DIR / "clip_embeddings.npz"
VIT_CACHE = EMBEDDING_DIR / "vit_embeddings.npz"
VIT_MODEL_DIR = BASE_DIR / "models" / "vit-base-patch16-224"

CLIP_THRESHOLD = 0.8
VIT_THRESHOLD = 0.5
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
CPU_THREADS = min(4, os.cpu_count() or 1)
STYLES = [
    "art_nouveau", "baroque", "expressionism", "impressionism",
    "post_impressionism", "realism", "renaissance", "romanticism",
    "surrealism", "ukiyo_e",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def gather_original_paths() -> list[Path]:
    """Return originals in the exact order used by the existing Datasets5 run."""
    paths: list[Path] = []
    for style in STYLES:
        style_dir = BASE_DIR / "Datasets5" / style
        paths.extend(
            path for path in sorted(style_dir.iterdir())
            if path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return paths


@torch.no_grad()
def calculate_positive_vit(transform_paths: np.ndarray, original_indices: np.ndarray,
                           original_vit: np.ndarray) -> np.ndarray:
    """Compute ViT cosine similarity for each original--transformation pair."""
    if not VIT_MODEL_DIR.exists():
        raise FileNotFoundError(f"Local ViT model not found: {VIT_MODEL_DIR}")

    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_DIR, local_files_only=True)
    model = ViTModel.from_pretrained(VIT_MODEL_DIR, local_files_only=True).to(DEVICE).eval()
    sims = np.empty(len(transform_paths), dtype=np.float32)
    original_vit = original_vit / np.linalg.norm(original_vit, axis=1, keepdims=True)

    for start in tqdm(range(0, len(transform_paths), BATCH_SIZE), desc="ViT positive pairs"):
        end = min(start + BATCH_SIZE, len(transform_paths))
        images = []
        for image_path in transform_paths[start:end]:
            with Image.open(str(image_path)) as image:
                images.append(image.convert("RGB"))
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        features = model(**inputs).last_hidden_state[:, 0, :]
        features = features / features.norm(dim=-1, keepdim=True)
        transformed_vit = features.cpu().numpy()
        sims[start:end] = np.sum(
            original_vit[original_indices[start:end]] * transformed_vit, axis=1
        )
    return sims


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float | int]:
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "total_pairs": total,
        "positive_pairs": tp + fn,
        "negative_pairs": tn + fp,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "false_positive_rate": fp / (fp + tn) if fp + tn else 0.0,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
    }


def write_positive_rows(path: Path, transform_paths: np.ndarray, original_indices: np.ndarray,
                        originals: list[Path], clip_sims: np.ndarray, vit_sims: np.ndarray) -> tuple[int, int]:
    predicted = (clip_sims > CLIP_THRESHOLD) & (vit_sims > VIT_THRESHOLD)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["original_path", "transform_path", "label", "clip_similarity", "vit_similarity", "predicted_similar", "outcome"])
        for idx, transform_path in enumerate(transform_paths):
            is_similar = bool(predicted[idx])
            writer.writerow([
                originals[int(original_indices[idx])], transform_path, 1,
                f"{clip_sims[idx]:.6f}", f"{vit_sims[idx]:.6f}",
                int(is_similar), "TP" if is_similar else "FN",
            ])
    return int(predicted.sum()), int((~predicted).sum())


def write_negative_rows(path: Path, originals: list[Path], clip_embeddings: np.ndarray,
                        vit_sims: np.ndarray) -> tuple[int, int]:
    triangular_i, triangular_j = np.triu_indices(len(originals), k=1)
    if len(vit_sims) != len(triangular_i):
        raise ValueError("ViT negative similarity count does not match all original-image pairs.")
    clip_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
    clip_sims = np.sum(clip_embeddings[triangular_i] * clip_embeddings[triangular_j], axis=1)
    predicted = (clip_sims > CLIP_THRESHOLD) & (vit_sims > VIT_THRESHOLD)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_a_path", "image_b_path", "label", "clip_similarity", "vit_similarity", "predicted_similar", "outcome"])
        for idx, (left, right) in enumerate(zip(triangular_i, triangular_j)):
            is_similar = bool(predicted[idx])
            writer.writerow([
                originals[int(left)], originals[int(right)], 0,
                f"{clip_sims[idx]:.6f}", f"{vit_sims[idx]:.6f}",
                int(is_similar), "FP" if is_similar else "TN",
            ])
    return int(predicted.sum()), int((~predicted).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, help="Start index for one resumable ViT-positive chunk.")
    parser.add_argument("--end", type=int, help="End index (exclusive) for one resumable ViT-positive chunk.")
    parser.add_argument("--finalize", action="store_true", help="Combine saved chunks and write all final results.")
    args = parser.parse_args()
    started = time.time()
    if DEVICE == "cpu":
        torch.set_num_threads(CPU_THREADS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}; ViT batch size: {BATCH_SIZE}; CPU threads: {torch.get_num_threads()}")
    print(f"Rule: CLIP > {CLIP_THRESHOLD} AND ViT > {VIT_THRESHOLD}")

    clip_data = np.load(CLIP_CACHE, allow_pickle=False)
    vit_data = np.load(VIT_CACHE, allow_pickle=False)
    clip_positive = clip_data["pos_sims"].astype(np.float32)
    transform_paths = clip_data["transform_paths"]
    positive_original_indices = clip_data["pos_orig_idx"].astype(np.int32)
    clip_original = clip_data["orig_clip"].astype(np.float32)
    vit_original = vit_data["orig_vit"].astype(np.float32)
    vit_negative = np.load(EMBEDDING_DIR / "vit_neg_sims.npy").astype(np.float32)
    originals = gather_original_paths()

    if not (len(originals) == len(clip_original) == len(vit_original)):
        raise ValueError("Original image paths and cached embeddings have inconsistent counts.")
    if not (len(clip_positive) == len(transform_paths) == len(positive_original_indices)):
        raise ValueError("Cached positive CLIP data has inconsistent counts.")

    chunks_dir = RESULTS_DIR / "positive_vit_chunks"
    chunks_dir.mkdir(exist_ok=True)
    if args.start is not None or args.end is not None:
        if args.start is None or args.end is None or not (0 <= args.start < args.end <= len(transform_paths)):
            raise ValueError("--start and --end must define a valid non-empty range.")
        output = chunks_dir / f"{args.start:05d}_{args.end:05d}.npy"
        if output.exists():
            print(f"Chunk already exists: {output}")
            return
        sims = calculate_positive_vit(
            transform_paths[args.start:args.end], positive_original_indices[args.start:args.end], vit_original
        )
        np.save(output, sims)
        print(f"Saved {output} ({len(sims)} similarities) in {time.time() - started:.2f}s")
        return
    if not args.finalize:
        raise ValueError("Run with --start/--end for chunks, then --finalize after all chunks complete.")

    chunk_files = list(chunks_dir.glob("*.npy"))
    intervals: dict[int, list[tuple[int, Path]]] = {}
    for chunk_file in chunk_files:
        start_text, end_text = chunk_file.stem.split("_")
        start, end = int(start_text), int(end_text)
        intervals.setdefault(start, []).append((end, chunk_file))
    expected_start = 0
    chunks = []
    while expected_start < len(transform_paths):
        candidates = intervals.get(expected_start, [])
        if not candidates:
            raise ValueError(f"Missing or non-contiguous chunk at index {expected_start}.")
        # Earlier interrupted test runs may leave overlapping chunks. Any complete
        # chunk uses the same model/settings; use the longest one at this position.
        start = expected_start
        end, chunk_file = max(candidates, key=lambda item: item[0])
        values = np.load(chunk_file)
        if len(values) != end - start:
            raise ValueError(f"Unexpected length in {chunk_file}.")
        chunks.append(values)
        expected_start = end
    if expected_start != len(transform_paths):
        raise ValueError(f"Chunks only cover 0:{expected_start}; expected 0:{len(transform_paths)}.")
    positive_vit = np.concatenate(chunks).astype(np.float32)
    np.save(RESULTS_DIR / "positive_vit_similarities.npy", positive_vit)

    tp, fn = write_positive_rows(
        RESULTS_DIR / "positive_pairs_results.csv", transform_paths,
        positive_original_indices, originals, clip_positive, positive_vit,
    )
    fp, tn = write_negative_rows(
        RESULTS_DIR / "negative_pairs_results.csv", originals, clip_original, vit_negative,
    )
    summary = {
        "dataset": "Datasets5 (ArtBench-10)",
        "decision_rule": "CLIP cosine similarity > 0.8 AND ViT cosine similarity > 0.5",
        "clip_threshold": CLIP_THRESHOLD,
        "vit_threshold": VIT_THRESHOLD,
        "comparison_operator": "strictly greater than",
        "device_for_missing_vit_positive_embeddings": DEVICE,
        "vit_batch_size": BATCH_SIZE,
        "metrics": metrics(tp, fp, tn, fn),
        "elapsed_seconds": round(time.time() - started, 2),
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with (RESULTS_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary["metrics"].keys())
        writer.writeheader()
        writer.writerow(summary["metrics"])

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
