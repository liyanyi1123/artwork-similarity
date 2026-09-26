#!/usr/bin/env python3
"""Evaluate CLIP-only and ViT-only rules on Datasets5 and Datasets6."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "analysis" / "ComparisonSep14"
CLIP_THRESHOLD = 0.87
VIT_THRESHOLD = 0.50


def metrics(pos_scores: np.ndarray, neg_scores: np.ndarray, threshold: float) -> dict[str, int | float]:
    positive_predictions = pos_scores > threshold
    negative_predictions = neg_scores > threshold
    tp = int(positive_predictions.sum())
    fn = int((~positive_predictions).sum())
    fp = int(negative_predictions.sum())
    tn = int((~negative_predictions).sum())
    total = tp + fn + fp + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "positive_pairs": tp + fn,
        "negative_pairs": tn + fp,
        "total_pairs": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "false_negative_rate": 1 - recall,
    }


def load_clip_scores(embedding_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(embedding_dir / "clip_embeddings.npz", allow_pickle=False)
    original = data["orig_clip"].astype(np.float32)
    positive = np.abs(data["pos_sims"].astype(np.float32))
    original = original / np.linalg.norm(original, axis=1, keepdims=True)
    first, second = np.triu_indices(len(original), k=1)
    negative = np.abs(np.sum(original[first] * original[second], axis=1))
    return positive, negative


def load_vit_scores(
    embedding_dir: Path,
    positive_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    original = np.load(embedding_dir / "vit_embeddings.npz")["orig_vit"].astype(np.float32)
    positive = np.abs(np.load(positive_path).astype(np.float32))
    original = original / np.linalg.norm(original, axis=1, keepdims=True)
    first, second = np.triu_indices(len(original), k=1)
    negative = np.abs(np.load(embedding_dir / "vit_neg_sims.npy").astype(np.float32))
    expected_negative = len(first)
    if len(negative) != expected_negative:
        raise ValueError(f"Expected {expected_negative} ViT negative pairs, found {len(negative)}")
    return positive, negative


def evaluate_dataset(
    name: str,
    clip_embedding_dir: Path,
    vit_positive_path: Path,
) -> dict[str, object]:
    clip_positive, clip_negative = load_clip_scores(clip_embedding_dir)
    vit_positive, vit_negative = load_vit_scores(clip_embedding_dir, vit_positive_path)
    if len(clip_positive) != len(vit_positive) or len(clip_negative) != len(vit_negative):
        raise ValueError(f"CLIP/ViT pair counts do not match for {name}")
    result = {
        "dataset": name,
        "similarity": "absolute cosine similarity",
        "clip_threshold": CLIP_THRESHOLD,
        "vit_threshold": VIT_THRESHOLD,
        "clip_only": metrics(clip_positive, clip_negative, CLIP_THRESHOLD),
        "vit_only": metrics(vit_positive, vit_negative, VIT_THRESHOLD),
    }
    print(
        f"{name}: positives={len(clip_positive):,}, negatives={len(clip_negative):,}, "
        f"CLIP F1={result['clip_only']['f1']:.6f}, ViT F1={result['vit_only']['f1']:.6f}"
    )
    return result


def merge_results(results: list[dict[str, object]]) -> dict[str, object]:
    merged: dict[str, object] = {
        "dataset": "Datasets5 + Datasets6",
        "similarity": "absolute cosine similarity",
        "clip_threshold": CLIP_THRESHOLD,
        "vit_threshold": VIT_THRESHOLD,
    }
    for rule in ("clip_only", "vit_only"):
        positive = sum(int(item[rule]["positive_pairs"]) for item in results)
        negative = sum(int(item[rule]["negative_pairs"]) for item in results)
        tp = sum(int(item[rule]["TP"]) for item in results)
        tn = sum(int(item[rule]["TN"]) for item in results)
        fp = sum(int(item[rule]["FP"]) for item in results)
        fn = sum(int(item[rule]["FN"]) for item in results)
        merged[rule] = metrics_from_counts(positive, negative, tp, tn, fp, fn)
    return merged


def metrics_from_counts(
    positive: int,
    negative: int,
    tp: int,
    tn: int,
    fp: int,
    fn: int,
) -> dict[str, int | float]:
    total = positive + negative
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "positive_pairs": positive,
        "negative_pairs": negative,
        "total_pairs": total,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "false_negative_rate": 1 - recall,
    }


def write_outputs(results: list[dict[str, object]], merged: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"protocol": {
        "clip_rule": f"absolute CLIP cosine similarity > {CLIP_THRESHOLD:.2f}",
        "vit_rule": f"absolute ViT cosine similarity > {VIT_THRESHOLD:.2f}",
        "positive_definition": "original image paired with one of its 31 retained transformations",
        "negative_definition": "unordered pair of two different original images",
    }, "datasets": results, "combined": merged}
    with (OUTPUT_DIR / "single_model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    fields = ["dataset", "rule", "positive_pairs", "negative_pairs", "total_pairs",
              "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "f1",
              "specificity", "false_positive_rate", "false_negative_rate"]
    with (OUTPUT_DIR / "single_model_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in [*results, merged]:
            for rule in ("clip_only", "vit_only"):
                row = {"dataset": item["dataset"], "rule": rule, **item[rule]}
                writer.writerow(row)


def main() -> None:
    datasets = [
        evaluate_dataset(
            "Datasets5",
            BASE_DIR / "analysis" / "threshold_eval_datasets5" / "embeddings",
            BASE_DIR / "exp4" / "results" / "positive_vit_similarities.npy",
        ),
        evaluate_dataset(
            "Datasets6",
            BASE_DIR / "exp4" / "datasets6_sweep" / "embeddings",
            BASE_DIR / "exp4" / "datasets6_sweep" / "embeddings" / "vit_pos_sims.npy",
        ),
    ]
    merged = merge_results(datasets)
    write_outputs(datasets, merged)
    print(f"Saved results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()