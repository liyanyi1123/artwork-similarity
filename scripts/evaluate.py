#!/usr/bin/env python3
"""Re-evaluate paired CLIP/ViT scores with a label-independent decision rule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("pos_clip", "pos_vit", "neg_clip", "neg_vit")
RULES = ("and", "or", "clip", "vit")


def load_scores(path: Path, mode: str = "signed") -> dict[str, np.ndarray]:
    if mode not in ("signed", "absolute"):
        raise ValueError("Score mode must be signed or absolute")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in KEYS}
    for key, values in arrays.items():
        if values.ndim != 1 or not len(values) or values.dtype.kind != "f":
            raise ValueError(f"{key} must be a nonempty one-dimensional floating array")
        if not np.isfinite(values).all() or (np.abs(values) > 1.00001).any():
            raise ValueError(f"{key} contains invalid cosine scores")
        if mode == "absolute":
            arrays[key] = np.abs(values)
    for prefix in ("pos", "neg"):
        if arrays[prefix + "_clip"].shape != arrays[prefix + "_vit"].shape:
            raise ValueError(f"CLIP/ViT {prefix} pair counts differ")
    return arrays


def validate_threshold(value: float) -> float:
    if not np.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("Cosine thresholds must be finite and in [-1, 1]")
    return value


def predict(clip, vit, clip_threshold: float, vit_threshold: float, rule: str = "and"):
    # Compare float32 stored scores against float64 thresholds, as in the original sweeps.
    c = clip.astype(np.float64) > validate_threshold(clip_threshold)
    v = vit.astype(np.float64) > validate_threshold(vit_threshold)
    if rule == "and":
        return c & v
    if rule == "or":
        return c | v
    if rule == "clip":
        return c
    if rule == "vit":
        return v
    raise ValueError(f"Unknown rule: {rule}")


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    tp, fp, tn, fn = map(int, (tp, fp, tn, fn))
    total = tp + fp + tn + fn
    return {
        "positive_pairs": tp + fn, "negative_pairs": tn + fp, "total_pairs": total,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def evaluate(scores: dict, clip_threshold: float, vit_threshold: float, rule: str) -> dict:
    positive = predict(scores["pos_clip"], scores["pos_vit"], clip_threshold, vit_threshold, rule)
    negative = predict(scores["neg_clip"], scores["neg_vit"], clip_threshold, vit_threshold, rule)
    tp, fp = int(positive.sum()), int(negative.sum())
    return metrics_from_counts(tp, fp, len(negative) - fp, len(positive) - tp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("--clip-threshold", type=float, default=0.87)
    parser.add_argument("--vit-threshold", type=float, default=0.50)
    parser.add_argument("--score-mode", choices=("signed", "absolute"), default="signed")
    parser.add_argument("--rule", choices=(*RULES, "all"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        scores = load_scores(args.scores, args.score_mode)
        rules = RULES if args.rule == "all" else (args.rule,)
        result = {"scores_file": args.scores.name, "score_mode": args.score_mode,
                  "comparison": ">", "threshold_precision": "float64; stored scores promoted without rounding",
                  "clip_threshold": args.clip_threshold, "vit_threshold": args.vit_threshold,
                  "rules": {r: evaluate(scores, args.clip_threshold, args.vit_threshold, r) for r in rules}}
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
