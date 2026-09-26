#!/usr/bin/env python3
"""Export the paper's FP/FN image pairs at CLIP > 0.87 AND ViT > 0.50."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


BASE = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent / "FNnFP"
CLIP_THRESHOLD = 0.87
VIT_THRESHOLD = 0.50
STYLES = (
    "art_nouveau", "baroque", "expressionism", "impressionism",
    "post_impressionism", "realism", "renaissance", "romanticism",
    "surrealism", "ukiyo_e",
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
CONFIGS = {
    "Datasets5": {
        "cache": "analysis/threshold_eval_datasets5/embeddings",
        "positive_vit": "exp4/results/positive_vit_similarities.npy",
        "sweep": "exp4/threshold_eval/full_sweep/threshold_sweep_summary.csv",
        "expected": {"FP": 121, "FN": 221},
    },
    "Datasets6": {
        "cache": "exp4/datasets6_sweep/embeddings",
        "positive_vit": "exp4/datasets6_sweep/embeddings/vit_pos_sims.npy",
        "sweep": "exp4/datasets6_sweep/threshold_sweep_summary.csv",
        "expected": {"FP": 225, "FN": 230},
    },
}


def current_transform_path(recorded_path: str) -> Path:
    # The repository moved into ArtworkDetection after the caches were created.
    parts = Path(recorded_path).parts
    return BASE.joinpath(*parts[parts.index("Transform"):])


def collect_pairs(dataset: str, config: dict) -> list[dict]:
    cache = BASE / config["cache"]
    with np.load(cache / "clip_embeddings.npz", allow_pickle=False) as data:
        orig_clip = data["orig_clip"].astype(np.float32)
        pos_clip = data["pos_sims"].astype(np.float32)
        pos_indices = data["pos_orig_idx"]
        transform_paths = data["transform_paths"]
    pos_vit = np.load(BASE / config["positive_vit"]).astype(np.float32)
    neg_vit = np.load(cache / "vit_neg_sims.npy").astype(np.float32)
    originals = [
        path
        for style in STYLES
        for path in sorted((BASE / dataset / style).iterdir())
        if path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    assert len(originals) == len(orig_clip) == 1000, dataset
    assert len(pos_clip) == len(pos_vit) == len(pos_indices) == len(transform_paths) == 31000
    assert Counter(map(int, pos_indices)) == Counter({i: 31 for i in range(1000)})
    for index, recorded_path in zip(pos_indices, transform_paths):
        original = originals[int(index)]
        assert Path(str(recorded_path)).parent.name == f"{original.parent.name}_{original.stem}"

    orig_clip /= np.linalg.norm(orig_clip, axis=1, keepdims=True)
    tri_i, tri_j = np.triu_indices(len(originals), k=1)
    assert len(neg_vit) == len(tri_i) == 499500
    neg_clip = np.empty(len(tri_i), dtype=np.float32)
    # Use the original sweep's float32 row-wise sum, with bounded memory use.
    for start in range(0, len(tri_i), 4096):
        stop = start + 4096
        neg_clip[start:stop] = np.sum(
            orig_clip[tri_i[start:stop]] * orig_clip[tri_j[start:stop]], axis=1
        )

    fp_indices = np.flatnonzero((neg_clip > CLIP_THRESHOLD) & (neg_vit > VIT_THRESHOLD))
    fn_indices = np.flatnonzero(~((pos_clip > CLIP_THRESHOLD) & (pos_vit > VIT_THRESHOLD)))
    counts = {"FP": len(fp_indices), "FN": len(fn_indices)}
    assert counts == config["expected"], (dataset, counts)
    with (BASE / config["sweep"]).open(newline="", encoding="utf-8") as stream:
        sweep = [
            row for row in csv.DictReader(stream)
            if float(row["clip_threshold"]) == CLIP_THRESHOLD
            and float(row["vit_threshold"]) == VIT_THRESHOLD
        ]
    assert len(sweep) == 1
    assert counts == {key: int(sweep[0][key]) for key in counts}

    pairs = []
    for outcome, indices in (("FP", fp_indices), ("FN", fn_indices)):
        for index in indices:
            index = int(index)
            is_fp = outcome == "FP"
            source_a = originals[int(tri_i[index] if is_fp else pos_indices[index])]
            source_b = (
                originals[int(tri_j[index])] if is_fp
                else current_transform_path(str(transform_paths[index]))
            )
            clip = float(neg_clip[index] if is_fp else pos_clip[index])
            vit = float(neg_vit[index] if is_fp else pos_vit[index])
            pairs.append({
                "dataset": dataset,
                "outcome": outcome,
                "cached_pair_index": index,
                "source_a": source_a,
                "source_b": source_b,
                "image_b_role": "original" if is_fp else "transform",
                "clip_similarity": clip,
                "vit_similarity": vit,
                "clip_pass": clip > CLIP_THRESHOLD,
                "vit_pass": vit > VIT_THRESHOLD,
            })
    print(f"{dataset}: FP={counts['FP']}, FN={counts['FN']}", flush=True)
    return pairs


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(dry_run: bool) -> None:
    pairs = [pair for dataset, config in CONFIGS.items() for pair in collect_pairs(dataset, config)]
    assert Counter(pair["outcome"] for pair in pairs) == {"FP": 346, "FN": 451}
    sequence = Counter()
    total_bytes = 0
    for pair in pairs:
        outcome = pair["outcome"]
        sequence[outcome] += 1
        pair["pair_id"] = f"{outcome}_{sequence[outcome]:04d}_{pair['dataset']}"
        for side in ("a", "b"):
            source = pair[f"source_{side}"]
            if not source.is_file():
                raise FileNotFoundError(source)
            role = "original" if side == "a" else pair["image_b_role"]
            filename = f"{side.upper()}_{role}__{source.name}"
            assert len(filename.encode("utf-8")) <= 255, filename
            relative = Path(outcome) / pair["pair_id"] / filename
            pair[f"saved_{side}"] = relative.as_posix()
            total_bytes += source.stat().st_size
            destination = OUTPUT / relative
            if destination.exists() and sha256(destination) != sha256(source):
                raise FileExistsError(f"Existing file differs from its source: {destination}")
    print(f"Preflight: {len(pairs)} pairs, {2 * len(pairs)} images, {total_bytes / 1024**2:.1f} MiB", flush=True)
    if dry_run:
        return

    manifests = {"FP": [], "FN": []}
    source_hashes = {}
    def save_pair(pair: dict) -> dict:
        row = dict(pair)
        for side in ("a", "b"):
            source = pair[f"source_{side}"]
            destination = OUTPUT / pair[f"saved_{side}"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            resolved = source.resolve()
            for attempt in range(3):
                try:
                    if resolved not in source_hashes:
                        source_hashes[resolved] = sha256(resolved)
                    if not destination.exists():
                        temporary = destination.with_name(destination.name + ".part")
                        with resolved.open("rb") as src, temporary.open("wb") as dst:
                            shutil.copyfileobj(src, dst, length=1024 * 1024)
                        assert sha256(temporary) == source_hashes[resolved], temporary
                        temporary.replace(destination)
                    break
                except TimeoutError:
                    if attempt == 2:
                        raise
                    print(f"Retrying iCloud read: {source.name}", flush=True)
            assert not destination.is_symlink(), destination
            assert sha256(destination) == source_hashes[resolved], destination
            row[f"source_{side}"] = source.relative_to(BASE).as_posix()
            row[f"sha256_{side}"] = source_hashes[resolved]
        return row

    errors = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(save_pair, pair): pair["pair_id"] for pair in pairs}
        for number, future in enumerate(as_completed(futures), start=1):
            try:
                row = future.result()
                manifests[row["outcome"]].append(row)
            except Exception as error:
                errors.append(f"{futures[future]}: {error}")
            if number % 100 == 0 or number == len(pairs):
                print(f"Processed {number}/{len(pairs)} pairs; errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError("Incomplete pairs:\n" + "\n".join(errors))

    for outcome, rows in manifests.items():
        rows.sort(key=lambda row: row["pair_id"])
        directory = OUTPUT / outcome
        actual_dirs = {path.name for path in directory.iterdir() if path.is_dir()}
        assert actual_dirs == {row["pair_id"] for row in rows}, directory
        for row in rows:
            images = {path.name for path in (directory / row["pair_id"]).iterdir() if path.is_file()}
            assert images == {Path(row["saved_a"]).name, Path(row["saved_b"]).name}
        with (directory / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "rule": "CLIP > 0.87 AND ViT > 0.50",
        "clip_threshold": CLIP_THRESHOLD,
        "vit_threshold": VIT_THRESHOLD,
        "counts_are_image_pairs": True,
        "datasets": {dataset: config["expected"] for dataset, config in CONFIGS.items()},
        "total_pairs": {"FP": 346, "FN": 451},
        "total_image_files": {"FP": 692, "FN": 902},
        "source_unique_files": len(source_hashes),
        "copied_bytes": total_bytes,
        "copy_method": "Byte-preserving copies, symlinks followed, every output SHA-256 verified",
        "FP_contents": "Two different original images per pair",
        "FN_contents": "Original image and the evaluated transformed image per pair",
        "source_paths_relative_to": str(BASE),
        "saved_paths_relative_to": str(OUTPUT),
        "cached_pair_index": "Zero-based index into the positive or negative similarity arrays",
        "similarity_sources": CONFIGS,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Export complete: {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check counts and paths without copying")
    export(parser.parse_args().dry_run)
