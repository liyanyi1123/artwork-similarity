#!/usr/bin/env python3
"""
Expand existing 10-class Open Images dataset from 10 to 100 images per class.
Preserves existing images at positions 01-10, adds 90 new ones as 11-100.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from concurrent import futures
from pathlib import Path

import boto3
import botocore
import requests
import tqdm

ROOT = Path(__file__).parent
OUTPUT_ROOT = ROOT / "datasets2"
SOURCES_FILE = OUTPUT_ROOT / "sources_openimages.txt"

ANNOTATIONS_URL = (
    "https://storage.googleapis.com/openimages/v5/"
    "train-annotations-human-imagelabels-boxable.csv"
)
BUCKET_NAME = "open-images-dataset"
MAX_WORKERS = 8
TARGET_PER_CLASS = 100

SELECTED_CLASSES: dict[str, str] = {
    "/m/01yrx": "cat",
    "/m/0bt9lr": "dog",
    "/m/0k4j": "car",
    "/m/0cmf2": "airplane",
    "/m/01mzpv": "chair",
    "/m/0c9ph5": "flower",
    "/m/015p6": "bird",
    "/m/0ch_cf": "fish",
    "/m/07j7r": "tree",
    "/m/03jm5": "house",
}


# ── Step 1: read existing IDs from sources file ──────

def read_existing_ids() -> dict[str, list[str]]:
    """Parse sources_openimages.txt to recover original 100 image IDs."""
    text = SOURCES_FILE.read_text()
    existing: dict[str, list[str]] = {}

    for label, dir_name in SELECTED_CLASSES.items():
        # match "Class: cat ..." block, extract all train/<id>.jpg URLs
        pattern = rf"Class: {dir_name}.*?\n-+\n(.*?)(?=\n\n|\nClass:|\Z)"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            ids = re.findall(r"train/(\w+)\.jpg", match.group(1))
            existing[label] = ids
        else:
            existing[label] = []

    return existing


# ── Step 2: stream-pick additional IDs ────────────────

def pick_additional_ids(
    existing: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Stream CSV, pick 90 more per class beyond existing 10.
    Skip any image already assigned (from existing or newly picked).
    """
    target_labels = set(SELECTED_CLASSES)
    need_per_class = {l: TARGET_PER_CLASS - len(existing.get(l, [])) for l in SELECTED_CLASSES}
    collected: dict[str, list[str]] = {l: [] for l in SELECTED_CLASSES}
    used: set[str] = set()

    # pre-mark existing IDs as used
    for ids in existing.values():
        used.update(ids)

    total_needed = sum(need_per_class.values())
    print(f"Need {total_needed} more IDs ({TARGET_PER_CLASS - 10} per class)")

    resp = requests.get(ANNOTATIONS_URL, stream=True, timeout=600)
    resp.raise_for_status()

    buf = ""
    rows = 0
    downloaded = 0

    for chunk in resp.iter_content(chunk_size=65536):
        downloaded += len(chunk)
        buf += chunk.decode("utf-8", errors="replace")

        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            rows += 1
            if rows == 1:
                continue

            parts = line.split(",")
            if len(parts) < 4:
                continue
            if parts[3].strip() != "1":
                continue

            label = parts[2].strip()
            if label not in target_labels:
                continue

            iid = parts[0].strip()
            if not iid or iid in used:
                continue

            if len(collected[label]) >= need_per_class[label]:
                # check if all done
                if all(len(collected[l]) >= need_per_class[l] for l in SELECTED_CLASSES):
                    resp.close()
                    print(f"  Done after {rows:,} rows.")
                    return collected
                continue

            collected[label].append(iid)
            used.add(iid)

        # progress
        mb = downloaded // 1_048_576
        if mb > 0 and mb % 10 == 0 and downloaded % 1_048_576 < 65536:
            got = sum(len(v) for v in collected.values())
            print(f"  {mb} MB | {rows:,} rows | {got}/{total_needed} new IDs", flush=True)

    print(f"  Finished: {rows:,} rows")
    return collected


# ── Step 3: download images ───────────────────────────

def _download_one(bucket, s3_key: str, dest: Path) -> bool:
    try:
        bucket.download_file(s3_key, str(dest))
        return True
    except botocore.exceptions.ClientError:
        return False


def download_new_images(
    bucket, existing: dict[str, list[str]], additional: dict[str, list[str]]
) -> int:
    """Download additional images, name them <class>11.jpg through <class>100.jpg."""
    total_ok = 0

    for label, dir_name in SELECTED_CLASSES.items():
        class_dir = OUTPUT_ROOT / dir_name
        existing_count = len(existing.get(label, []))

        new_ids = additional.get(label, [])
        if not new_ids:
            print(f"  {dir_name}: no new images needed")
            continue

        # Build download tasks, numbering from existing_count+1
        tasks: list[tuple[str, str, Path]] = []
        for i, iid in enumerate(new_ids, start=existing_count + 1):
            dest = class_dir / f"{dir_name}{i:02d}.jpg"
            tasks.append((iid, f"train/{iid}.jpg", dest))

        ok = 0
        with tqdm.tqdm(total=len(tasks), desc=f"  {dir_name}", leave=True) as pbar:
            with futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futs = {
                    ex.submit(_download_one, bucket, s3_key, dest): iid
                    for iid, s3_key, dest in tasks
                }
                for f in futures.as_completed(futs):
                    if f.result():
                        ok += 1
                    pbar.update(1)
        total_ok += ok

    return total_ok


# ── Step 4: update sources file ───────────────────────

def update_sources(existing: dict[str, list[str]], additional: dict[str, list[str]]):
    """Rewrite sources_openimages.txt with all 1000 image URLs."""
    lines: list[str] = []
    lines.append("Open Images Dataset - 1000 Image Sources")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Dataset:    Open Images V5 (Boxable subset)")
    lines.append("Website:    https://storage.googleapis.com/openimages/web/index.html")
    lines.append(
        "Annotation: https://storage.googleapis.com/openimages/v5/"
        "train-annotations-human-imagelabels-boxable.csv"
    )
    lines.append("License:    https://creativecommons.org/licenses/by/4.0/ (CC BY 4.0)")
    lines.append(
        'Paper:      "The Open Images Dataset V4", Kuznetsova et al., IJCV 2020'
    )
    lines.append("            https://doi.org/10.1007/s11263-020-01316-z")
    lines.append("")
    lines.append("Download:   Amazon S3 unsigned access")
    lines.append("Bucket:     open-images-dataset")
    lines.append("Split:      train")
    lines.append(
        f"Filters:    Confidence=1 (positive samples only), "
        f"{TARGET_PER_CLASS} images per class, no duplicates"
    )
    lines.append("")
    lines.append("=" * 70)
    lines.append("S3 Paths by Class")
    lines.append("=" * 70)

    for label, dir_name in SELECTED_CLASSES.items():
        all_ids = existing.get(label, []) + additional.get(label, [])
        lines.append("")
        lines.append(f"Class: {dir_name} (LabelName: {label})")
        lines.append("-" * 40)
        for i, iid in enumerate(all_ids, start=1):
            lines.append(
                f"{i:03d}  https://open-images-dataset.s3.amazonaws.com/train/{iid}.jpg"
            )

    lines.append("")
    lines.append("=" * 70)
    lines.append("Citation")
    lines.append("=" * 70)
    lines.append("If you use this data, please cite:")
    lines.append("  @article{kuznetsova2020open,")
    lines.append("    title={The Open Images Dataset V4: Unified image classification,")
    lines.append("           object detection, and visual relationship detection at scale},")
    lines.append("    author={Kuznetsova, Alina and others},")
    lines.append("    journal={IJCV},")
    lines.append("    year={2020}")
    lines.append("  }")
    lines.append("")

    SOURCES_FILE.write_text("\n".join(lines))
    print(f"\nSources updated: {SOURCES_FILE}")


# ── main ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Open Images — expand 10→100 images per class")
    print("=" * 60)

    # 1. Read existing IDs
    existing = read_existing_ids()
    for label, dir_name in SELECTED_CLASSES.items():
        print(f"  {dir_name}: {len(existing.get(label, []))} existing")

    # 2. Pick additional IDs
    print()
    additional = pick_additional_ids(existing)

    # 3. Download
    print("\nConnecting to S3 (unsigned)...")
    bucket = boto3.resource(
        "s3",
        config=botocore.config.Config(
            signature_version=botocore.UNSIGNED,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    ).Bucket(BUCKET_NAME)

    print("Downloading new images:\n")
    ok = download_new_images(bucket, existing, additional)

    # 4. Update sources
    update_sources(existing, additional)

    # 5. Summary
    total_new = sum(len(v) for v in additional.values())
    print(f"\n{'=' * 60}")
    print(f"Downloaded: {ok}/{total_new} new images")
    for dir_name in SELECTED_CLASSES.values():
        count = len(list((OUTPUT_ROOT / dir_name).glob("*.jpg")))
        print(f"  {dir_name}/: {count} images")
    print(f"Total: {sum(len(list((OUTPUT_ROOT / d).glob('*.jpg'))) for d in SELECTED_CLASSES.values())} images")


if __name__ == "__main__":
    main()
