#!/usr/bin/env python3
"""Generate transforms for only the missing images in Transform/datasets5."""
import sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_datasets5_transforms import process_single_image

OUTPUT_DIR = Path(__file__).resolve().parent / "Transform" / "datasets5"
INPUT_DIR = Path(__file__).resolve().parent / "Datasets5"

existing = {d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()}
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
needed = []
for f in sorted(INPUT_DIR.rglob("*")):
    if f.suffix.lower() in extensions:
        folder = f"{f.parent.name}_{f.stem}"
        if folder not in existing:
            needed.append(f)

print(f"Found {len(needed)} images needing transforms")
t0 = time.time()
done = 0
for p in needed:
    result = process_single_image((p, OUTPUT_DIR))
    done += 1
    if done % 20 == 0:
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(needed) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(needed)}] {rate:.1f} img/s ETA {eta:.0f}s")

elapsed = time.time() - t0
new_folders = sum(1 for d in OUTPUT_DIR.iterdir() if d.is_dir())
total_files = sum(1 for _ in OUTPUT_DIR.rglob("*.jpg"))
print(f"\nDone! {done} images processed in {elapsed:.0f}s")
print(f"Total: {new_folders} folders, {total_files} files")
