#!/usr/bin/env python3
"""Parallel download + validate ukiyo_e images, replacing corrupted ones."""

import csv
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request

import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "ArtBench-10.csv"
STYLE = "ukiyo_e"
NUM_NEEDED = 100
MAX_WORKERS = 12

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def is_valid(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 500:
        return False
    raw = path.read_bytes()
    if raw[:15].lower().startswith(b"<!doctype html"):
        return False
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) is not None


def download_one(url: str, dest: Path) -> bool:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=30, context=SSL_CTX) as resp:
            data = resp.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return is_valid(dest)
    except Exception:
        return False


def main():
    style_dir = BASE_DIR / STYLE
    style_dir.mkdir(parents=True, exist_ok=True)

    # 1. Find valid existing files
    valid_names = {f.name for f in style_dir.glob("*.jpg") if is_valid(f)}
    bad_count = sum(1 for f in style_dir.glob("*.jpg") if not is_valid(f))
    print(f"Before: {len(valid_names)} valid, {bad_count} corrupted")

    # Delete corrupted
    for f in style_dir.glob("*.jpg"):
        if f.name not in valid_names:
            f.unlink()

    # 2. Read all ukiyo_e URLs from CSV (sorted so we skip already-used names)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        all_entries = [(row["url"], row["name"]) for row in csv.DictReader(f)
                       if row["label"] == STYLE and "wikiart.org" in row["url"]]
    print(f"CSV: {len(all_entries)} wikiart URLs for {STYLE}")

    needed = NUM_NEEDED - len(valid_names)
    print(f"Need: {needed} more images")

    # 3. Build task list: try next URLs not already valid
    tasks = []
    for url, name in all_entries:
        if name not in valid_names:
            tasks.append((url, style_dir / name))
            if len(tasks) >= needed * 3:  # 3x overbook to handle failures
                break

    print(f"Downloading {len(tasks)} candidates (parallel, {MAX_WORKERS} workers)...")

    # 4. Download in parallel
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, url, dest): dest for url, dest in tasks}
        for future in as_completed(futures):
            dest = futures[future]
            ok = future.result()
            if ok:
                valid_names.add(dest.name)
            elif dest.exists():
                dest.unlink()
            done += 1
            if done % 20 == 0 or len(valid_names) >= NUM_NEEDED:
                print(f"  [{done}/{len(tasks)}] collected {len(valid_names)}/{NUM_NEEDED}")
            if len(valid_names) >= NUM_NEEDED:
                pool.shutdown(wait=False, cancel_futures=True)
                break

    # 5. Report
    final_valid = sum(1 for f in style_dir.glob("*.jpg") if is_valid(f))
    final_bad = sum(1 for f in style_dir.glob("*.jpg") if not is_valid(f))
    print(f"\nDone! {STYLE}: {final_valid} valid, {final_bad} corrupted")
    if final_valid < NUM_NEEDED:
        print(f"  ⚠ Still short {NUM_NEEDED - final_valid} images")


if __name__ == "__main__":
    main()
