#!/usr/bin/env python3
"""Download 1000 images from wikiart CDN using ArtBench-10.csv URLs."""

import csv
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request

BASE_DIR = Path(__file__).parent
ARTBENCH_CSV = BASE_DIR.parent / "Datasets5" / "ArtBench-10.csv"
CSV_PATH = BASE_DIR / "sources.csv"
NUM_PER_STYLE = 100  # 10 styles × 100 = 1000
MAX_WORKERS = 8
TIMEOUT = 60

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def download_one(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 100:
        return True
    if dest.exists():
        dest.unlink()
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as resp:
            data = resp.read()
        if len(data) < 100:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  ✗ {dest.name}: {e}", file=sys.stderr)
        return False


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Read ArtBench-10.csv
    print(f"Reading {ARTBENCH_CSV}...")
    style_entries: dict[str, list[dict]] = {}
    with open(ARTBENCH_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            style_entries.setdefault(label, []).append(row)

    styles = sorted(style_entries.keys())
    print(f"Found {len(styles)} styles: {styles}")
    for s in styles:
        print(f"  {s}: {len(style_entries[s])} images")

    # 2. Select NUM_PER_STYLE from each
    selected: list[dict] = []
    for style in styles:
        entries = style_entries[style][:NUM_PER_STYLE]
        for entry in entries:
            selected.append({
                "url": entry["url"],
                "artist": entry.get("artist", ""),
                "title": entry.get("name", "").replace(".jpg", ""),
                "style": style,
                "width": entry.get("width", ""),
                "length": entry.get("length", ""),
            })

    print(f"\nSelected {len(selected)} images ({len(styles)} styles × {NUM_PER_STYLE})")

    # 3. Check existing files
    existing = set()
    for f in BASE_DIR.glob("*.jpg"):
        if f.stat().st_size > 100:
            existing.add(f.name)

    print(f"Already downloaded: {len(existing)}")

    # 4. Build tasks and sources
    tasks = []
    sources = []
    for i, entry in enumerate(selected):
        ext = "jpg"
        if entry["url"].lower().endswith(".png"):
            ext = "png"

        def sanitize(s):
            r = []
            for c in s.lower():
                if c.isalnum():
                    r.append(c)
                elif c in " _-":
                    if r and r[-1] != "-":
                        r.append("-")
            return "".join(r).strip("-")[:40] or "unknown"

        artist_clean = sanitize(entry["artist"])
        title_clean = sanitize(entry["title"])
        filename = f"{i+1:04d}_{artist_clean}_{title_clean}.{ext}"
        dest = BASE_DIR / filename

        sources.append({
            "filename": filename,
            "artist": entry["artist"],
            "title": entry["title"],
            "style": entry["style"],
            "dimensions": f"{entry['width']}x{entry.get('length', '')}" if entry.get("width") else "",
            "format": ext.upper(),
            "image_url": entry["url"],
        })

        if filename not in existing:
            tasks.append((entry["url"], dest))

    print(f"Need to download: {len(tasks)}/{len(selected)}")

    if not tasks:
        print("All images already downloaded!")
    else:
        # 5. Download in parallel
        print(f"\nDownloading ({MAX_WORKERS} workers)...")
        success = 0
        failed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(download_one, u, d): (u, d) for u, d in tasks}
            done = 0
            for future in as_completed(futures):
                done += 1
                if future.result():
                    success += 1
                else:
                    failed += 1
                if done % 50 == 0 or done == len(tasks):
                    print(f"  {done}/{len(tasks)}  ✓{success}  ✗{failed}")

        print(f"\nDownload done: ✓{success}  ✗{failed}")

    # 6. Write sources.csv
    print(f"Writing {CSV_PATH}...")
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "artist", "title", "style",
            "dimensions", "format", "image_url"
        ])
        writer.writeheader()
        for row in sources:
            writer.writerow(row)

    # 7. Summary
    total = len(list(BASE_DIR.glob("*.jpg")))
    print(f"\n{'='*50}")
    print(f"✓ {total} images in {BASE_DIR}")
    print(f"✓ {len(sources)} entries in {CSV_PATH}")


if __name__ == "__main__":
    main()
