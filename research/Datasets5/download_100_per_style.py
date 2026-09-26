#!/usr/bin/env python3
"""Download 100 images per art style from ArtBench-10 CSV."""

import csv
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen, Request

BASE_DIR = Path(__file__).parent
CSV_PATH = BASE_DIR / "ArtBench-10.csv"
NUM_PER_STYLE = 100
MAX_WORKERS = 8
TIMEOUT = 60

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Bypass SSL for wikiart upload servers
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE


def download_one(url: str, dest: Path) -> tuple[str, bool]:
    if dest.exists() and dest.stat().st_size > 0:
        return (dest.name, True)
    if dest.exists():
        dest.unlink()
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(req, timeout=TIMEOUT, context=SSL_CTX) as resp:
            data = resp.read()
        if len(data) < 100:
            return (dest.name, False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return (dest.name, True)
    except Exception as e:
        print(f"  ✗ {dest.name}: {e}", file=sys.stderr)
        return (dest.name, False)


def main():
    # 1. Read CSV and group by style
    print("Reading ArtBench-10.csv ...")
    style_urls: dict[str, list[tuple[str, str]]] = {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = row["label"]
            if label not in style_urls:
                style_urls[label] = []
            style_urls[label].append((row["url"], row["name"]))

    print(f"Found {len(style_urls)} styles:")
    for s, urls in sorted(style_urls.items()):
        print(f"  {s}: {len(urls)} urls")

    # 2. Build tasks: fill up to 100 per style, skipping existing files
    tasks: list[tuple[str, str, Path]] = []
    for style in sorted(style_urls.keys()):
        style_dir = BASE_DIR / style
        existing = set(f.name for f in style_dir.glob("*.jpg")) if style_dir.exists() else set()
        needed = NUM_PER_STYLE - len(existing)
        print(f"  {style}: have {len(existing)}, need {needed}")
        if needed <= 0:
            continue
        for url, name in style_urls[style]:
            if name not in existing:
                tasks.append((url, name, style_dir / name))
                needed -= 1
                if needed == 0:
                    break
        if needed > 0:
            print(f"    ⚠ only {NUM_PER_STYLE - needed} images available for {style}")

    print(f"\nTotal to download: {len(tasks)} images")

    if not tasks:
        print("Nothing to download — all done!")
        return

    # 3. Download in parallel
    success, failed = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, t[0], t[2]): t for t in tasks}
        done = 0
        for future in as_completed(futures):
            done += 1
            _, ok = future.result()
            if ok:
                success += 1
            else:
                failed += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  ✓{success}  ✗{failed}")

    # 4. Final summary
    print(f"\n{'='*50}")
    print(f"Done! ✓ {success} downloaded, ✗ {failed} failed")
    print(f"\nPer-style counts:")
    for style in sorted(style_urls.keys()):
        style_dir = BASE_DIR / style
        count = len(list(style_dir.glob("*.jpg"))) if style_dir.exists() else 0
        bar = "█" * (count // 10) + ("░" * ((100 - count) // 10))
        print(f"  {style:<22s} {count:>3d}  {bar}")


if __name__ == "__main__":
    main()
