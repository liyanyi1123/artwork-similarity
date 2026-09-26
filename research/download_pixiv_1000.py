"""
Download 1000 Pixiv images using parallel downloads via pixiv.cat proxy.

Extracts Pixiv illustration IDs from the dataset and downloads via proxy services
with ThreadPoolExecutor for speed.
"""

import os
import re
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset

OUTPUT_DIR = "pixiv1000"
MAX_IMAGES = 1000
REQUEST_TIMEOUT = 12
MAX_WORKERS = 8

PROXIES = [
    "https://pixiv.cat/{pid}.jpg",
    "https://pixiv.cat/{pid}.png",
    "https://pixiv.cat/{pid}-1.jpg",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pixiv_id(path: str) -> str | None:
    """Extract Pixiv illustration ID from file path."""
    match = re.search(r"/(\d+)_p\d+_master", path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d+)", path)
    if match:
        return match.group(1)
    return None


def download_image(pid: str, save_path: str) -> tuple[str, bool]:
    """Try to download a Pixiv image. Returns (pid, success)."""
    for proxy_url_template in PROXIES:
        url = proxy_url_template.format(pid=pid)
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                              headers={"User-Agent": "Mozilla/5.0"},
                              allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 2000:
                header = resp.content[:4]
                if header in (b'\xff\xd8\xff', b'\x89PNG', b'GIF8', b'RIFF'):
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
                    return (pid, True)
        except Exception:
            continue
    return (pid, False)


def main():
    print("Loading dataset (streaming mode)...")
    dataset = load_dataset(
        "Chars/pixiv-top-daily-illustration-2019-2020",
        split="train",
        streaming=True,
    )

    # Collect unique Pixiv IDs (up to MAX_IMAGES)
    print("Collecting Pixiv IDs from dataset...")
    ids_to_download = []
    seen = set()
    for item in dataset:
        if len(ids_to_download) >= MAX_IMAGES:
            break
        pid = extract_pixiv_id(item["text"])
        if pid and pid not in seen:
            seen.add(pid)
            # Skip already downloaded
            sp = os.path.join(OUTPUT_DIR, f"{pid}.jpg")
            if not os.path.exists(sp) or os.path.getsize(sp) == 0:
                ids_to_download.append(pid)

    print(f"Need to download {len(ids_to_download)} images")
    print(f"Using {MAX_WORKERS} parallel workers")
    print("=" * 50)

    downloaded = 0
    failed = 0
    start_time = time.time()

    # Download in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for pid in ids_to_download:
            save_path = os.path.join(OUTPUT_DIR, f"{pid}.jpg")
            future = executor.submit(download_image, pid, save_path)
            futures[future] = pid

        for i, future in enumerate(as_completed(futures)):
            pid, success = future.result()
            if success:
                downloaded += 1
            else:
                failed += 1

            if (downloaded + failed) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (downloaded + failed) / elapsed
                print(f"  Progress: {downloaded} ok / {failed} fail | "
                      f"{downloaded + failed}/{len(ids_to_download)} | "
                      f"{rate:.1f} req/s")

    elapsed = time.time() - start_time
    print("=" * 50)
    print(f"Done in {elapsed:.1f}s!")
    print(f"  Downloaded: {downloaded}")
    print(f"  Failed:     {failed}")
    print(f"  Rate:       {(downloaded+failed)/elapsed:.1f} req/s")

    # Final stats
    files = [f for f in os.listdir(OUTPUT_DIR) if os.path.isfile(os.path.join(OUTPUT_DIR, f))]
    sizes = [os.path.getsize(os.path.join(OUTPUT_DIR, f)) for f in files]
    if sizes:
        total_mb = sum(sizes) / (1024 * 1024)
        print(f"  Total: {len(files)} files, {total_mb:.1f} MB")


if __name__ == "__main__":
    main()
