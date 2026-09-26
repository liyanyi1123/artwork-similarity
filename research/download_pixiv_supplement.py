"""
Supplement datasets7 to 1000 Pixiv images.

Keeps existing numeric-ID images and downloads more from the
Chars/pixiv-top-daily-illustration-2019-2020 dataset via the pixiv.cat proxy
until the folder holds 1000 valid images.

Rate-limit friendly: LOW concurrency + jittered delays, because pixiv.cat
blocks aggressive parallel bursts. Failed IDs are retried in later rounds.
"""

import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets7")
CANDIDATE_CACHE = os.path.join(BASE_DIR, ".pixiv_candidates.txt")
TARGET_TOTAL = 1000
REQUEST_TIMEOUT = 20
MAX_WORKERS = 4
MIN_DELAY, MAX_DELAY = 0.4, 1.2     # seconds between request starts
MAX_COLLECT = 4000                  # candidates gathered per collect pass
MAX_ROUNDS = 8                      # download/retry rounds before giving up

PROXIES = [
    "https://pixiv.cat/{pid}.jpg",
    "https://pixiv.cat/{pid}.png",
    "https://pixiv.cat/{pid}-1.jpg",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_pixiv_id(path: str) -> str | None:
    """Extract Pixiv illustration ID from a path like .../123456_p0_master1200.jpg."""
    match = re.search(r"/(\d+)_p\d+_master", path)
    if match:
        return match.group(1)
    match = re.search(r"/(\d+)", path)
    if match:
        return match.group(1)
    return None


def existing_numeric_ids() -> set[str]:
    """Return the set of numeric-image IDs already present in OUTPUT_DIR."""
    ids = set()
    for fname in os.listdir(OUTPUT_DIR):
        m = re.fullmatch(r"(\d+)\.jpg", fname)
        if m:
            ids.add(m.group(1))
    return ids


def load_candidates() -> list[str]:
    if os.path.exists(CANDIDATE_CACHE):
        with open(CANDIDATE_CACHE) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return []


def save_candidates(cands: list[str]) -> None:
    with open(CANDIDATE_CACHE, "w") as f:
        f.write("\n".join(cands))


def collect_more(existing: set[str], seen: set[str], limit: int) -> list[str]:
    """Stream the dataset and collect up to `limit` fresh candidate IDs."""
    dataset = load_dataset(
        "Chars/pixiv-top-daily-illustration-2019-2020",
        split="train",
        streaming=True,
    )
    fresh: list[str] = []
    for item in dataset:
        if len(fresh) >= limit:
            break
        pid = extract_pixiv_id(item["text"])
        if pid and pid not in seen and pid not in existing:
            seen.add(pid)
            fresh.append(pid)
    return fresh


def download_image(pid: str, save_path: str) -> tuple[str, bool]:
    """Try to download a Pixiv image via proxy services. Returns (pid, success)."""
    for proxy_url_template in PROXIES:
        url = proxy_url_template.format(pid=pid)
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT,
                                headers={"User-Agent": "Mozilla/5.0"},
                                allow_redirects=True)
            if resp.status_code == 200 and len(resp.content) > 2000:
                header = resp.content[:4]
                if header in (b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"RIFF"):
                    with open(save_path, "wb") as f:
                        f.write(resp.content)
                    return (pid, True)
        except Exception:
            continue
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))   # slow down between URLs
    return (pid, False)


def main():
    existing = existing_numeric_ids()
    have = len(existing)
    need = TARGET_TOTAL - have
    print(f"Existing numeric IDs: {have}, need: {need}", flush=True)

    candidates = load_candidates()
    seen = set(candidates)
    print(f"Loaded {len(candidates)} cached candidates", flush=True)

    for round_idx in range(MAX_ROUNDS):
        have = len(existing_numeric_ids())
        if have >= TARGET_TOTAL:
            print(f"✅ Reached target: {have}/{TARGET_TOTAL}", flush=True)
            break

        # Top up candidates if running low
        if len(candidates) < need * 1.5:
            fresh = collect_more(existing, seen, MAX_COLLECT)
            if fresh:
                candidates.extend(fresh)
                save_candidates(candidates)
                print(f"Collected {len(fresh)} more candidates "
                      f"(total pool {len(candidates)})", flush=True)

        pending = [pid for pid in candidates
                   if pid not in existing_numeric_ids()]
        if not pending:
            print("No pending candidates left — giving up.", flush=True)
            break

        print(f"\n── Round {round_idx + 1}: {len(pending)} pending, "
              f"{have}/{TARGET_TOTAL} have ──", flush=True)
        ok = fail = 0
        start = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(download_image, pid,
                                os.path.join(OUTPUT_DIR, f"{pid}.jpg")): pid
                for pid in pending
            }
            for i, future in enumerate(as_completed(futures), 1):
                pid, success = future.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                if i % 50 == 0:
                    now_have = len(existing_numeric_ids())
                    print(f"  {i}/{len(pending)} tried | +{ok} ok / {fail} fail "
                          f"| have {now_have}/{TARGET_TOTAL} | {time.time()-start:.0f}s",
                          flush=True)
                time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))  # pace submission
        now_have = len(existing_numeric_ids())
        print(f"  Round done: +{ok} ok, have {now_have}/{TARGET_TOTAL}", flush=True)

        if now_have >= TARGET_TOTAL:
            print(f"✅ Reached target!", flush=True)
            break

    have = len(existing_numeric_ids())
    print("=" * 50, flush=True)
    print(f"FINAL: {have}/{TARGET_TOTAL} numeric-ID images", flush=True)


if __name__ == "__main__":
    main()
