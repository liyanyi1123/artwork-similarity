#!/usr/bin/env python3
"""Download a prefix of images from a large remote ZIP using HTTP ranges."""

import argparse
import binascii
import bz2
import csv
import io
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


DATASET_URL = (
    "https://huggingface.co/datasets/Chars/"
    "pixiv-top-daily-illustration-2019-2020/resolve/main/"
    "pixiv-top-daily-illustration-2019-2020.zip"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
LOCAL_FILE_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_SIGNATURE = 0x04034B50


def run_curl(args, expected_size=None):
    command = [
        "curl",
        "--location",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "5",
        "--retry-delay",
        "2",
        "--retry-all-errors",
    ]
    if expected_size is not None:
        # The Hugging Face redirect response itself is about 1.2 KiB. Keep a
        # small allowance while still preventing an ignored Range header from
        # streaming the complete 11+ GB archive.
        command.extend(["--max-filesize", str(max(expected_size, 4096))])
    command.extend(args)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError("curl failed: {}".format(error))
    return result.stdout


def remote_file_size(url):
    with tempfile.NamedTemporaryFile() as header_file:
        run_curl(
            [
                "--range",
                "0-0",
                "--dump-header",
                header_file.name,
                "--output",
                os.devnull,
                url,
            ]
        )
        header_file.seek(0)
        headers = header_file.read().decode("iso-8859-1")
    matches = re.findall(r"^content-range:\s*bytes\s+\d+-\d+/(\d+)\s*$", headers, re.I | re.M)
    if not matches:
        raise RuntimeError("Server did not return a Content-Range header")
    return int(matches[-1])


def fetch_range(url, start, end):
    if start < 0 or end < start:
        raise ValueError("Invalid byte range: {}-{}".format(start, end))
    expected_size = end - start + 1
    data = run_curl(
        ["--range", "{}-{}".format(start, end), url],
        expected_size=expected_size,
    )
    if len(data) != expected_size:
        raise RuntimeError(
            "Range {}-{} returned {} bytes, expected {}".format(
                start, end, len(data), expected_size
            )
        )
    return data


class CurlRangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.size = remote_file_size(url)
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Unsupported seek mode: {}".format(whence))
        if position < 0:
            raise ValueError("Negative seek position")
        self.position = min(position, self.size)
        return self.position

    def read(self, size=-1):
        if self.position >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size == 0:
            return b""
        start = self.position
        end = start + size - 1
        data = fetch_range(self.url, start, end)
        self.position += len(data)
        return data


def decompress_member(info, block):
    if len(block) < LOCAL_FILE_HEADER.size:
        raise RuntimeError("Local ZIP header is truncated")
    fields = LOCAL_FILE_HEADER.unpack_from(block)
    signature = fields[0]
    name_length = fields[-2]
    extra_length = fields[-1]
    if signature != LOCAL_FILE_SIGNATURE:
        raise RuntimeError("Invalid local ZIP header signature")

    data_start = LOCAL_FILE_HEADER.size + name_length + extra_length
    data_end = data_start + info.compress_size
    compressed = block[data_start:data_end]
    if len(compressed) != info.compress_size:
        raise RuntimeError("Compressed member data is truncated")

    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        decompressor = zipfile._get_decompressor(info.compress_type)
        data = decompressor.decompress(compressed) + decompressor.flush()
    elif info.compress_type == zipfile.ZIP_BZIP2:
        data = bz2.decompress(compressed)
    elif info.compress_type == zipfile.ZIP_LZMA:
        decompressor = zipfile._get_decompressor(info.compress_type)
        data = decompressor.decompress(compressed)
    else:
        raise RuntimeError("Unsupported ZIP compression method: {}".format(info.compress_type))

    if len(data) != info.file_size:
        raise RuntimeError(
            "Size mismatch for {}: got {}, expected {}".format(
                info.filename, len(data), info.file_size
            )
        )
    crc = binascii.crc32(data) & 0xFFFFFFFF
    if crc != info.CRC:
        raise RuntimeError("CRC mismatch for {}".format(info.filename))
    return data


def safe_basename(archive_name):
    name = Path(archive_name.replace("\\", "/")).name
    name = name.replace("\x00", "_")
    return name or "image"


def download_one(url, output_dir, order, info, end_offset):
    output_name = "{:04d}_{}".format(order, safe_basename(info.filename))
    output_path = output_dir / output_name
    if output_path.exists() and output_path.stat().st_size == info.file_size:
        return order, info, output_path, 0, True

    block = fetch_range(url, info.header_offset, end_offset - 1)
    data = decompress_member(info, block)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    temp_path.write_bytes(data)
    temp_path.replace(output_path)
    return order, info, output_path, len(block), False


def build_archive_index(url, limit):
    reader = CurlRangeReader(url)
    with zipfile.ZipFile(reader) as archive:
        infos = archive.infolist()
        images = [
            info
            for info in infos
            if not info.is_dir() and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
        ]
        selected = images[:limit]
        if len(selected) < limit:
            raise RuntimeError(
                "Archive contains only {} recognized images, fewer than requested {}".format(
                    len(selected), limit
                )
            )
        offsets = sorted(info.header_offset for info in infos)
        next_offsets = dict(zip(offsets, offsets[1:] + [archive.start_dir]))
    return reader.size, len(infos), len(images), selected, next_offsets


def write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "order",
                "saved_filename",
                "archive_path",
                "uncompressed_bytes",
                "compressed_bytes",
                "crc32",
            ]
        )
        for order, info, output_path, _, _ in sorted(rows):
            writer.writerow(
                [
                    order,
                    output_path.name,
                    info.filename,
                    info.file_size,
                    info.compress_size,
                    "{:08x}".format(info.CRC),
                ]
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination directory",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Number of images")
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads")
    parser.add_argument("--url", default=DATASET_URL, help="Remote ZIP URL")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    images_dir = args.output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Reading remote ZIP index...", flush=True)
    archive_size, member_count, image_count, selected, next_offsets = build_archive_index(
        args.url, args.limit
    )
    print(
        "Archive: {} bytes, {} members, {} images; downloading first {}.".format(
            archive_size, member_count, image_count, len(selected)
        ),
        flush=True,
    )

    started = time.monotonic()
    rows = []
    failures = []
    downloaded_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                download_one,
                args.url,
                images_dir,
                order,
                info,
                next_offsets[info.header_offset],
            ): (order, info)
            for order, info in enumerate(selected, start=1)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            order, info = futures[future]
            try:
                row = future.result()
                rows.append(row)
                downloaded_bytes += row[3]
            except Exception as error:
                failures.append((order, info.filename, str(error)))
            if completed % 25 == 0 or completed == len(futures):
                elapsed = max(time.monotonic() - started, 0.001)
                print(
                    "Progress: {}/{} complete, {} failures, {:.1f} MiB fetched, {:.1f}s".format(
                        completed,
                        len(futures),
                        len(failures),
                        downloaded_bytes / (1024 * 1024),
                        elapsed,
                    ),
                    flush=True,
                )

    write_manifest(args.output / "manifest.csv", rows)
    if failures:
        failure_path = args.output / "failures.csv"
        with failure_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["order", "archive_path", "error"])
            writer.writerows(failures)
        print("Download finished with {} failures: {}".format(len(failures), failure_path))
        return 1

    print("Download complete: {}".format(args.output), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
