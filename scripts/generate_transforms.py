#!/usr/bin/env python3
"""Portable entry point for the existing 32-variant Datasets5 operations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not source.exists():
        parser.error(f"Source does not exist: {source}")
    if source.is_dir() and (output == source or source in output.parents):
        parser.error("Output must be outside the input image directory")
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file())
    paths = [p for p in paths if p.suffix.lower() in EXTENSIONS]
    if not paths:
        parser.error("No supported images found")
    spec = importlib.util.spec_from_file_location("original_transforms", ROOT / "research/generate_datasets5_transforms.py")
    operations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(operations)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.csv").exists():
        parser.error("Output already has a manifest; choose a fresh output directory")
    source_root = source.parent if source.is_file() else source
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source", "transform", "operation", "level", "retained_for_positive_pairs"])
        for path in paths:
            relative = path.relative_to(source_root).as_posix()
            identifier = hashlib.sha256(relative.encode()).hexdigest()[:12]
            directory = output / f"{path.stem[:60]}_{identifier}"
            directory.mkdir(exist_ok=True)
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Cannot read image: {path}")
            for symbol, function, name, tunable in operations.OPERATIONS:
                settings = operations.OPERATION_SETTINGS[symbol] if tunable else [(None, "single")]
                for strength, level in settings:
                    transformed, params = function(image, strength=strength) if tunable else function(image)
                    filename = f"{symbol}_{level}_{params or name}.jpg"
                    target = directory / filename
                    if not cv2.imwrite(str(target), transformed):
                        raise OSError(f"Cannot write transform: {target}")
                    writer.writerow([relative, target.relative_to(output).as_posix(), symbol, level, int(symbol != "F2")])
    print(f"Generated {32 * len(paths):,} images and a manifest in {output}")


if __name__ == "__main__":
    main()
