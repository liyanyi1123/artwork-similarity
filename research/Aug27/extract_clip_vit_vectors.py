#!/usr/bin/env python3
"""Extract normalized CLIP and ViT image vectors from an image directory."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert images to L2-normalized CLIP (512-D) and ViT (768-D) vectors."
    )
    parser.add_argument(
        "source", type=Path, help="A source image file or a directory containing images"
    )
    parser.add_argument("output_dir", type=Path, help="Directory for the two NPZ files")
    parser.add_argument(
        "--recursive", action="store_true", help="Search for images in subdirectories"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="auto uses CUDA when available and otherwise uses CPU",
    )
    parser.add_argument(
        "--clip-model",
        default=str(REPO_ROOT / "models" / "clip-vit-base-patch32"),
    )
    parser.add_argument(
        "--vit-model",
        default=str(REPO_ROOT / "models" / "vit-base-patch16-224"),
    )
    return parser.parse_args()


def collect_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in IMAGE_EXTENSIONS else []

    iterator: Iterable[Path] = source.rglob("*") if recursive else source.iterdir()
    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images, key=lambda path: path.relative_to(source).as_posix())


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_batch(paths: list[Path]) -> list[Image.Image]:
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return images


def extract_clip(
    image_paths: list[Path], model_source: str, device: str, batch_size: int
) -> np.ndarray:
    import torch
    from transformers import CLIPModel, CLIPProcessor

    local_only = Path(model_source).exists()
    processor = CLIPProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = CLIPModel.from_pretrained(model_source, local_files_only=local_only)
    model = model.to(device).eval()

    batches = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="CLIP"):
            images = load_batch(image_paths[start : start + batch_size])
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            if isinstance(features, tuple):
                features = features[0]
            features = torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy().astype(np.float32))

    del model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def extract_vit(
    image_paths: list[Path], model_source: str, device: str, batch_size: int
) -> np.ndarray:
    import torch
    from transformers import ViTImageProcessor, ViTModel

    local_only = Path(model_source).exists()
    processor = ViTImageProcessor.from_pretrained(model_source, local_files_only=local_only)
    model = ViTModel.from_pretrained(model_source, local_files_only=local_only)
    model = model.to(device).eval()

    batches = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(image_paths), batch_size), desc="ViT"):
            images = load_batch(image_paths[start : start + batch_size])
            inputs = processor(images=images, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            features = model(**inputs).last_hidden_state[:, 0, :]
            features = torch.nn.functional.normalize(features, dim=-1)
            batches.append(features.cpu().numpy().astype(np.float32))

    del model, processor
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(batches, axis=0)


def save_vectors(
    output_path: Path,
    vectors: np.ndarray,
    image_paths: list[Path],
    dataset_dir: Path,
    model_source: str,
) -> None:
    relative_paths = np.array(
        [path.relative_to(dataset_dir).as_posix() for path in image_paths]
    )
    np.savez_compressed(
        output_path,
        embeddings=vectors,
        image_paths=relative_paths,
        filenames=np.array([path.name for path in image_paths]),
        model=np.array(model_source),
        normalized=np.array(True),
    )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    if not source.is_file() and not source.is_dir():
        raise SystemExit(f"Source must be an image file or directory: {source}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    image_paths = collect_images(source, args.recursive)
    if not image_paths:
        raise SystemExit(f"No supported images found in: {source}")

    output_dir.mkdir(parents=True, exist_ok=True)
    path_root = source.parent if source.is_file() else source
    device = resolve_device(args.device)
    print(f"Images: {len(image_paths)}")
    print(f"Device: {device}")

    clip_vectors = extract_clip(
        image_paths, args.clip_model, device=device, batch_size=args.batch_size
    )
    save_vectors(
        output_dir / "clip_vectors.npz",
        clip_vectors,
        image_paths,
        path_root,
        args.clip_model,
    )
    print(f"Saved CLIP: {clip_vectors.shape}")

    vit_vectors = extract_vit(
        image_paths, args.vit_model, device=device, batch_size=args.batch_size
    )
    save_vectors(
        output_dir / "vit_vectors.npz",
        vit_vectors,
        image_paths,
        path_root,
        args.vit_model,
    )
    print(f"Saved ViT: {vit_vectors.shape}")


if __name__ == "__main__":
    main()
