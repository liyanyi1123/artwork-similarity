#!/usr/bin/env python3
"""
Plot ViT similarity between ai_art1.jpg and the other 99 original archive images.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor


project_root = Path(__file__).parents[2]
sys.path.append(str(project_root))

from src.utils.file_manager import ProjectPaths, save_figure


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
ARCHIVE_SUBDIRS = [
    "ex1_ex2_inputs",
    "ex2_ai_art",
    "ex2_car",
    "ex2_cat",
    "ex2_china_art",
    "ex2_football",
    "ex2_house",
    "ex2_nba",
    "ex2_painting",
    "ex2_view",
]
TARGET_IMAGE = "ai_art1.jpg"
TARGET_SUBDIR = "ex2_ai_art"
VIT_MODEL_NAME = "google/vit-base-patch16-224"
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def collect_archive_images(archive_dir: Path) -> list[Path]:
    images = []
    for subdir in ARCHIVE_SUBDIRS:
        current_dir = archive_dir / subdir
        if not current_dir.exists():
            continue
        for image_path in sorted(current_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(image_path)
    return images


def load_vit_model():
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(VIT_MODEL_NAME).to(DEVICE).eval()
    return processor, model


@torch.no_grad()
def compute_embeddings(image_paths: list[Path], processor, model) -> np.ndarray:
    embeddings = []
    for start in tqdm(range(0, len(image_paths), BATCH_SIZE), desc="Computing ViT embeddings"):
        batch_paths = image_paths[start:start + BATCH_SIZE]
        batch_images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=batch_images, return_tensors="pt")
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        outputs = model.vit(**inputs)
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        normalized = torch.nn.functional.normalize(cls_tokens, p=2, dim=1)
        embeddings.append(normalized.cpu().numpy())
    return np.vstack(embeddings)


def build_similarity_dataframe(image_paths: list[Path], embeddings: np.ndarray, target_path: Path) -> pd.DataFrame:
    name_to_index = {path: index for index, path in enumerate(image_paths)}
    target_index = name_to_index[target_path]
    target_embedding = embeddings[target_index]
    similarities = embeddings @ target_embedding

    rows = []
    for index, path in enumerate(image_paths):
        if path == target_path:
            continue
        rows.append(
            {
                "image_name": f"{path.parent.name}/{path.name}",
                "similarity": float(similarities[index]),
            }
        )

    return pd.DataFrame(rows).sort_values("image_name").reset_index(drop=True)


def make_plot(df: pd.DataFrame, target_label: str):
    fig, ax = plt.subplots(figsize=(24, 8))
    x = np.arange(len(df))
    colors = ["#d62728" if name.startswith("ex2_ai_art/") else "#1f77b4" for name in df["image_name"]]

    ax.bar(x, df["similarity"], color=colors, width=0.85)
    ax.axhline(0.5, color="#2ca02c", linestyle="--", linewidth=1.5, label="ViT threshold = 0.5")
    ax.set_title(f"ViT similarity: {target_label} vs. other 99 archive images")
    ax.set_xlabel("Other 99 archive images")
    ax.set_ylabel("Cosine similarity")
    ax.set_xticks(x)
    ax.set_xticklabels(df["image_name"], rotation=90, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    return fig


def main():
    paths = ProjectPaths(project_root)
    archive_dir = paths.base_dir / "Archive"
    output_dir = Path(__file__).parent / "plots"

    image_paths = collect_archive_images(archive_dir)
    target_path = archive_dir / TARGET_SUBDIR / TARGET_IMAGE
    if target_path not in image_paths:
        raise FileNotFoundError(f"Target image not found: {target_path}")
    if len(image_paths) != 100:
        raise ValueError(f"Expected 100 archive images, found {len(image_paths)}")

    processor, model = load_vit_model()
    embeddings = compute_embeddings(image_paths, processor, model)
    df = build_similarity_dataframe(image_paths, embeddings, target_path)
    fig = make_plot(df, f"{TARGET_SUBDIR}/{TARGET_IMAGE}")

    save_figure(fig, "vit_similarity_ai_art1_vs_others", output_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
