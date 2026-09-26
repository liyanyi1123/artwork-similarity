#!/usr/bin/env python3
"""Render the 10 most similar negative original-image pairs in Datasets7.

These are different original images, evaluated by CLIP and ViT cosine similarity.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
EXP4_DIR = Path(__file__).resolve().parent
DATASETS7_DIR = BASE_DIR / "datasets7"
SWEEP_DIR = EXP4_DIR / "datasets7_sweep"
EMBED_DIR = SWEEP_DIR / "embeddings"
OUTPUT_IMAGE = SWEEP_DIR / "datasets7_top10_most_similar_negative_pairs.png"
OUTPUT_CSV = SWEEP_DIR / "datasets7_top10_most_similar_negative_pairs.csv"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
THUMBNAIL_SIZE = (320, 320)
TILE_WIDTH = 960
TILE_HEIGHT = 360
COLUMNS = 2
ROWS = 5
PADDING = 10
BG_COLOR = (255, 255, 255)
BORDER_COLOR = (200, 200, 200)
TEXT_COLOR = (0, 0, 0)
TITLE_FONT_SIZE = 24
LABEL_FONT_SIZE = 17
SMALL_FONT_SIZE = 14


def gather_original_paths() -> list[Path]:
    return sorted([p for p in DATASETS7_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    clip_cache = EMBED_DIR / "clip_embeddings.npz"
    vit_cache = EMBED_DIR / "vit_embeddings.npz"
    if not clip_cache.exists() or not vit_cache.exists():
        raise FileNotFoundError("Cached embeddings not found in exp4/datasets7_sweep/embeddings.")
    clip_data = np.load(clip_cache, allow_pickle=True)
    orig_clip = clip_data["orig_clip"].astype(np.float32)
    vit_data = np.load(vit_cache, allow_pickle=True)
    orig_vit = vit_data["orig_vit"].astype(np.float32)
    return orig_clip, orig_vit


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def find_top_negative_pairs(orig_clip: np.ndarray, orig_vit: np.ndarray, k: int = 10):
    clip_norm = normalize(orig_clip).astype(np.float64)
    vit_norm = normalize(orig_vit).astype(np.float64)
    n = clip_norm.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    clip_sims = np.einsum("ij,ij->i", clip_norm[tri_i], clip_norm[tri_j])
    vit_sims = np.einsum("ij,ij->i", vit_norm[tri_i], vit_norm[tri_j])
    combined = (clip_sims + vit_sims) / 2.0
    idx = np.argsort(combined)[::-1][:k]
    return [
        (int(tri_i[i]), int(tri_j[i]), float(clip_sims[i]), float(vit_sims[i]), float(combined[i]))
        for i in idx
    ]


def load_thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        thumb = Image.new("RGB", THUMBNAIL_SIZE, BG_COLOR)
        x = (THUMBNAIL_SIZE[0] - img.width) // 2
        y = (THUMBNAIL_SIZE[1] - img.height) // 2
        thumb.paste(img, (x, y))
    return thumb


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/Library/Fonts/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render_image(pairs: list[tuple[int, int, float, float, float]], originals: list[Path]) -> Image.Image:
    width = TILE_WIDTH * COLUMNS
    height = TILE_HEIGHT * ROWS
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(TITLE_FONT_SIZE)
    label_font = load_font(LABEL_FONT_SIZE)
    small_font = load_font(SMALL_FONT_SIZE)

    title = "Datasets7 Top 10 Most Similar Negative Original Pairs"
    draw.text((PADDING, PADDING), title, fill=TEXT_COLOR, font=title_font)
    start_y = PADDING + TITLE_FONT_SIZE + 8

    for rank, (i, j, clip_sim, vit_sim, combined) in enumerate(pairs, start=1):
        col = (rank - 1) % COLUMNS
        row = (rank - 1) // COLUMNS
        x0 = col * TILE_WIDTH
        y0 = start_y + row * TILE_HEIGHT
        box = (x0 + PADDING, y0 + PADDING, x0 + TILE_WIDTH - PADDING, y0 + TILE_HEIGHT - PADDING)
        draw.rectangle(box, outline=BORDER_COLOR, width=2)

        thumb_a = load_thumbnail(originals[i])
        thumb_b = load_thumbnail(originals[j])
        img_x = x0 + PADDING + 10
        img_y = y0 + PADDING + 10
        canvas.paste(thumb_a, (img_x, img_y))
        canvas.paste(thumb_b, (img_x + THUMBNAIL_SIZE[0] + 10, img_y))

        text_x = img_x + THUMBNAIL_SIZE[0] * 2 + 30
        text_y = img_y
        draw.text((text_x, text_y), f"Pair #{rank}", fill=TEXT_COLOR, font=label_font)
        text_y += LABEL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"Image A: {originals[i].name}", fill=TEXT_COLOR, font=small_font)
        text_y += SMALL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"Image B: {originals[j].name}", fill=TEXT_COLOR, font=small_font)
        text_y += SMALL_FONT_SIZE + 10
        draw.text((text_x, text_y), f"CLIP similarity: {clip_sim:.4f}", fill=TEXT_COLOR, font=label_font)
        text_y += LABEL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"ViT similarity: {vit_sim:.4f}", fill=TEXT_COLOR, font=label_font)
        text_y += LABEL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"Avg similarity: {combined:.4f}", fill=TEXT_COLOR, font=label_font)

    return canvas


def save_csv(pairs: list[tuple[int, int, float, float, float]], originals: list[Path]) -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "image_a", "image_b", "clip_similarity", "vit_similarity", "avg_similarity"])
        for rank, (i, j, clip_sim, vit_sim, combined) in enumerate(pairs, start=1):
            writer.writerow([rank, originals[i].name, originals[j].name, f"{clip_sim:.6f}", f"{vit_sim:.6f}", f"{combined:.6f}"])


def main() -> None:
    originals = gather_original_paths()
    if not originals:
        raise FileNotFoundError("No original images found in datasets7.")
    orig_clip, orig_vit = load_embeddings()
    if len(originals) != orig_clip.shape[0] or len(originals) != orig_vit.shape[0]:
        raise ValueError("Original image count does not match cached embedding count.")

    pairs = find_top_negative_pairs(orig_clip, orig_vit, k=10)
    save_csv(pairs, originals)
    image = render_image(pairs, originals)
    image.save(OUTPUT_IMAGE, format="PNG")
    print(f"Saved CSV to {OUTPUT_CSV}")
    print(f"Saved image to {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
