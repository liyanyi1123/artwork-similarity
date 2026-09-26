#!/usr/bin/env python3
"""Render the 10 least similar original image pairs from Datasets7 in one image.

The output is saved to exp4/datasets7_sweep/datasets7_top10_least_similar.png.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent.parent
EXP4_DIR = Path(__file__).resolve().parent
DATASETS7_DIR = BASE_DIR / "datasets7"
SWEEP_DIR = EXP4_DIR / "datasets7_sweep"
EMBED_DIR = SWEEP_DIR / "embeddings"
OUTPUT_PATH = SWEEP_DIR / "datasets7_top10_least_similar.png"

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
TITLE_FONT_SIZE = 28
LABEL_FONT_SIZE = 18
SMALL_FONT_SIZE = 14


def gather_original_paths() -> list[Path]:
    return sorted([p for p in DATASETS7_DIR.iterdir() if p.suffix.lower() in IMG_EXTS])


def load_embeddings() -> tuple[np.ndarray, np.ndarray]:
    clip_cache = EMBED_DIR / "clip_embeddings.npz"
    vit_cache = EMBED_DIR / "vit_embeddings.npz"
    if not clip_cache.exists() or not vit_cache.exists():
        raise FileNotFoundError("Expected cached embeddings in exp4/datasets7_sweep/embeddings/")
    clip_data = np.load(clip_cache, allow_pickle=True)
    orig_clip = clip_data["orig_clip"].astype(np.float32)
    vit_data = np.load(vit_cache)
    orig_vit = vit_data["orig_vit"].astype(np.float32)
    return orig_clip, orig_vit


def normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def compute_least_similar_pairs(orig_clip: np.ndarray, orig_vit: np.ndarray, top_k: int = 10):
    clip = normalize(orig_clip)
    vit = normalize(orig_vit)
    n = clip.shape[0]
    tri_i, tri_j = np.triu_indices(n, k=1)
    clip_sims = np.sum(clip[tri_i] * clip[tri_j], axis=1)
    vit_sims = np.sum(vit[tri_i] * vit[tri_j], axis=1)
    score = (clip_sims + vit_sims) / 2.0
    idx = np.argsort(score)[:top_k]
    return [(int(tri_i[i]), int(tri_j[i]), float(clip_sims[i]), float(vit_sims[i])) for i in idx]


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


def render_image(pairs: list[tuple[int, int, float, float]], originals: list[Path]) -> Image.Image:
    width = TILE_WIDTH * COLUMNS
    height = TILE_HEIGHT * ROWS
    canvas = Image.new("RGB", (width, height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(TITLE_FONT_SIZE)
    label_font = load_font(LABEL_FONT_SIZE)
    small_font = load_font(SMALL_FONT_SIZE)

    title = "Datasets7 Top 10 Least Similar Original Image Pairs"
    draw.text((PADDING, PADDING), title, fill=TEXT_COLOR, font=title_font)

    # reserve title height
    start_y = PADDING + TITLE_FONT_SIZE + 8
    row_height = TILE_HEIGHT
    for rank, (i, j, clip_sim, vit_sim) in enumerate(pairs, start=1):
        col = (rank - 1) % COLUMNS
        row = (rank - 1) // COLUMNS
        x0 = col * TILE_WIDTH
        y0 = start_y + row * row_height
        pair_box = (x0 + PADDING, y0 + PADDING,
                    x0 + TILE_WIDTH - PADDING, y0 + TILE_HEIGHT - PADDING)
        draw.rectangle(pair_box, outline=BORDER_COLOR, width=2)

        thumb_a = load_thumbnail(originals[i])
        thumb_b = load_thumbnail(originals[j])

        img_x = x0 + PADDING + 10
        img_y = y0 + PADDING + 10
        canvas.paste(thumb_a, (img_x, img_y))
        canvas.paste(thumb_b, (img_x + THUMBNAIL_SIZE[0] + 10, img_y))

        text_x = img_x + 2 * THUMBNAIL_SIZE[0] + 30
        text_y = img_y
        draw.text((text_x, text_y), f"Pair #{rank}", fill=TEXT_COLOR, font=label_font)
        text_y += LABEL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"Image A: {originals[i].name}", fill=TEXT_COLOR, font=small_font)
        text_y += SMALL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"Image B: {originals[j].name}", fill=TEXT_COLOR, font=small_font)
        text_y += SMALL_FONT_SIZE + 8
        draw.text((text_x, text_y), f"ViT similarity: {vit_sim:.4f}", fill=TEXT_COLOR, font=label_font)
        text_y += LABEL_FONT_SIZE + 4
        draw.text((text_x, text_y), f"CLIP similarity: {clip_sim:.4f}", fill=TEXT_COLOR, font=label_font)

    return canvas


def main() -> None:
    originals = gather_original_paths()
    if not originals:
        raise RuntimeError("No original images found in datasets7.")
    orig_clip, orig_vit = load_embeddings()
    if len(originals) != orig_clip.shape[0] or len(originals) != orig_vit.shape[0]:
        raise ValueError("Original image count does not match cached embedding count.")

    pairs = compute_least_similar_pairs(orig_clip, orig_vit, top_k=10)
    image = render_image(pairs, originals)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_PATH, format="PNG")
    print(f"Saved least-similar top 10 image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
