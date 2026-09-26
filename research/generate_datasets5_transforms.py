#!/usr/bin/env python3
"""
Generate 32 image transforms for each of the 1000 images in Datasets5.
Output: Transform/datasets5/{category}_{image_stem}/ (32 files each)

12 operations:
  F1, F2         — 1 output each (flips)
  C, S1, S2, R, B, V, T, Sh, K, N — 3 levels each (low/mid/high)
  Total: 2 + 10×3 = 32 transforms per image
"""

import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Datasets5"
OUTPUT_DIR = BASE_DIR / "Transform" / "datasets5"

# ── Operation functions ────────────────────────────────────────────────

def op_C(img, strength=1.0):
    h, w = img.shape[:2]
    removal = min(0.05 * strength, 0.95)
    crop_ratio = 1.0 - removal
    param_str = f"crop_{crop_ratio:.3f}"
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y0, x0 = (h - new_h) // 2, (w - new_w) // 2
    cropped = img[y0:y0 + new_h, x0:x0 + new_w]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return resized, param_str

def op_S1(img, strength=1.0):
    h, w = img.shape[:2]
    shift_px = int(0.05 * w * strength)
    param_str = f"shiftX_{shift_px}px"
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, param_str

def op_S2(img, strength=1.0):
    h, w = img.shape[:2]
    shift_px = int(0.05 * h * strength)
    param_str = f"shiftY_{shift_px}px"
    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, param_str

def op_R(img, strength=1.0):
    h, w = img.shape[:2]
    angle = 2.0 * strength
    param_str = f"rot_{angle:.1f}deg"
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated, param_str

def op_B(img, strength=1.0):
    alpha = 1.0 + 0.04 * strength
    beta = 8.0 * strength
    param_str = f"alpha_{alpha:.2f}_beta_{beta:.1f}"
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted, param_str

def op_V(img, strength=1.0):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dx, dy = (x - cx) / (w / 2), (y - cy) / (h / 2)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    reduction = 0.25 * strength
    exponent = 1.5 * strength
    param_str = f"dark_{reduction:.2f}_exp_{exponent:.2f}"
    mask = np.clip(1.0 - dist * reduction, 0.0, 1.0)
    mask = np.power(mask, exponent)
    vignetted = (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
    return vignetted, param_str

def op_T(img, strength=1.0):
    result = img.astype(np.float32)
    r_factor = 1.0 + 0.06 * strength
    g_factor = 1.0 + 0.02 * strength
    b_factor = 1.0 - 0.05 * strength
    param_str = f"R{r_factor:.2f}_G{g_factor:.2f}_B{b_factor:.2f}"
    result[:, :, 2] = np.clip(result[:, :, 2] * r_factor, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * g_factor, 0, 255)
    result[:, :, 0] = np.clip(result[:, :, 0] * b_factor, 0, 255)
    return result.astype(np.uint8), param_str

def op_Sh(img, strength=1.0):
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
    identity_kernel = np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]], dtype=np.float32)
    sharpen_weight = min(0.4 * strength, 0.95)
    identity_weight = 1.0 - sharpen_weight
    param_str = f"sharp_{sharpen_weight:.2f}"
    final_kernel = sharpen_weight * sharpen_kernel + identity_weight * identity_kernel
    sharpened = cv2.filter2D(img, -1, final_kernel)
    return sharpened, param_str

def op_K(img, strength=1.0):
    h, w = img.shape[:2]
    margin = 0.03 * strength
    param_str = f"skew_{margin:.3f}"
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [0, 0],
        [w, h * margin],
        [w, h * (1 - margin)],
        [0, h]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    skewed = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return skewed, param_str

def op_N(img, strength=1.0):
    rng = np.random.default_rng(42)
    noise_std = 3.0 * strength
    param_str = f"noiseStd_{noise_std:.1f}"
    noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    grained = np.clip(result, 0, 255).astype(np.uint8)
    return grained, param_str

def op_F1(img):
    return cv2.flip(img, 1), None

def op_F2(img):
    return cv2.flip(img, 0), None

# ── Operation registry ──────────────────────────────────────────────────

OPERATION_SETTINGS = {
    "C":  [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],
    "S1": [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],
    "S2": [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],
    "R":  [(1.5, "low"),  (3.0, "mid"),  (6.0, "high")],
    "B":  [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],
    "V":  [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],
    "T":  [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],
    "Sh": [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],
    "K":  [(0.33, "low"), (1.0, "mid"),  (1.67, "high")],
    "N":  [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],
}

OPERATIONS = [
    ("F1", op_F1, "Horizontal_flip", False),
    ("F2", op_F2, "Vertical_flip", False),
    ("C",  op_C,  "Crop_Resize", True),
    ("S1", op_S1, "Horizontal_shift", True),
    ("S2", op_S2, "Vertical_shift", True),
    ("R",  op_R,  "Rotation", True),
    ("B",  op_B,  "Brightness_Contrast", True),
    ("V",  op_V,  "Vignette", True),
    ("T",  op_T,  "Color_temperature", True),
    ("Sh", op_Sh, "Sharpen", True),
    ("K",  op_K,  "Perspective_skew", True),
    ("N",  op_N,  "Film_grain", True),
]


def process_single_image(args):
    """Apply all 32 transforms to one image. (runs in worker process)"""
    img_path, output_dir = args
    category = img_path.parent.name
    stem = img_path.stem
    folder_name = f"{category}_{stem}"
    out_dir = output_dir / folder_name

    img = cv2.imread(str(img_path))
    if img is None:
        return (img_path.name, False, "read error")

    out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for symbol, func, name, tunable in OPERATIONS:
        if not tunable:
            result, _ = func(img)
            out_path = out_dir / f"{symbol}_{name}.jpg"
            cv2.imwrite(str(out_path), result)
            generated += 1
        else:
            for strength, label in OPERATION_SETTINGS[symbol]:
                result, param_str = func(img, strength=strength)
                out_path = out_dir / f"{symbol}_{label}_{param_str}_{name}.jpg"
                cv2.imwrite(str(out_path), result)
                generated += 1

    return (img_path.name, True, generated)


def gather_images(input_dir: Path) -> list[Path]:
    """Collect all images from category subdirectories."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = []
    for subdir in sorted(input_dir.iterdir()):
        if subdir.is_dir():
            for fpath in sorted(subdir.iterdir()):
                if fpath.suffix.lower() in extensions:
                    images.append(fpath)
    return images


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = gather_images(INPUT_DIR)
    print(f"Found {len(images)} images in {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Transforms per image: 32  |  Total outputs: {len(images) * 32:,}")
    print()

    t0 = time.time()
    tasks = [(p, OUTPUT_DIR) for p in images]

    # Use ProcessPoolExecutor for speed
    workers = min(8, len(images))
    done = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_single_image, t): t for t in tasks}
        for f in as_completed(futures):
            name, ok, gen = f.result()
            if ok:
                done += 1
            else:
                failed += 1
            if (done + failed) % 50 == 0 or (done + failed) == len(images):
                elapsed = time.time() - t0
                rate = (done + failed) / elapsed if elapsed > 0 else 0
                eta = (len(images) - done - failed) / rate if rate > 0 else 0
                print(f"  [{done + failed}/{len(images)}] done={done} failed={failed}  "
                      f"{rate:.1f} img/s  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done! {done} images processed, {failed} failed")
    print(f"Total: {done * 32:,} transform images generated")
    print(f"Time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
