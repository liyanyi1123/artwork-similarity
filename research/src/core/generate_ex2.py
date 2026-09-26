"""
Generate image variations using 12 base operations.
For each tunable operation, produce 3 subtle settings (low / mid / high)
that are specific to that operation.
Filenames include the level and the actual parameter values.

Now loops over all images in the 'Images' folder.
Each image gets its own output subfolder (named after the image stem).
"""
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "Images" / "ex2_ai_art"  # folder containing source images
OUT_DIR_BASE = BASE_DIR / "Output"             # base output directory

# ----------------------------------------------------------------------
# Operation functions – each takes a `strength` multiplier.
# The multiplier is interpreted differently per operation.
# Returns (image, param_string) for the filename.
# ----------------------------------------------------------------------

def op_C(img, strength=1.0):
    """Crop + Resize: removal = 0.05 * strength (crop 5% at strength=1)."""
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
    """Horizontal shift: shift = 0.05 * strength * width."""
    h, w = img.shape[:2]
    shift_px = int(0.05 * w * strength)
    param_str = f"shiftX_{shift_px}px"
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, param_str


def op_S2(img, strength=1.0):
    """Vertical shift: shift = 0.05 * strength * height."""
    h, w = img.shape[:2]
    shift_px = int(0.05 * h * strength)
    param_str = f"shiftY_{shift_px}px"
    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    shifted = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return shifted, param_str


def op_R(img, strength=1.0):
    """Rotation: angle = 2.0 * strength degrees."""
    h, w = img.shape[:2]
    angle = 2.0 * strength
    param_str = f"rot_{angle:.1f}deg"
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    return rotated, param_str


def op_B(img, strength=1.0):
    """Brightness/Contrast: alpha=1+0.04*strength, beta=8*strength."""
    alpha = 1.0 + 0.04 * strength
    beta = 8.0 * strength
    param_str = f"alpha_{alpha:.2f}_beta_{beta:.1f}"
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted, param_str


def op_V(img, strength=1.0):
    """Vignette: dark_factor=0.25*strength, exponent=1.5*strength."""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dx = (x - cx) / (w / 2)
    dy = (y - cy) / (h / 2)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    reduction = 0.25 * strength
    exponent = 1.5 * strength
    param_str = f"dark_{reduction:.2f}_exp_{exponent:.2f}"
    mask = np.clip(1.0 - dist * reduction, 0.0, 1.0)
    mask = np.power(mask, exponent)
    vignetted = (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)
    return vignetted, param_str


def op_T(img, strength=1.0):
    """Warm tint: R=1+0.06*strength, G=1+0.02*strength, B=1-0.05*strength."""
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
    """Sharpen: sharpen_weight = 0.4 * strength (capped)."""
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
    """Perspective skew: margin = 0.03 * strength."""
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
    """Film grain: noise_std = 3.0 * strength."""
    rng = np.random.default_rng(42)
    noise_std = 3.0 * strength
    param_str = f"noiseStd_{noise_std:.1f}"
    noise = rng.normal(0, noise_std, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    grained = np.clip(result, 0, 255).astype(np.uint8)
    return grained, param_str


# ----------------------------------------------------------------------
# Non‑tunable operations (flips – single output only)
# ----------------------------------------------------------------------

def op_F1(img):
    return cv2.flip(img, 1), None


def op_F2(img):
    return cv2.flip(img, 0), None


# ----------------------------------------------------------------------
# Define per‑operation strength settings (low, mid, high)
# Each operation has its own multipliers for subtle variations.
# ----------------------------------------------------------------------

OPERATION_SETTINGS = {
    "C":  [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],   # crop removal 2%, 5%, 8%
    "S1": [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],   # shift 2%, 5%, 8%
    "S2": [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],
    "R":  [(1.5, "low"),  (3.0, "mid"),  (6.0, "high")],   # rotation 1°, 2°, 3°
    "B":  [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],   # α=1.02,1.04,1.06; β=4,8,12
    "V":  [(0.4, "low"),  (1.0, "mid"),  (1.6, "high")],   # dark=0.1,0.25,0.4; exp=0.6,1.5,2.4
    "T":  [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],   # R=1.03,1.06,1.09; G=1.01,1.02,1.03; B=0.975,0.95,0.925
    "Sh": [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],   # sharpen weight 0.20, 0.40, 0.60
    "K":  [(0.33, "low"), (1.0, "mid"),  (1.67, "high")],  # skew margin 0.01, 0.03, 0.05
    "N":  [(0.5, "low"),  (1.0, "mid"),  (1.5, "high")],   # noise std 1.5, 3.0, 4.5
}

# Operations list: (symbol, function, base_name, is_tunable)
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


def get_image_files(folder):
    """Return a list of image file paths (common extensions)."""
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    return [p for p in folder.iterdir() if p.suffix.lower() in extensions]


def main():
    if not INPUT_DIR.exists():
        print(f"Input directory not found: {INPUT_DIR}")
        return

    image_paths = get_image_files(INPUT_DIR)
    if not image_paths:
        print(f"No image files found in {INPUT_DIR}")
        return

    print(f"Found {len(image_paths)} image(s) in {INPUT_DIR}")
    print(f"Base output directory: {OUT_DIR_BASE}\n")

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Failed to read {img_path.name}, skipping.")
            continue

        # Create output subfolder named after the image stem
        stem = img_path.stem
        out_dir = OUT_DIR_BASE / stem
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Processing: {img_path.name} -> {out_dir}")

        for symbol, func, name, tunable in OPERATIONS:
            if not tunable:
                # Flips – just one output
                result, _ = func(img)
                filename = f"{symbol}_{name}.jpg"
                out_path = out_dir / filename
                cv2.imwrite(str(out_path), result)
                print(f"  [{symbol}] {name} → {filename}")
            else:
                settings = OPERATION_SETTINGS.get(symbol, [(1.0, "mid")])
                for strength, label in settings:
                    result, param_str = func(img, strength=strength)
                    filename = f"{symbol}_{label}_{param_str}_{name}.jpg"
                    out_path = out_dir / filename
                    cv2.imwrite(str(out_path), result)
                    print(f"  [{symbol}] {name} ({label}, {param_str}) → {filename}")

    print(f"\nAll done. Outputs saved under {OUT_DIR_BASE}")


if __name__ == "__main__":
    main()