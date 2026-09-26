"""
12 base image operations — generate all combinations of 1, 2, 3, and 4 operations
Total: C(12,1) + C(12,2) + C(12,3) + C(12,4) = 12 + 66 + 220 + 495 = 793 output images.
"""
import cv2
import numpy as np
from pathlib import Path
import itertools

BASE_DIR = Path(__file__).parent
SRC_PATH = BASE_DIR / "Images" / "horses.jpg"
OUT_DIR = BASE_DIR / "Output" / "combinations_1_to_4"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 12 base operations (same as original) ──────────────────────────────

def op_F1(img):
    """F₁: Horizontal flip"""
    return cv2.flip(img, 1)

def op_F2(img):
    """F₂: Vertical flip"""
    return cv2.flip(img, 0)

def op_C(img):
    """C: Crop + Resize — crop ~95% then resize back to original"""
    h, w = img.shape[:2]
    crop_ratio = 0.95
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y0, x0 = (h - new_h) // 2, (w - new_w) // 2
    cropped = img[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

def op_S1(img):
    """S₁: Horizontal shift — shift right by 5% width, reflect edges"""
    h, w = img.shape[:2]
    shift_px = int(0.05 * w)
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_S2(img):
    """S₂: Vertical shift — shift down by 5% height, reflect edges"""
    h, w = img.shape[:2]
    shift_px = int(0.05 * h)
    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_R(img):
    """R: Tiny rotation — rotate 2° clockwise"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_B(img):
    """B: Brightness/Contrast tweak — slightly brighter + more contrast"""
    return cv2.convertScaleAbs(img, alpha=1.04, beta=8)

def op_V(img):
    """V: Vignette — subtle darkening at edges and corners"""
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w / 2, h / 2
    dx = (x - cx) / (w / 2)
    dy = (y - cy) / (h / 2)
    dist = np.sqrt(dx ** 2 + dy ** 2)
    mask = np.clip(1.0 - dist * 0.25, 0.0, 1.0)
    mask = np.power(mask, 1.5)
    return (img.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)

def op_T(img):
    """T: Color temp/tint shift — warm tone (boost R, reduce B)"""
    result = img.astype(np.float32)
    result[:, :, 2] = np.clip(result[:, :, 2] * 1.06, 0, 255)  # R up
    result[:, :, 1] = np.clip(result[:, :, 1] * 1.02, 0, 255)  # G slight up
    result[:, :, 0] = np.clip(result[:, :, 0] * 0.95, 0, 255)  # B down
    return result.astype(np.uint8)

def op_Sh(img):
    """Sh: Sharpen — subtle unsharp-mask style"""
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]], dtype=np.float32)
    kernel = kernel * 0.4 + np.eye(3, dtype=np.float32)[::-1] * 0.0
    kernel_2d = np.array([[0, 0, 0],
                           [0, 1, 0],
                           [0, 0, 0]], dtype=np.float32)
    final_kernel = kernel * 0.4 + kernel_2d * 0.6
    return cv2.filter2D(img, -1, final_kernel)

def op_K(img):
    """K: Minor perspective skew — top-right corner shifted outward"""
    h, w = img.shape[:2]
    margin = 0.03
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([
        [0, 0],
        [w, h * margin],          # top-right shifted down
        [w, h * (1 - margin)],    # bottom-right shifted up
        [0, h]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_N(img):
    """N: Faint film grain / Noise — subtle Gaussian noise"""
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0, img.shape).astype(np.float32)
    result = img.astype(np.float32) + noise
    return np.clip(result, 0, 255).astype(np.uint8)


# ── List of (symbol, function, description) ────────────────────────────
OPERATIONS = [
    ("F1",  op_F1,  "Horizontal flip"),
    ("F2",  op_F2,  "Vertical flip"),
    ("C",   op_C,   "Crop + Resize"),
    ("S1",  op_S1,  "Horizontal shift"),
    ("S2",  op_S2,  "Vertical shift"),
    ("R",   op_R,   "Tiny rotation"),
    ("B",   op_B,   "Brightness/Contrast tweak"),
    ("V",   op_V,   "Vignette"),
    ("T",   op_T,   "Color temp/tint shift"),
    ("Sh",  op_Sh,  "Sharpen"),
    ("K",   op_K,   "Minor perspective skew"),
    ("N",   op_N,   "Faint film grain/Noise"),
]

# Extract only the function objects for sequential application
OP_FUNCS = [func for _, func, _ in OPERATIONS]
OP_SYMBOLS = [sym for sym, _, _ in OPERATIONS]


def apply_operations(img, indices):
    """Apply a list of operation functions (by index) in sequence."""
    result = img.copy()
    for idx in indices:
        result = OP_FUNCS[idx](result)
    return result


def main():
    if not SRC_PATH.exists():
        print(f"Source image not found: {SRC_PATH}")
        return

    img = cv2.imread(str(SRC_PATH))
    if img is None:
        print(f"Failed to read image: {SRC_PATH}")
        return

    print(f"Source: {SRC_PATH}  shape={img.shape}")
    print(f"Output directory: {OUT_DIR}")
    print("Generating all combinations of 1, 2, 3, and 4 operations...")
    print(f"Expected total: 12 + 66 + 220 + 495 = 793 images\n")

    total = 0
    # Generate combinations for k = 1, 2, 3, 4
    for k in range(1, 5):  # k = 1, 2, 3, 4
        print(f"\n--- Generating combinations of {k} operation{'s' if k > 1 else ''} ---")
        combo_count = 0
        for combo in itertools.combinations(range(12), k):
            # Build filename from operation symbols
            filename = "_".join(OP_SYMBOLS[i] for i in combo) + ".png"
            out_path = OUT_DIR / filename

            # Apply the sequence of operations
            result_img = apply_operations(img, combo)
            cv2.imwrite(str(out_path), result_img)
            
            total += 1
            combo_count += 1
            
            if combo_count % 50 == 0:
                print(f"  {combo_count} / {len(list(itertools.combinations(range(12), k)))} done for k={k}...")
        
        # Get actual count for this k
        actual_count = len(list(itertools.combinations(range(12), k)))
        print(f"  Completed {actual_count} combinations of {k} operation{'s' if k > 1 else ''}")

    print(f"\n✅ Done! Total {total} images saved to {OUT_DIR} (expected 793)")


if __name__ == "__main__":
    main()