#!/usr/bin/env python3
"""Generate 32 image transforms for datasets7 (pixiv1000 anime images)."""

import cv2, numpy as np, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "datasets7"
OUTPUT_DIR = BASE_DIR / "Transform" / "datasets7"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Operation functions ──

def op_C(img, strength=1.0):
    h, w = img.shape[:2]
    removal = min(0.05 * strength, 0.95)
    crop_ratio = 1.0 - removal
    new_h, new_w = int(h * crop_ratio), int(w * crop_ratio)
    y0, x0 = (h - new_h) // 2, (w - new_w) // 2
    cropped = img[y0:y0 + new_h, x0:x0 + new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

def op_S1(img, strength=1.0):
    h, w = img.shape[:2]
    shift_px = int(0.05 * w * strength)
    M = np.float32([[1, 0, shift_px], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_S2(img, strength=1.0):
    h, w = img.shape[:2]
    shift_px = int(0.05 * h * strength)
    M = np.float32([[1, 0, 0], [0, 1, shift_px]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_R(img, strength=1.0):
    h, w = img.shape[:2]
    angle = 2.0 * strength
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

def op_B(img, strength=1.0):
    alpha = 1.0 + 0.04 * strength; beta = 8.0 * strength
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def op_V(img, strength=1.0):
    h, w = img.shape[:2]
    y, x = np.ogrid[:h, :w]
    dx, dy = (x - w/2) / (w/2), (y - h/2) / (h/2)
    dist = np.sqrt(dx**2 + dy**2)
    reduction, exponent = 0.25 * strength, 1.5 * strength
    mask = np.clip(1.0 - dist * reduction, 0.0, 1.0)
    mask = np.power(mask, exponent)
    return (img.astype(np.float32) * mask[:,:,np.newaxis]).astype(np.uint8)

def op_T(img, strength=1.0):
    result = img.astype(np.float32)
    result[:,:,2] = np.clip(result[:,:,2] * (1.0 + 0.06*strength), 0, 255)
    result[:,:,1] = np.clip(result[:,:,1] * (1.0 + 0.02*strength), 0, 255)
    result[:,:,0] = np.clip(result[:,:,0] * (1.0 - 0.05*strength), 0, 255)
    return result.astype(np.uint8)

def op_Sh(img, strength=1.0):
    sk = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    ik = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)
    w = min(0.4*strength, 0.95)
    return cv2.filter2D(img, -1, w*sk + (1.0-w)*ik)

def op_K(img, strength=1.0):
    h, w = img.shape[:2]
    margin = 0.03 * strength
    src = np.float32([[0,0],[w,0],[w,h],[0,h]])
    dst = np.float32([[0,0],[w,h*margin],[w,h*(1-margin)],[0,h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_REFLECT_101)

def op_N(img, strength=1.0):
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0*strength, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32)+noise, 0, 255).astype(np.uint8)

def op_F1(img): return cv2.flip(img, 1)
def op_F2(img): return cv2.flip(img, 0)

OP_SETTINGS = {
    "C": [(0.4,"low"),(1.0,"mid"),(1.6,"high")],
    "S1":[(0.4,"low"),(1.0,"mid"),(1.6,"high")],
    "S2":[(0.4,"low"),(1.0,"mid"),(1.6,"high")],
    "R": [(1.5,"low"),(3.0,"mid"),(6.0,"high")],
    "B": [(0.5,"low"),(1.0,"mid"),(1.5,"high")],
    "V": [(0.4,"low"),(1.0,"mid"),(1.6,"high")],
    "T": [(0.5,"low"),(1.0,"mid"),(1.5,"high")],
    "Sh":[(0.5,"low"),(1.0,"mid"),(1.5,"high")],
    "K": [(0.33,"low"),(1.0,"mid"),(1.67,"high")],
    "N": [(0.5,"low"),(1.0,"mid"),(1.5,"high")],
}

OPERATIONS = [
    ("F1",op_F1,False),("F2",op_F2,False),
    ("C",op_C,True),("S1",op_S1,True),("S2",op_S2,True),
    ("R",op_R,True),("B",op_B,True),("V",op_V,True),
    ("T",op_T,True),("Sh",op_Sh,True),("K",op_K,True),("N",op_N,True),
]

def process_one(args):
    img_path, output_dir = args
    stem = img_path.stem
    out_dir = output_dir / stem
    img = cv2.imread(str(img_path))
    if img is None:
        return (img_path.name, False, 0)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = 0
    for symbol, func, tunable in OPERATIONS:
        if not tunable:
            result = func(img)
            cv2.imwrite(str(out_dir / f"{symbol}_{symbol}.jpg"), result)
            generated += 1
        else:
            for strength, label in OP_SETTINGS[symbol]:
                result = func(img, strength=strength)
                cv2.imwrite(str(out_dir / f"{symbol}_{label}_{symbol}.jpg"), result)
                generated += 1
    return (img_path.name, True, generated)

def gather_images():
    images = []
    for p in sorted(INPUT_DIR.iterdir()):
        if p.suffix.lower() in IMG_EXTS:
            images.append(p)
    return images

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = gather_images()
    print(f"Found {len(images)} images in datasets7/")

    # Skip already-done images
    already_done = set()
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir() and len(list(d.glob("*.jpg"))) >= 32:
            already_done.add(d.name)

    tasks = [(p, OUTPUT_DIR) for p in images if p.stem not in already_done]

    if not tasks:
        print(f"All {len(images)} images already have transforms → skipping")
        return

    print(f"Images: {len(images)} total, {len(tasks)} to process")
    print(f"Output: {OUTPUT_DIR}")

    t0 = time.time()
    done, failed = 0, 0
    with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as pool:
        futures = {pool.submit(process_one, t): t for t in tasks}
        for f in as_completed(futures):
            name, ok, gen = f.result()
            if ok: done += 1
            else: failed += 1
            if (done+failed) % 100 == 0:
                elapsed = time.time()-t0
                rate = (done+failed)/elapsed
                eta = (len(tasks)-done-failed)/rate if rate>0 else 0
                print(f"  [{done+failed}/{len(tasks)}] done={done} failed={failed}  {rate:.1f}/s  ETA {eta:.0f}s")

    elapsed = time.time()-t0
    print(f"Done in {elapsed:.1f}s: {done} ok, {failed} failed")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
