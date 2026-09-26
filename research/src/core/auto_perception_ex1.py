import os
import glob
import io
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import CLIPProcessor, CLIPModel
from perception import hashers
import xlsxwriter

# ============================================================
# CONFIGURATION – CHANGE THESE PATHS AS NEEDED
# ============================================================
IMAGES_DIR = "../100 images (10 category)/Images_AI_ART"  # folder with ONE reference image
OUTPUT_DIR = "Output/combinations_1_to_4"          # folder with many images to compare against
EXCEL_REPORT = "test_report.xlsx"

# Paths to your locally saved transformer models
VIT_MODEL_PATH = "google/vit-base-patch16-224"
CLIP_MODEL_PATH = "../clip-vit-base-patch32"

# Supported image extensions
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# ============================================================
# HELPER: GET ALL IMAGE FILES FROM A FOLDER (alphabetical)
# ============================================================
def get_image_files(folder):
    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(files)   # alphabetical order: B, B_K, B_K_N, ...

# ============================================================
# PERCEPTUAL HASHERS
# ============================================================
def get_hashers():
    hasher_dict = {
        "pHash": hashers.PHash(),
        "aHash": hashers.AverageHash(),
        "dHash": hashers.DHash(),
        "wHash": hashers.WaveletHash(),
    }
    if hasattr(hashers, 'PDQHash'):
        hasher_dict["PDQ"] = hashers.PDQHash()
    else:
        print("⚠️ PDQHash not found. PDQ column will show 'N/A'.")
    return hasher_dict

def precompute_hashes(image_paths, hashers_dict):
    hash_store = {}
    for path in image_paths:
        hash_store[path] = {}
        for name, hasher in hashers_dict.items():
            try:
                hash_store[path][name] = hasher.compute(path)
            except Exception as e:
                print(f"⚠️ Failed to compute {name} for {os.path.basename(path)}: {e}")
                hash_store[path][name] = None
    return hash_store

def compute_distances(ref_hashes, out_hashes, hashers_dict):
    distances = {}
    for name, hasher in hashers_dict.items():
        h1 = ref_hashes.get(name)
        h2 = out_hashes.get(name)
        if h1 is not None and h2 is not None:
            try:
                distances[name] = hasher.compute_distance(h1, h2)
            except Exception:
                distances[name] = None
        else:
            distances[name] = None
    return distances

# ============================================================
# TRANSFORMER LOADING
# ============================================================
print("Loading transformer models...")
vit_processor = None
vit_model = None
clip_processor = None
clip_model = None

try:
    vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_PATH, local_files_only=False)
    vit_model = ViTForImageClassification.from_pretrained(VIT_MODEL_PATH, local_files_only=False)
    vit_model.eval()
    print("✅ ViT model loaded.")
except Exception as e:
    print(f"⚠️ ViT loading failed: {e}")

try:
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH, local_files_only=True)
    clip_model.eval()
    print("✅ CLIP model loaded.")
except Exception as e:
    print(f"⚠️ CLIP loading failed: {e}")

# ============================================================
# EMBEDDING CACHE
# ============================================================
_embedding_cache = {}

def get_vit_embedding(img_path):
    cache_key = f"vit:{img_path}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    if vit_model is None or vit_processor is None:
        return None
    image = Image.open(img_path).convert("RGB")
    inputs = vit_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = vit_model.vit(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    _embedding_cache[cache_key] = emb
    return emb

def get_clip_embedding(img_path):
    cache_key = f"clip:{img_path}"
    if cache_key in _embedding_cache:
        return _embedding_cache[cache_key]
    if clip_model is None or clip_processor is None:
        return None
    image = Image.open(img_path).convert("RGB")
    inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
        if hasattr(features, 'pooler_output'):
            features = features.pooler_output
        features = features / features.norm(dim=-1, keepdim=True)
    _embedding_cache[cache_key] = features
    return features

# ============================================================
# IMAGE COMPRESSION & RESIZING FOR EXCEL
# ============================================================
def prepare_thumbnail(img_path, target_height=200, quality=70):
    """Resize image to target_height, compress as JPEG with given quality, return BytesIO."""
    try:
        with Image.open(img_path) as img:
            # Convert to RGB (JPEG doesn't support alpha)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            # Resize preserving aspect ratio
            w, h = img.size
            scale = target_height / h
            new_size = (int(w * scale), target_height)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            # Save to BytesIO as JPEG with quality
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            buf.seek(0)
            return buf
    except Exception as e:
        print(f"⚠️ Could not prepare thumbnail for {img_path}: {e}")
        return None

# ============================================================
# MAIN COMPARISON (NO SORTING – KEEP NATURAL ORDER)
# ============================================================
def compare_folders(images_dir, output_dir, excel_path):
    # 1. Validate folders
    if not os.path.isdir(images_dir):
        print(f"❌ Error: Images folder '{images_dir}' not found.")
        return
    if not os.path.isdir(output_dir):
        print(f"❌ Error: Output folder '{output_dir}' not found.")
        return

    # 2. Get reference image (first one in Images folder)
    ref_images = get_image_files(images_dir)
    if not ref_images:
        print(f"No image files found in '{images_dir}'.")
        return
    if len(ref_images) > 1:
        print(f"⚠️ Warning: '{images_dir}' contains more than one image. Using only the first one: {ref_images[0]}")
    ref_path = ref_images[0]
    ref_name = os.path.basename(ref_path)
    print(f"Reference image: {ref_name}")

    # 3. Get all output images (already sorted alphabetically)
    out_paths = get_image_files(output_dir)
    if not out_paths:
        print(f"No image files found in '{output_dir}'.")
        return
    print(f"Found {len(out_paths)} images in output folder.")

    # 4. Pre‑compute perceptual hashes
    hashers_dict = get_hashers()
    print("⏳ Pre‑computing perceptual hashes...")
    all_images = [ref_path] + out_paths
    hash_store = precompute_hashes(all_images, hashers_dict)
    ref_hashes = hash_store[ref_path]

    # 5. Pre‑compute ViT & CLIP embeddings
    print("⏳ Pre‑computing transformer embeddings...")
    ref_vit_emb = get_vit_embedding(ref_path)
    ref_clip_emb = get_clip_embedding(ref_path)

    # 6. Prepare results in original order (no sorting)
    results = []
    for out_path in out_paths:
        out_name = os.path.basename(out_path)
        # Remove extension for display
        base_name = os.path.splitext(out_name)[0]
        print(f"Processing {out_name}...")
        # Hash distances
        hash_dists = compute_distances(ref_hashes, hash_store[out_path], hashers_dict)
        # ViT similarity
        out_vit_emb = get_vit_embedding(out_path)
        vit_sim = None
        if ref_vit_emb is not None and out_vit_emb is not None:
            vit_sim = torch.mm(ref_vit_emb, out_vit_emb.T).item()
        # CLIP similarity
        out_clip_emb = get_clip_embedding(out_path)
        clip_sim = None
        if ref_clip_emb is not None and out_clip_emb is not None:
            clip_sim = (ref_clip_emb @ out_clip_emb.T).item()
        results.append({
            "filename": base_name,      # without extension
            "path": out_path,
            "vit": vit_sim,
            "clip": clip_sim,
            "distances": hash_dists
        })

    # 7. Write Excel report using xlsxwriter
    workbook = xlsxwriter.Workbook(excel_path)
    worksheet = workbook.add_worksheet("Comparison")

    # Headers
    headers = ["Output Filename", "Reference Image", "Output Image", "ViT Cosine", "CLIP Cosine"]
    hash_names = list(hashers_dict.keys())  # e.g., ["pHash","aHash","dHash","wHash","PDQ"]
    headers.extend([f"{h} Sim" for h in hash_names])
    col_widths = [25, 25, 25, 12, 12] + [10] * len(hash_names)

    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
        worksheet.set_column(col, col, col_widths[col])

    # Insert image with compression (quality 70, resized)
    def insert_compressed_image(worksheet, row, col, img_path):
        buf = prepare_thumbnail(img_path, target_height=200, quality=70)
        if buf:
            # Pass BytesIO object as the source, with offset options
            worksheet.insert_image(row, col, buf, {'x_offset': 2, 'y_offset': 2})
        else:
            worksheet.write(row, col, "[Image error]")

    # Write rows in original order
    for i, res in enumerate(results, start=1):
        worksheet.write(i, 0, res["filename"])          # no extension
        insert_compressed_image(worksheet, i, 1, ref_path)   # reference thumbnail
        insert_compressed_image(worksheet, i, 2, res["path"]) # output thumbnail
        worksheet.write(i, 3, round(res["vit"], 4) if res["vit"] is not None else "N/A")
        worksheet.write(i, 4, round(res["clip"], 4) if res["clip"] is not None else "N/A")
        col = 5
        for h in hash_names:
            dist = res["distances"].get(h)
            sim = round(1 - dist, 6) if dist is not None else None
            worksheet.write(i, col, sim if sim is not None else "N/A")
            col += 1
        worksheet.set_row(i, 150)

    workbook.close()
    print(f"\n✅ Excel report saved to: {os.path.abspath(excel_path)}")
    print("   - Hash distances converted to similarity (1 = most similar).")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    compare_folders(IMAGES_DIR, OUTPUT_DIR, EXCEL_REPORT)