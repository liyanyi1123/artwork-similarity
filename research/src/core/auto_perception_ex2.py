import os
import glob
import io
from pathlib import Path
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers import CLIPProcessor, CLIPModel
from perception import hashers
import xlsxwriter
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# CONFIGURATION – CHANGE THESE PATHS AS NEEDED
# ============================================================
IMAGES_DIR = BASE_DIR / "Images" / "ex2_ai_art"  # folder with reference images
OUTPUT_DIR = BASE_DIR / "Output"                # folder containing subfolders named after each reference image
REPORT_OUTPUT_DIR = BASE_DIR / "table"           # folder for generated Excel reports

# Report files will be named <reference_base>_report.xlsx and saved in REPORT_OUTPUT_DIR.

# Paths to your locally saved transformer models
VIT_MODEL_PATH = "google/vit-base-patch16-224"
CLIP_MODEL_PATH = BASE_DIR.parent / "clip-vit-base-patch32"

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
    return sorted(files)   # alphabetical order

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
# EMBEDDING CACHE (shared across all images)
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

# def get_clip_embedding(img_path):
#     if img_path in _embedding_cache:
#         return _embedding_cache[img_path]
#     if clip_model is None or clip_processor is None:
#         return None
#     image = Image.open(img_path).convert("RGB")
#     inputs = clip_processor(images=[image], return_tensors="pt", padding=True)
#     with torch.no_grad():
#         features = clip_model.get_image_features(pixel_values=inputs['pixel_values'])
#         if hasattr(features, 'pooler_output'):
#             features = features.pooler_output
#         features = features / features.norm(dim=-1, keepdim=True)
#     _embedding_cache[img_path] = features
#     return features

def get_clip_embedding(image_path):
    image = Image.open(image_path)
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)
    
    with torch.no_grad():
        image_features = clip_model.get_image_features(pixel_values)
        # 如果返回的是对象，则提取 pooler_output
        if hasattr(image_features, 'pooler_output'):
            image_features = image_features.pooler_output
    
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features.cpu().numpy().flatten()

# ============================================================
# IMAGE COMPRESSION & RESIZING FOR EXCEL
# ============================================================
def prepare_thumbnail(img_path, target_height=200, quality=70):
    """Resize image to target_height, compress as JPEG with given quality, return BytesIO."""
    try:
        with Image.open(img_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            w, h = img.size
            scale = target_height / h
            new_size = (int(w * scale), target_height)
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=quality, optimize=True)
            buf.seek(0)
            return buf
    except Exception as e:
        print(f"⚠️ Could not prepare thumbnail for {img_path}: {e}")
        return None

# ============================================================
# FUNCTION TO WRITE A SINGLE EXCEL REPORT FOR ONE REFERENCE
# ============================================================
def write_report_for_reference(ref_path, out_paths, ref_base, hashers_dict, hash_names):
    """Create an Excel report comparing ref_path with all out_paths."""
    if not out_paths:
        return

    # Compute hashes for this group
    group_images = [ref_path] + out_paths
    hash_store = precompute_hashes(group_images, hashers_dict)
    ref_hashes = hash_store[ref_path]

    # Compute embeddings for reference
    ref_vit_emb = get_vit_embedding(ref_path)
    ref_clip_emb = get_clip_embedding(ref_path)

    # Build results list for this reference
    results = []
    for out_path in out_paths:
        out_name = os.path.basename(out_path)
        base_name = os.path.splitext(out_name)[0]

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
            clip_sim = cosine_similarity([ref_clip_emb], [out_clip_emb])[0][0]
            # clip_sim = (ref_clip_emb @ out_clip_emb).item()
        results.append({
            "out_filename": base_name,
            "out_path": out_path,
            "vit": vit_sim,
            "clip": clip_sim,
            "distances": hash_dists
        })

    REPORT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate Excel file name: <ref_base>_report.xlsx
    excel_filename = f"{ref_base}_report.xlsx"
    excel_path = REPORT_OUTPUT_DIR / excel_filename

    workbook = xlsxwriter.Workbook(str(excel_path))
    worksheet = workbook.add_worksheet("Comparison")

    # Headers: Output Filename, Reference Image, Output Image, ViT Cosine, CLIP Cosine, then hash distances
    headers = ["Output Filename", "Reference Image", "Output Image", "ViT Cosine", "CLIP Cosine"]
    headers.extend([f"{h} Sim" for h in hash_names])
    col_widths = [25, 25, 25, 12, 12] + [10] * len(hash_names)

    for col, header in enumerate(headers):
        worksheet.write(0, col, header)
        worksheet.set_column(col, col, col_widths[col])

    def insert_compressed_image(worksheet, row, col, img_path):
        buf = prepare_thumbnail(img_path, target_height=200, quality=70)
        if buf:
            worksheet.insert_image(row, col, buf, {'x_offset': 2, 'y_offset': 2})
        else:
            worksheet.write(row, col, "[Image error]")

    # Write rows
    for i, res in enumerate(results, start=1):
        worksheet.write(i, 0, res["out_filename"])
        insert_compressed_image(worksheet, i, 1, ref_path)
        insert_compressed_image(worksheet, i, 2, res["out_path"])
        worksheet.write(i, 3, f"{res['vit']:.4f}" if res["vit"] is not None else "N/A")
        worksheet.write(i, 4, f"{res['clip']:.4f}" if res["clip"] is not None else "N/A")
        col = 5
        for h in hash_names:
            dist = res["distances"].get(h)
            sim = round(1 - dist, 6) if dist is not None else None
            worksheet.write(i, col, sim if sim is not None else "N/A")
            col += 1
        worksheet.set_row(i, 150)

    workbook.close()
    print(f"   ✅ Report saved: {excel_path} (compared {len(results)} images)")

# ============================================================
# MAIN – LOOP OVER ALL REFERENCE IMAGES, GENERATE PER‑REPORT
# ============================================================
def compare_folders(images_dir, output_dir):
    # 1. Validate folders
    if not os.path.isdir(images_dir):
        print(f"❌ Error: Images folder '{images_dir}' not found.")
        return
    if not os.path.isdir(output_dir):
        print(f"❌ Error: Output folder '{output_dir}' not found.")
        return

    print(f"Using reference images from: {images_dir}")
    print(f"Using generated outputs from: {output_dir}")
    print(f"Saving reports to: {REPORT_OUTPUT_DIR}")

    # 2. Get all reference images (alphabetical)
    ref_images = get_image_files(images_dir)
    if not ref_images:
        print(f"No image files found in '{images_dir}'.")
        return
    print(f"Found {len(ref_images)} reference images.")

    # 3. Load hashers once
    hashers_dict = get_hashers()
    hash_names = list(hashers_dict.keys())

    # 4. Process each reference image separately
    for ref_path in ref_images:
        ref_name = os.path.basename(ref_path)
        ref_base = os.path.splitext(ref_name)[0]   # without extension
        print(f"\nProcessing reference: {ref_name}")

        # Construct output subfolder
        out_subdir = os.path.join(output_dir, ref_base)
        if not os.path.isdir(out_subdir):
            print(f"⚠️ Warning: Subfolder '{out_subdir}' not found. Skipping this reference.")
            # continue
            out_subdir = os.path.join(output_dir)

        # Get all images in that subfolder (alphabetical)
        out_paths = get_image_files(out_subdir)
        if not out_paths:
            print(f"⚠️ No images found in '{out_subdir}'. Skipping.")
            continue
        print(f"   Found {len(out_paths)} images in '{out_subdir}'.")

        # Generate a separate Excel report for this reference
        write_report_for_reference(ref_path, out_paths, ref_base, hashers_dict, hash_names)

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    compare_folders(IMAGES_DIR, OUTPUT_DIR)