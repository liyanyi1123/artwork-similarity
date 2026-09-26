"""
Image Similarity Comparator using CLIP and ViT

Compares all images in two folders pairwise (Cartesian product). For each pair,
it computes cosine similarity with CLIP and ViT embeddings. It prints "similar"
if CLIP similarity > CLIP_THRESHOLD AND ViT similarity > VIT_THRESHOLD,
otherwise "different".
"""

import torch
import clip
from PIL import Image
import os
import glob
import argparse
from transformers import ViTModel, ViTImageProcessor
from torch.nn.functional import cosine_similarity

# ---------------------------------------------------------------------
# Global configuration (edit these values as needed)

# ---- Folder paths ----
DEFAULT_FOLDER_A = "images/original"   # e.g., "/path/to/folderA"
DEFAULT_FOLDER_B = "images/target"   # e.g., "/path/to/folderB"

# ---- Similarity thresholds ----
CLIP_THRESHOLD = 0.8
VIT_THRESHOLD = 0.5

# ---- CLIP model settings ----
# Model name: "ViT-B/32", "ViT-B/16", "RN50", "RN101", etc.
CLIP_MODEL_NAME = "ViT-B/32"
# Local cache/download directory (leave empty "" to use default torch cache)
CLIP_CACHE_DIR = ""   # e.g., "/home/user/models/clip"

# ---- ViT (HuggingFace) model settings ----
# Model name or local folder path, e.g.:
# "google/vit-base-patch16-224-in21k"
# "facebook/dino-vitb16"
# "./my_local_vit_model"
VIT_MODEL_NAME = "google/vit-base-patch16-224-in21k"
# Local cache/download directory (leave empty "" to use default HF cache)
VIT_CACHE_DIR = ""    # e.g., "/home/user/models/huggingface"
# ---------------------------------------------------------------------


def load_clip_model(device):
    """Load OpenAI CLIP model using global CLIP_MODEL_NAME and CLIP_CACHE_DIR."""
    cache_root = CLIP_CACHE_DIR if CLIP_CACHE_DIR else None
    model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, download_root=cache_root)
    return model, preprocess


def load_vit_model(device):
    """Load ViT model using global VIT_MODEL_NAME and VIT_CACHE_DIR."""
    cache_root = VIT_CACHE_DIR if VIT_CACHE_DIR else None
    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME, cache_dir=cache_root)
    model = ViTModel.from_pretrained(VIT_MODEL_NAME, cache_dir=cache_root).to(device)
    model.eval()
    return model, processor


def get_clip_embedding(image, model, preprocess, device):
    """Extract CLIP image embedding."""
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model.encode_image(image_tensor)
    return embedding


def get_vit_embedding(image, model, processor, device):
    """Extract ViT embedding (CLS token) from the image."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use the [CLS] token as the image representation
        embedding = outputs.last_hidden_state[:, 0, :]
    return embedding


def compute_similarity(emb1, emb2):
    """Cosine similarity between two embedding vectors."""
    return cosine_similarity(emb1, emb2).item()


def main():
    parser = argparse.ArgumentParser(
        description="Compare all images from two folders using CLIP and ViT (Cartesian product)."
    )
    parser.add_argument(
        "--folder_a", type=str, default=None,
        help="Path to folder A (overrides DEFAULT_FOLDER_A)"
    )
    parser.add_argument(
        "--folder_b", type=str, default=None,
        help="Path to folder B (overrides DEFAULT_FOLDER_B)"
    )
    parser.add_argument(
        "--clip_thresh", type=float, default=None,
        help="CLIP similarity threshold (overrides global CLIP_THRESHOLD)"
    )
    parser.add_argument(
        "--vit_thresh", type=float, default=None,
        help="ViT similarity threshold (overrides global VIT_THRESHOLD)"
    )
    args = parser.parse_args()

    # Determine folder paths: command line takes precedence, otherwise use globals
    folder_a = args.folder_a if args.folder_a is not None else DEFAULT_FOLDER_A
    folder_b = args.folder_b if args.folder_b is not None else DEFAULT_FOLDER_B

    if not folder_a or not folder_b:
        print("ERROR: Both folder paths must be provided either via")
        print("  --folder_a / --folder_b arguments or by setting")
        print("  DEFAULT_FOLDER_A and DEFAULT_FOLDER_B in the script.")
        return

    # Update thresholds if given on command line
    global CLIP_THRESHOLD, VIT_THRESHOLD
    if args.clip_thresh is not None:
        CLIP_THRESHOLD = args.clip_thresh
    if args.vit_thresh is not None:
        VIT_THRESHOLD = args.vit_thresh

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load models (using global model paths)
    print("Loading CLIP model...")
    clip_model, clip_preprocess = load_clip_model(device)

    print("Loading ViT model...")
    vit_model, vit_processor = load_vit_model(device)

    # Collect image files from both folders (by filename)
    extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    files_a = {
        os.path.basename(f): f
        for f in glob.glob(os.path.join(folder_a, "*"))
        if f.lower().endswith(extensions)
    }

    files_b = {
        os.path.basename(f): f
        for f in glob.glob(os.path.join(folder_b, "*"))
        if f.lower().endswith(extensions)
    }

    if not files_a or not files_b:
        print("ERROR: One or both folders contain no image files. Exiting.")
        return

    print(f"Found {len(files_a)} images in folder A and {len(files_b)} images in folder B.")
    print(f"Comparing all {len(files_a) * len(files_b)} pairs...\n\n")

    print(f"Default CLIP Threshold={CLIP_THRESHOLD}, Default ViT Threshold={VIT_THRESHOLD}\n")
    # Iterate over all combinations
    for name_a, path_a in files_a.items():
        img_a = Image.open(path_a).convert("RGB")
        # Precompute embeddings for img_a to avoid recomputing for each b
        emb_a_clip = get_clip_embedding(img_a, clip_model, clip_preprocess, device)
        emb_a_vit = get_vit_embedding(img_a, vit_model, vit_processor, device)

        for name_b, path_b in files_b.items():
            img_b = Image.open(path_b).convert("RGB")
            emb_b_clip = get_clip_embedding(img_b, clip_model, clip_preprocess, device)
            emb_b_vit = get_vit_embedding(img_b, vit_model, vit_processor, device)

            sim_clip = compute_similarity(emb_a_clip, emb_b_clip)
            sim_vit = compute_similarity(emb_a_vit, emb_b_vit)

            if sim_clip > CLIP_THRESHOLD and sim_vit > VIT_THRESHOLD:
                decision = "similar"
            else:
                decision = "different"

            print(
                f"{name_a} <-> {name_b}: CLIP score = {sim_clip:.4f}, ViT score = {sim_vit:.4f}  ->  {decision}"
            )


if __name__ == "__main__":
    main()