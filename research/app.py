import os
import numpy as np
import torch
import clip
from flask import Flask, request, jsonify
from PIL import Image
from transformers import CLIPProcessor, ViTImageProcessor, ViTModel, CLIPModel
import time

app = Flask(__name__)

# ---------- Config ----------
IMAGE_FOLDER = "data/images"
EMBEDDINGS_FILE = "dataset/embeddings.npz"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL = "models/clip-vit-base-patch32"
ViT_MODEL = "models/vit-base-patch16-224"

# ---------- Load Models ----------
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE, download_root='/app/models/clip')
clip_model.eval()

local_only = True
vit_processor = ViTImageProcessor.from_pretrained(ViT_MODEL, local_files_only=local_only)
vit_model = ViTModel.from_pretrained(ViT_MODEL, local_files_only=local_only)
vit_model = vit_model.to(DEVICE).eval()

# ---------- Embedding Functions ----------
def get_clip_embedding(image):
    image = clip_preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip_model.encode_image(image)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def get_vit_embedding(image):
    inputs = vit_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = vit_model(**inputs)
        cls_emb = outputs.last_hidden_state[:, 0, :]
    cls_emb = cls_emb / cls_emb.norm(dim=-1, keepdim=True)
    return cls_emb.cpu().numpy().flatten()

# ---------- Load / Compute Reference Embeddings ----------
def compute_reference_embeddings():
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER, exist_ok=True)
        raise FileNotFoundError(f"Folder {IMAGE_FOLDER} is empty. Please add images.")
    
    img_files = [f for f in os.listdir(IMAGE_FOLDER) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not img_files:
        raise ValueError("No images found in data/images/")
    
    filenames = []
    clip_embs = []
    vit_embs = []
    
    for fname in img_files:
        img_path = os.path.join(IMAGE_FOLDER, fname)
        image = Image.open(img_path).convert("RGB")
        clip_emb = get_clip_embedding(image)
        vit_emb = get_vit_embedding(image)
        filenames.append(fname)
        clip_embs.append(clip_emb)
        vit_embs.append(vit_emb)
    
    np.savez_compressed(
        EMBEDDINGS_FILE,
        filenames=np.array(filenames, dtype=object),
        clip_embeddings=np.array(clip_embs),
        vit_embeddings=np.array(vit_embs)
    )
    return filenames, clip_embs, vit_embs

def load_reference_embeddings():
    if os.path.exists(EMBEDDINGS_FILE):
        data = np.load(EMBEDDINGS_FILE, allow_pickle=True)
        filenames = data["filenames"].tolist()
        clip_embs = data["clip_embeddings"]
        vit_embs = data["vit_embeddings"]
        return filenames, clip_embs, vit_embs
    else:
        print("Embeddings file not found. Computing from images...")
        return compute_reference_embeddings()

# Load reference data
filenames, ref_clip_embs, ref_vit_embs = load_reference_embeddings()
n_ref = len(filenames)
print(f"Loaded {n_ref} reference embeddings.")

# ---------- Helper: Append New Embedding ----------
def append_embedding(filename, clip_emb, vit_emb):
    global filenames, ref_clip_embs, ref_vit_embs
    clip_emb = clip_emb.reshape(1, -1)
    vit_emb = vit_emb.reshape(1, -1)
    filenames.append(filename)
    ref_clip_embs = np.vstack([ref_clip_embs, clip_emb])
    ref_vit_embs = np.vstack([ref_vit_embs, vit_emb])
    np.savez_compressed(
        EMBEDDINGS_FILE,
        filenames=np.array(filenames, dtype=object),
        clip_embeddings=ref_clip_embs,
        vit_embeddings=ref_vit_embs
    )
    print(f"Added new reference: {filename}")


# ---------- Similarity Helpers ----------
def find_top_k(query_clip, query_vit, k=3):
    clip_scores = np.dot(ref_clip_embs, query_clip)
    vit_scores = np.dot(ref_vit_embs, query_vit)
    top_indices = np.argsort(clip_scores)[-k:][::-1]
    top3 = []
    for idx in top_indices:
        top3.append({
            "filename": filenames[idx],
            "clip_score": float(clip_scores[idx]),
            "vit_score": float(vit_scores[idx])
        })
    return top3

# ---------- Flask Routes ----------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    query_clip = get_clip_embedding(image)
    query_vit = get_vit_embedding(image)

    top3 = find_top_k(query_clip, query_vit, k=3)

    similar = any(
        item["clip_score"] > 0.87 and item["vit_score"] > 0.5
        for item in top3
    )

    # ----- 新增逻辑：若不相似，则添加嵌入，然后删除临时保存的图像文件 -----
    # if not similar:
    #     # 生成唯一文件名（仅用于记录，文件随后删除）
    #     ext = os.path.splitext(file.filename)[1] or ".jpg"
    #     new_fname = f"upload_{int(time.time())}{ext}"
    #     save_path = os.path.join(IMAGE_FOLDER, new_fname)
    #     try:
    #         # 保存图像（只是为了能得到一个文件名，但随即删除）
    #         os.makedirs(IMAGE_FOLDER, exist_ok=True)
    #         image.save(save_path)
    #         # 添加嵌入到参考集
    #         append_embedding(new_fname, query_clip, query_vit)
    #     finally:
    #         # 无论是否成功添加嵌入，都删除图像文件（避免残留）
    #         if os.path.exists(save_path):
    #             os.remove(save_path)

    return jsonify({
        "similar": similar,
        "top3": top3
    })
# ---------- 新增 /update 端点 ----------
@app.route("/update", methods=["POST"])
def update():
    """接收新图像，计算嵌入并保存到参考数据集"""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    # 生成唯一文件名（时间戳 + 原始文件名）
    base, ext = os.path.splitext(file.filename)
    if not ext:
        ext = ".jpg"
    new_fname = f"{int(time.time())}_{base}{ext}"
    save_path = os.path.join(IMAGE_FOLDER, new_fname)

    # 保存图像文件
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    image.save(save_path)

    # 计算嵌入
    clip_emb = get_clip_embedding(image)
    vit_emb = get_vit_embedding(image)

    # 追加到参考集
    append_embedding(new_fname, clip_emb, vit_emb)

    return jsonify({
        "status": "success",
        "message": f"Image {new_fname} added to dataset",
        "filename": new_fname,
        "clip_embedding_shape": clip_emb.shape,
        "vit_embedding_shape": vit_emb.shape
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)