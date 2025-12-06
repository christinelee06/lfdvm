import os
import numpy as np
import pandas as pd
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from math import comb
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from PIL import Image, ImageStat
import requests
from io import BytesIO
import cv2
import faiss
import random

# --- CONFIG ---
BUFFALO_DIR = "/home/christine/lfd/embeddings"
ANTELOPE_DIR = "/home/christine/lfd/embeddings/antelope"
TSV_FILE = "/home/christine/lfd/data/ms1m/OpenDataLab___MS-Celeb-1M/raw/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"
THRESH = 0.62
K_NEIGHBORS = 100
SAMPLE_SIZE = 5000  # CPU-friendly sample
PRINT_INTERVAL = 5  # how often to print progress

# --------------------------
# 1. Helper functions
# --------------------------
def person_id(fname):
    return "_".join(fname.split("_")[:-1])

def load_embeddings(emb_dir):
    print(f"Loading embeddings from {emb_dir}...")
    embs, labels, files = [], [], []
    for fname in os.listdir(emb_dir):
        if not fname.endswith(".npy"):
            continue
        path = os.path.join(emb_dir, fname)
        emb = np.load(path).astype("float32")
        embs.append(emb)
        labels.append(person_id(fname))
        files.append(fname)
    print(f"Loaded {len(embs)} embeddings from {emb_dir}")
    return np.vstack(embs), np.array(labels), np.array(files)

def cluster_embeddings(embs, threshold=THRESH, k=K_NEIGHBORS):
    print("Clustering embeddings...")
    faiss.normalize_L2(embs)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    D, I = index.search(embs, k)
    
    visited = set()
    cluster_ids = np.full(len(embs), -1, dtype=int)
    cid = 0
    for i in range(len(embs)):
        if i in visited:
            continue
        cluster_ids[i] = cid
        visited.add(i)
        for j, score in zip(I[i], D[i]):
            if score >= threshold and j not in visited:
                cluster_ids[j] = cid
                visited.add(j)
        cid += 1
    print("Clustering done.")
    return cluster_ids, cid

def extract_image_metadata_from_url(url):
    try:
        resp = requests.get(url, timeout=5)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print("Failed to load image:", url, e)
        return None
    stat = ImageStat.Stat(img)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    w, h = img.size
    face_fraction = 1.0  # placeholder
    roll = 0  # placeholder
    return {"brightness": brightness, "contrast": contrast, "face_fraction": face_fraction, "roll": roll, "filename": os.path.basename(url)}

# --------------------------
# 2. Sample TSV rows
# --------------------------
print(f"Sampling {SAMPLE_SIZE} rows from TSV...")
total_rows = sum(1 for _ in open(TSV_FILE)) - 1
sample_indices = sorted(random.sample(range(total_rows), SAMPLE_SIZE))
tsv_df = pd.read_csv(
    TSV_FILE, sep="\t", header=None, usecols=[0, 2], names=["person_id","url"],
    skiprows=lambda x: x not in sample_indices and x != 0
)
print("Sampled rows loaded.")

# --------------------------
# 3. Load embeddings
# --------------------------
buffalo_embs, buffalo_labels, buffalo_files = load_embeddings(BUFFALO_DIR)
antelope_embs, antelope_labels, antelope_files = load_embeddings(ANTELOPE_DIR)

# --------------------------
# 4. Cluster embeddings
# --------------------------
buffalo_clusters, _ = cluster_embeddings(buffalo_embs)
antelope_clusters, _ = cluster_embeddings(antelope_embs)

def compute_success(cluster_ids, labels):
    cluster_map = defaultdict(list)
    for cid, lbl in zip(cluster_ids, labels):
        cluster_map[cid].append(lbl)
    success_dict = {}
    for cid, lbls in cluster_map.items():
        counts = Counter(lbls)
        for i, lbl in enumerate(lbls):
            fname = f"{lbl}_{i}.npy"
            success_dict[fname] = int(counts[lbl] > 1)
    return success_dict

buffalo_success = compute_success(buffalo_clusters, buffalo_labels)
antelope_success = compute_success(antelope_clusters, antelope_labels)

# --------------------------
# 5. Extract metadata from sampled TSV
# --------------------------
metadata_list = []
print("Extracting metadata from sampled images...")
for idx, row in tsv_df.iterrows():
    meta = extract_image_metadata_from_url(row["url"])
    if meta is None:
        continue
    fname = f"{row['person_id']}_{idx}.npy"
    b_succ = buffalo_success.get(fname, 0)
    a_succ = antelope_success.get(fname, 0)
    intersection_succ = int(b_succ and a_succ)
    if intersection_succ:
        best_model = "intersection"
    elif b_succ:
        best_model = "buffalo"
    elif a_succ:
        best_model = "antelope"
    else:
        best_model = "none"
    meta["best_model"] = best_model
    metadata_list.append(meta)
    
    if (idx+1) % PRINT_INTERVAL == 0:
        print(f"Processed {idx+1}/{len(tsv_df)} images...")

df = pd.DataFrame(metadata_list)
print("Metadata extraction complete.")

# --------------------------
# 6. Train decision tree
# --------------------------
print("Training decision tree...")
features = ["brightness", "contrast", "face_fraction", "roll"]
X = df[features]
y = df["best_model"]

tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X, y)
print("Decision tree trained.")

# --------------------------
# 7. Evaluate tree
# --------------------------
preds = tree.predict(X)
precision = precision_score(y, preds, average="weighted", zero_division=0)
recall = recall_score(y, preds, average="weighted", zero_division=0)
f1 = f1_score(y, preds, average="weighted", zero_division=0)
accuracy = accuracy_score(y, preds)

print("\n=== Decision Tree Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

df.to_csv("metadata_best_model_sample.csv", index=False)
print("Saved metadata CSV.")
