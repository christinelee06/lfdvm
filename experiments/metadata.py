#!/usr/bin/env python3
"""
choose_model_by_metadata.py

1) Sample a manageable set of identities/images from Buffalo & Antelope embeddings.
2) Extract image metadata: brightness, blur, (pose yaw/pitch/roll if insightface), face area ratio.
3) Determine per-image success for Buffalo, Antelope, and Intersection (nearest-neighbor correctness).
4) Label best_model per image (0=buffalo,1=antelope,2=intersection,3=none).
5) Train a decision tree to predict best_model from metadata and evaluate metrics.
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import faiss

# -------------------------
# CONFIGURE PATHS & SAMPLE
# -------------------------
BUFFALO_EMB_DIR = "/home/christine/lfd/embeddings/buffalo"   # adjust
ANTELOPE_EMB_DIR = "/home/christine/lfd/embeddings/antelope" # adjust
IMAGE_DIR = "/home/christine/lfd/images"                     # directory with original images (filenames must match embedding file prefixes)
# (if IMAGE_DIR not available, script will still run but skip image metadata that requires image files)

SAMPLES_PER_ID = 20    # tune: 10-30 is fast
NUM_IDENTITIES = 50    # tune: 30/50/100
RANDOM_SEED = 42
NN_K = 2               # nearest neighbor: we will ignore self so use k=2 and check neighbor at index 1
THRESHOLD_DEBUG = None # optional: you could set thresholds later

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -------------------------
# HELPERS
# -------------------------
def person_id_from_filename(fname):
    """Adjust this if your filename format differs; assumes <personid>_... .npy"""
    return fname.split("_")[0]

def load_embeddings_dir(emb_dir, sample_files=None):
    files = [f for f in os.listdir(emb_dir) if f.endswith(".npy")]
    if sample_files is not None:
        files = [f for f in files if f in sample_files]
    embs = []
    labels = []
    for f in files:
        vec = np.load(os.path.join(emb_dir, f)).astype("float32")
        embs.append(vec)
        labels.append(person_id_from_filename(f))
    if len(embs) == 0:
        return np.zeros((0,)), [], []
    embs = np.vstack(embs)
    return embs, labels, files

def compute_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())

def compute_blur(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

# optional InsightFace pose + bbox if available
USE_INSIGHTFACE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(providers=["CPU"])
    app.prepare(ctx_id=-1)
    USE_INSIGHTFACE = True
    print("InsightFace loaded for pose/bbox extraction.")
except Exception as e:
    print("InsightFace unavailable; skipping pose/bbox extraction.", str(e))
    USE_INSIGHTFACE = False

def extract_image_metadata(fname):
    """
    Returns: dict with brightness, blur, yaw, pitch, roll, face_area_ratio
    If IMAGE_DIR not present or face detection fails, yaw/pitch/roll/area -> np.nan
    """
    out = {"brightness": np.nan, "blur": np.nan, "yaw": np.nan, "pitch": np.nan, "roll": np.nan, "face_area_ratio": np.nan}
    imgpath = os.path.join(IMAGE_DIR, fname.replace(".npy", ".jpg"))  # adjust extension mapping if needed
    if not os.path.exists(imgpath):
        # try png
        imgpath2 = imgpath[:-4] + ".png"
        if os.path.exists(imgpath2):
            imgpath = imgpath2
        else:
            return out
    img = cv2.imread(imgpath)
    if img is None:
        return out
    out["brightness"] = compute_brightness(img)
    out["blur"] = compute_blur(img)
    h, w = img.shape[:2]

    if USE_INSIGHTFACE:
        try:
            faces = app.get(img)
            if faces:
                face = faces[0]
                # pose: face.pose is (yaw, pitch, roll) or similar
                pose = getattr(face, "pose", None)
                if pose is not None:
                    out["yaw"] = float(pose[0])
                    out["pitch"] = float(pose[1])
                    out["roll"] = float(pose[2])
                # bbox
                bbox = getattr(face, "bbox", None)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox[:4]
                    face_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
                    out["face_area_ratio"] = float(face_area) / (w * h)
        except Exception:
            pass
    else:
        # fallback: try Haar cascade face detection to get bbox size (coarse)
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
            if len(faces) > 0:
                x,y,fw,fh = faces[0]
                out["face_area_ratio"] = float(fw*fh) / (w*h)
        except Exception:
            pass
    return out

# -------------------------
# STEP A: SELECT IDENTITIES & SAMPLE FILES
# -------------------------
def build_person_to_files(directory):
    mapping = defaultdict(list)
    for f in os.listdir(directory):
        if not f.endswith(".npy"):
            continue
        pid = person_id_from_filename(f)
        mapping[pid].append(f)
    return mapping

print("Scanning embeddings to build identity lists...")
buffalo_map = build_person_to_files(BUFFALO_EMB_DIR)
antelope_map = build_person_to_files(ANTELOPE_EMB_DIR)

# restrict to identities that appear in at least one of the sets? Since filenames encode person id,
# we need identities that exist in either; but for model comparison we prefer identities present in both.
common_ids = sorted(set(buffalo_map.keys()) & set(antelope_map.keys()))
if len(common_ids) < NUM_IDENTITIES:
    # fall back: use whichever has most overlap, else use buffalo ids
    candidates = sorted(buffalo_map.keys(), key=lambda k: len(buffalo_map[k]), reverse=True)
    common_ids = candidates[:NUM_IDENTITIES]
else:
    common_ids = random.sample(common_ids, min(NUM_IDENTITIES, len(common_ids)))

print(f"Using {len(common_ids)} identities for sampling.")

sample_files_buffalo = set()
sample_files_antelope = set()
for pid in common_ids:
    bfiles = buffalo_map.get(pid, [])
    afiles = antelope_map.get(pid, [])
    # sample up to SAMPLES_PER_ID from each (if not enough files, sample with replacement from available)
    if len(bfiles) == 0 or len(afiles) == 0:
        continue
    choose_b = bfiles if len(bfiles) <= SAMPLES_PER_ID else random.sample(bfiles, SAMPLES_PER_ID)
    choose_a = afiles if len(afiles) <= SAMPLES_PER_ID else random.sample(afiles, SAMPLES_PER_ID)
    sample_files_buffalo.update(choose_b)
    sample_files_antelope.update(choose_a)

print("Sampled", len(sample_files_buffalo), "buffalo files and", len(sample_files_antelope), "antelope files.")

# -------------------------
# STEP B: LOAD SAMPLE EMBEDDINGS
# -------------------------
buffalo_embs, buffalo_labels, buffalo_files = load_embeddings_dir(BUFFALO_EMB_DIR, sample_files_buffalo)
antelope_embs, antelope_labels, antelope_files = load_embeddings_dir(ANTELOPE_EMB_DIR, sample_files_antelope)

# unify set of file basenames we'll analyze (use union so each image considered if present in either embedding set)
all_files = sorted(list(set(buffalo_files) | set(antelope_files)))
print("Total files to analyze:", len(all_files))

# Map filenames to index in each embedding array (or None)
buffalo_index = {f: i for i, f in enumerate(buffalo_files)}
antelope_index = {f: i for i, f in enumerate(antelope_files)}

# -------------------------
# STEP C: NORMALIZE EMBEDDINGS (cosine)
# -------------------------
if len(buffalo_embs) > 0:
    faiss.normalize_L2(buffalo_embs)
if len(antelope_embs) > 0:
    faiss.normalize_L2(antelope_embs)

# -------------------------
# STEP D: EXTRACT IMAGE METADATA (brightness, blur, pose, face_ratio)
# -------------------------
print("Extracting image metadata (brightness, blur, pose, face ratio)...")
metadata = {}
for f in all_files:
    md = extract_image_metadata(f)
    metadata[f] = md

# -------------------------
# STEP E: PER-IMAGE SUCCESS BY NEAREST-NEIGHBOR
# We'll say "model succeeds for this image" if its nearest neighbor (excluding itself) in the embedding set has same identity.
# -------------------------
def compute_nn_success(embs, files):
    """
    Return dict file->(success_bool, top_sim)
    """
    res = {}
    if len(files) == 0:
        return res
    # build mapping file->index
    idx_map = {f:i for i,f in enumerate(files)}
    # embeddings already normalized
    # use faiss for nearest neighbors
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    K = NN_K
    D, I = index.search(embs, K)
    for i, f in enumerate(files):
        # I[i][0] is itself, I[i][1] is nearest other
        if K < 2:
            # fallback: mark unknown
            res[f] = (False, 0.0)
            continue
        neigh_idx = I[i][1]
        sim = float(D[i][1])
        neigh_file = files[neigh_idx]
        same = person_id_from_filename(neigh_file) == person_id_from_filename(f)
        res[f] = (bool(same), sim)
    return res

print("Computing nearest-neighbor success for Buffalo...")
buffalo_nn = compute_nn_success(buffalo_embs, buffalo_files)
print("Computing nearest-neighbor success for Antelope...")
antelope_nn = compute_nn_success(antelope_embs, antelope_files)

# -------------------------
# STEP F: INTERSECTION EMBEDDING (simple average) and NN success
# -------------------------
print("Computing intersection embeddings (average of available buffalo+antelope) and NN success...")
# For each file, build intersection embedding if both exist; else skip intersection for that file.
inter_files = []
inter_embs = []
inter_labels = []

for f in all_files:
    bidx = buffalo_index.get(f, None)
    aidx = antelope_index.get(f, None)
    if bidx is not None and aidx is not None:
        bvec = buffalo_embs[bidx]
        avec = antelope_embs[aidx]
        ivec = bvec + avec
        inter_embs.append(ivec / np.linalg.norm(ivec))
        inter_files.append(f)
        inter_labels.append(person_id_from_filename(f))

inter_embs = np.vstack(inter_embs) if len(inter_embs) > 0 else np.zeros((0,))
if len(inter_embs) > 0:
    faiss.normalize_L2(inter_embs)
    inter_nn = compute_nn_success(inter_embs, inter_files)
else:
    inter_nn = {}

# -------------------------
# STEP G: BUILD DATAFRAME WITH FEATURES AND LABEL (best_model)
# best_model: choose the method that yields NN success; tie-breaker: higher top_sim; if none succeed => 3 (none)
# Map: 0=buffalo, 1=antelope, 2=intersection, 3=none
# -------------------------
rows = []
for f in all_files:
    pid = person_id_from_filename(f)
    b_success, b_sim = buffalo_nn.get(f, (False, 0.0))
    a_success, a_sim = antelope_nn.get(f, (False, 0.0))
    i_success, i_sim = inter_nn.get(f, (False, 0.0))

    # determine best
    # prefer methods that succeed; among successes choose highest sim
    modalities = []
    if b_success:
        modalities.append(("buffalo", b_sim, 0))
    if a_success:
        modalities.append(("antelope", a_sim, 1))
    if i_success:
        modalities.append(("intersection", i_sim, 2))

    if len(modalities) == 0:
        best = 3
    else:
        # pick by max similarity
        best = max(modalities, key=lambda x: x[1])[2]

    md = metadata.get(f, {})
    rows.append({
        "filename": f,
        "person_id": pid,
        "brightness": md.get("brightness", np.nan),
        "blur": md.get("blur", np.nan),
        "yaw": md.get("yaw", np.nan),
        "pitch": md.get("pitch", np.nan),
        "roll": md.get("roll", np.nan),
        "face_area_ratio": md.get("face_area_ratio", np.nan),
        "buffalo_success": int(b_success),
        "buffalo_top_sim": float(b_sim),
        "antelope_success": int(a_success),
        "antelope_top_sim": float(a_sim),
        "intersection_success": int(i_success),
        "intersection_top_sim": float(i_sim),
        "best_model": int(best)
    })

df = pd.DataFrame(rows)
out_csv = "model_choice_metadata_sample.csv"
df.to_csv(out_csv, index=False)
print(f"Saved metadata+labels to {out_csv}; rows={len(df)}")

# -------------------------
# STEP H: TRAIN DECISION TREE (predict best_model from metadata)
# Use simple features; drop NaNs by filling with medians
# -------------------------
feature_cols = ["brightness", "blur", "yaw", "pitch", "roll", "face_area_ratio"]
X = df[feature_cols].copy()
# fill NA with median
X = X.fillna(X.median())
y = df["best_model"].astype(int).values

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED, stratify=y)

clf = DecisionTreeClassifier(max_depth=4, random_state=RANDOM_SEED)
clf.fit(X_train, y_train)

# rules & feature importances
print("\n=== Decision Tree Rules ===")
print(export_text(clf, feature_names=feature_cols))

importances = dict(zip(feature_cols, clf.feature_importances_))
print("\nFeature importances:", importances)

# -------------------------
# STEP I: EVALUATE DECISION TREE
# -------------------------
y_pred = clf.predict(X_test)
print("\n=== Classification Report (decision tree predicting best_model) ===")
print(classification_report(y_test, y_pred, digits=4))

prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted")
acc = accuracy_score(y_test, y_pred)
print(f"Weighted metrics - Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, Accuracy: {acc:.4f}")

# -------------------------
# STEP J: SIMULATE DEPLOYED DECISION LOGIC TO MEASURE FINAL END-TO-END PERFORMANCE
# For each test sample, use decision tree to pick method and see if that chosen method actually succeeds (i.e., had NN success)
# -------------------------
test_idx = X_test.index
test_rows = df.loc[test_idx].reset_index(drop=True)
chosen = clf.predict(X_test)

# map chosen code to success field names
code_to_field = {0: "buffalo_success", 1: "antelope_success", 2: "intersection_success", 3: None}
successes = []
for i, row in test_rows.iterrows():
    pick = int(chosen[i])
    if pick == 3:
        successes.append(0)
    else:
        field = code_to_field[pick]
        successes.append(int(row[field]))

# compute final metrics of the selection strategy
precision_sel, recall_sel, f1_sel, _ = precision_recall_fscore_support(test_rows["best_model"], chosen, average="weighted")
acc_sel = accuracy_score(test_rows["best_model"], chosen)
print("\n=== Decision Selection vs True best_model ===")
print(f"Selection Precision (weighted): {precision_sel:.4f}, Recall: {recall_sel:.4f}, F1: {f1_sel:.4f}, Acc: {acc_sel:.4f}")

# compute practical success rate: fraction where chosen method actually succeeded (i.e., produced correct NN)
practical_success_rate = np.mean(successes)
print(f"Practical success rate of the chosen method on test set: {practical_success_rate:.4f}")

print("\nDone.")
