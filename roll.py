#!/usr/bin/env python3
import os
import random
import numpy as np
import cv2
import dlib
import requests
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
BUFFALO_DIR = "/home/christine/lfd/embeddings"
ANTELOPE_DIR = "/home/christine/lfd/embeddings/antelope"
TSV_FILE = "/home/christine/lfd/data/ms1m/OpenDataLab___MS-Celeb-1M/raw/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"

THRESH = 0.62
K_NEIGHBORS = 5
SAMPLE_SIZE = 500
PRINT_INTERVAL = 20

OUT_CSV = "metadata.csv"
LOG_FILE = "metadata.log"

# -----------------------------
# Logging helper
# -----------------------------
def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# -----------------------------
# Helper to load embeddings
# -----------------------------
def person_id(fname):
    return "_".join(fname.split("_")[:-1])

def load_embeddings(emb_dir):
    embs, labels, fnames = [], [], []
    for fname in os.listdir(emb_dir):
        if not fname.endswith(".npy"):
            continue
        arr = np.load(os.path.join(emb_dir, fname)).astype("float32")
        embs.append(arr)
        labels.append(person_id(fname))
        fnames.append(fname)
    if len(embs) == 0:
        return np.array([]), np.array([]), np.array([])
    return np.vstack(embs), np.array(labels), np.array(fnames)

log("Loading embeddings...")
buff_embs, buff_labels, buff_files = load_embeddings(BUFFALO_DIR)
ant_embs, ant_labels, ant_files = load_embeddings(ANTELOPE_DIR)
log(f"Loaded {len(buff_embs)} buffalo embeddings")
log(f"Loaded {len(ant_embs)} antelope embeddings")

# -----------------------------
# Face detector + landmarks
# -----------------------------
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(
    "/home/christine/lfd/shape_predictor_68_face_landmarks.dat"
)

# -----------------------------
# Pose Estimation Utils
# -----------------------------
def compute_pose(shape, w, h):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye
        (shape.part(45).x, shape.part(45).y),  # Right eye
        (shape.part(48).x, shape.part(48).y),  # Left mouth
        (shape.part(54).x, shape.part(54).y)   # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ])

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, None

    rotM, _ = cv2.Rodrigues(rvec)
    proj = np.hstack((rotM, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    pitch, yaw, roll = euler.flatten()
    return float(yaw), float(pitch), float(roll)

# -----------------------------
# Image Download + Metadata Extraction
# -----------------------------
def extract_metadata_from_url(url):
    try:
        r = requests.get(url, timeout=5)
        img_arr = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
    except Exception as e:
        log(f"Failed to download image: {url}, {e}")
        return None

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return None

    face = faces[0]
    shape = shape_predictor(gray, face)
    yaw, pitch, roll = compute_pose(shape, w, h)
    if yaw is None:
        return None

    face_area = face.width() * face.height()
    frac = face_area / (w * h)

    left_eye = gray[shape.part(36).y-3:shape.part(39).y+3,
                    shape.part(36).x-3:shape.part(39).x+3]
    right_eye = gray[shape.part(42).y-3:shape.part(45).y+3,
                     shape.part(42).x-3:shape.part(45).x+3]

    left_b = np.mean(left_eye) if left_eye.size > 0 else 0
    right_b = np.mean(right_eye) if right_eye.size > 0 else 0
    bright_diff = abs(left_b - right_b)

    return {
        "yaw": yaw,
        "pitch": pitch,
        "roll": roll,
        "face_fraction": frac,
        "eye_brightness_diff": bright_diff,
    }

# -----------------------------
# Best Model Computation
# -----------------------------
def best_model(buff_emb, ant_emb, true_label):
    d_b = np.linalg.norm(buff_embs - buff_emb, axis=1)
    idx_b = np.argsort(d_b)[:K_NEIGHBORS]
    top_b = buff_labels[idx_b]
    buffalo_correct = np.sum(top_b == true_label)

    d_a = np.linalg.norm(ant_embs - ant_emb, axis=1)
    idx_a = np.argsort(d_a)[:K_NEIGHBORS]
    top_a = ant_labels[idx_a]
    ant_correct = np.sum(top_a == true_label)

    if buffalo_correct > ant_correct:
        return "buffalo"
    elif ant_correct > buffalo_correct:
        return "antelope"
    elif buffalo_correct == ant_correct and buffalo_correct > 0:
        return "intersection"
    return "none"

# -----------------------------
# SAMPLE THE TSV (filter for embeddings first)
# -----------------------------
log("Sampling TSV rows...")
with open(TSV_FILE, "r") as f:
    rows = f.readlines()
rows = rows[1:]  # skip header

# only keep rows for which we have embeddings
valid_labels = set(buff_labels) & set(ant_labels)
rows = [r for r in rows if r.split("\t")[0] in valid_labels]

if len(rows) == 0:
    log("No rows with matching embeddings found in TSV!")
    exit(1)

sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

# -----------------------------
# MAIN EXTRACTION LOOP
# -----------------------------
metadata_list = []

log("Extracting metadata...")
for i, line in enumerate(sample):
    parts = line.strip().split("\t")
    if len(parts) < 2:
        continue
    label = parts[0]
    url = parts[1]

    md = extract_metadata_from_url(url)
    if md is None:
        log(f"Failed metadata extraction for {url}")
        continue

    fname_buff = next((f for f in buff_files if f.startswith(label)), None)
    fname_ant = next((f for f in ant_files if f.startswith(label)), None)
    if fname_buff is None or fname_ant is None:
        log(f"No embedding found for label {label}")
        continue

    buff_emb = np.load(os.path.join(BUFFALO_DIR, fname_buff))
    ant_emb = np.load(os.path.join(ANTELOPE_DIR, fname_ant))

    choice = best_model(buff_emb, ant_emb, label)
    md["label"] = label
    md["best_model"] = choice

    metadata_list.append(md)

    if (i + 1) % PRINT_INTERVAL == 0:
        log(f"Processed {i+1}/{len(sample)}")

# -----------------------------
# SAVE CSV
# -----------------------------
df = pd.DataFrame(metadata_list)
df.to_csv(OUT_CSV, index=False)
log(f"Saved {OUT_CSV} with {len(df)} rows")

# -----------------------------
# TRAIN DECISION TREE
# -----------------------------
if len(df) == 0:
    log("No metadata to train decision tree!")
    exit(1)

log("Training decision tree classifier...")
X = df[["yaw", "pitch", "roll", "face_fraction", "eye_brightness_diff"]]
y = df["best_model"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier(max_depth=6)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, pred, average="micro"
)
acc = accuracy_score(y_test, pred)

log("=== DECISION TREE METRICS ===")
log(f"Precision: {precision:.4f}")
log(f"Recall:    {recall:.4f}")
log(f"F1-score:  {f1:.4f}")
log(f"Accuracy:  {acc:.4f}")
log("Done.")
