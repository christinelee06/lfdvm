import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict, Counter
from math import comb
import faiss
import random
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
BUFFALO_DIR = "/home/christine/lfd/embeddings"
ANTELOPE_DIR = "/home/christine/lfd/embeddings/antelope"
D = 512                  # embedding dimension
THRESH = 0.62            # FAISS cosine similarity threshold
K_NEIGHBORS = 50         # number of nearest neighbors for clustering

# ===============================
# Helper to extract person ID
# ===============================
def person_id(fname):
    return "_".join(fname.split("_")[:-1])

# ===============================
# Load embeddings function
# ===============================
def load_embeddings(emb_dir):
    emb_files = [f for f in os.listdir(emb_dir) if f.endswith(".npy")]
    embs, labels, filelist = [], [], []
    for f in emb_files:
        path = os.path.join(emb_dir, f)
        if os.path.getsize(path) < 1024:
            continue
        vec = np.load(path)
        embs.append(vec)
        labels.append(person_id(f))
        filelist.append(f)
    embs = np.array(embs).astype("float32")
    labels = np.array(labels)
    faiss.normalize_L2(embs)
    return embs, labels, filelist

# ===============================
# FAISS clustering function
# ===============================
def cluster_embeddings(embs, threshold=THRESH):
    index = faiss.IndexFlatIP(D)
    index.add(embs)
    Dmat, Imat = index.search(embs, K_NEIGHBORS)
    visited = set()
    cluster_ids = np.full(len(embs), -1, dtype=int)
    cid = 0
    for i in range(len(embs)):
        if i in visited:
            continue
        cluster_ids[i] = cid
        visited.add(i)
        for j, score in zip(Imat[i], Dmat[i]):
            if score > threshold and j not in visited:
                cluster_ids[j] = cid
                visited.add(j)
        cid += 1
    return cluster_ids, cid

# ===============================
# Build cluster maps
# ===============================
def build_cluster_map(cluster_ids, labels, filelist):
    cluster_map = defaultdict(set)
    for cid_val, fname in zip(cluster_ids, filelist):
        cluster_map[cid_val].add(fname)
    return cluster_map

# ===============================
# Match clusters by largest intersection
# ===============================
def match_clusters(map1, map2):
    matches = {}
    for cid1, files1 in map1.items():
        best_cid = None
        best_overlap = 0
        for cid2, files2 in map2.items():
            overlap = len(files1 & files2)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cid = cid2
        if best_cid is not None:
            matches[cid1] = best_cid
    return matches

# ===============================
# Compute confusion metrics
# ===============================
def compute_metrics(cluster_map, labels_map):
    # TP & FP
    TP = 0
    FP = 0
    for files in cluster_map.values():
        lbls = [labels_map[f] for f in files]
        counts = Counter(lbls)
        tp_cluster = sum(comb(k, 2) for k in counts.values())
        total_pairs = comb(len(lbls), 2) if len(lbls) > 1 else 0
        TP += tp_cluster
        FP += total_pairs - tp_cluster
    # FN
    person_counts = Counter(labels_map.values())
    total_pos_pairs = sum(comb(k, 2) for k in person_counts.values())
    FN = total_pos_pairs - TP
    # TN
    N = len(labels_map)
    total_pairs = comb(N, 2)
    total_neg_pairs = total_pairs - total_pos_pairs
    TN = total_neg_pairs - FP
    # Derived metrics
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    return TP, FP, FN, TN, precision, recall, f1, accuracy

# ===============================
# Load embeddings
# ===============================
print("Loading Buffalo embeddings...")
buffalo_embs, buffalo_labels, buffalo_files = load_embeddings(BUFFALO_DIR)
buffalo_labels_map = {f: person_id(f) for f in buffalo_files}
print("Loaded embeddings:", len(buffalo_embs), "Unique identities:", len(set(buffalo_labels)))

print("Loading Antelope embeddings...")
antelope_embs, antelope_labels, antelope_files = load_embeddings(ANTELOPE_DIR)
antelope_labels_map = {f: person_id(f) for f in antelope_files}
print("Loaded embeddings:", len(antelope_embs), "Unique identities:", len(set(antelope_labels)))

# ===============================
# Cluster embeddings
# ===============================
print("\nClustering Buffalo embeddings...")
buffalo_cluster_ids, num_buffalo_clusters = cluster_embeddings(buffalo_embs)
print("Buffalo clusters formed:", num_buffalo_clusters)

print("Clustering Antelope embeddings...")
antelope_cluster_ids, num_antelope_clusters = cluster_embeddings(antelope_embs)
print("Antelope clusters formed:", num_antelope_clusters)
import os
import numpy as np
import faiss
from collections import defaultdict, Counter
from math import comb
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    roc_curve,
    auc
)
import random
import matplotlib.pyplot as plt

# ===============================
# CONFIG
# ===============================
EMB_DIR = "/home/christine/lfd/embeddings"
THRESH = 0.62          # tune 0.58–0.65
K = 200                # more neighbors = better precision

# ===============================
# Load embeddings
# ===============================
print("Loading embeddings...")
files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy")]

embs = []
labels = []
filelist = []

def person_id(fname):
    return "_".join(fname.split("_")[:-1])

for f in files:
    path = os.path.join(EMB_DIR, f)
    if os.path.getsize(path) < 1024:
        continue
    vec = np.load(path)
    embs.append(vec)
    labels.append(person_id(f))
    filelist.append(f)

embs = np.array(embs).astype("float32")
labels = np.array(labels)
N, D = embs.shape

print(f"Loaded embeddings: {N}")
print(f"Embedding dimension: {D}")
print("Unique identities:", len(set(labels)))

# Normalize for cosine similarity
faiss.normalize_L2(embs)

# ===============================
# Build FAISS index
# ===============================
print("\nBuilding FAISS index...")
index = faiss.IndexFlatIP(D)
index.add(embs)

print(f"Searching top-{K} neighbors...")
Dmat, Imat = index.search(embs, K)

# ===============================
# GRAPH-BASED CLUSTERING
# (connected components)
# ===============================
print("Clustering...")

visited = np.zeros(N, dtype=bool)
cluster_ids = np.full(N, -1, dtype=int)
cid = 0

for i in range(N):
    if visited[i]:
        continue

    # BFS queue
    queue = [i]
    visited[i] = True
    cluster_ids[i] = cid

    while queue:
        v = queue.pop(0)
        # check neighbors
        for sim, j in zip(Dmat[v], Imat[v]):
            if sim > THRESH and not visited[j]:
                visited[j] = True
                cluster_ids[j] = cid
                queue.append(j)

    cid += 1

num_clusters = cid
print("Clusters formed:", num_clusters)

# ===============================
# Evaluate clustering
# ===============================
print("\nEvaluating clustering...")

# Count per cluster
cluster_map = defaultdict(list)
for c, pid in zip(cluster_ids, labels):
    cluster_map[c].append(pid)

# --- TP & FP ---
TP = 0
FP = 0

for c, pids in cluster_map.items():
    counts = Counter(pids)
    tp_cluster = sum(comb(k, 2) for k in counts.values())
    total_pairs = comb(len(pids), 2)

    TP += tp_cluster
    FP += total_pairs - tp_cluster

# --- FN ---
person_counts = Counter(labels)
total_pos_pairs = sum(comb(k, 2) for k in person_counts.values())
FN = total_pos_pairs - TP

# --- TN ---
total_pairs = comb(N, 2)
total_neg_pairs = total_pairs - total_pos_pairs
TN = total_neg_pairs - FP

print("\n=== Confusion Counts ===")
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)

precision = TP / (TP + FP) if TP + FP else 0
recall = TP / (TP + FN) if TP + FN else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
accuracy = (TP + TN) / total_pairs

print("\n=== Cluster Accuracy Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

# Additional metrics
nmi = normalized_mutual_info_score(labels, cluster_ids)
ari = adjusted_rand_score(labels, cluster_ids)

print("\n=== Additional Cluster Quality ===")
print(f"NMI: {nmi:.4f}")
print(f"ARI: {ari:.4f}")

# ===============================
# ROC Curve
# ===============================
print("\nComputing ROC curve...")

scores = []
binary = []

by_person = defaultdict(list)
for f, pid in zip(filelist, labels):
    by_person[pid].append(f)

# Positive pairs
for pid, imgs in by_person.items():
    for i in range(len(imgs)):
        for j in range(i+1, len(imgs)):
            v1 = np.load(os.path.join(EMB_DIR, imgs[i]))
            v2 = np.load(os.path.join(EMB_DIR, imgs[j]))
            sim = float(np.dot(v1, v2) /
                        (np.linalg.norm(v1) * np.linalg.norm(v2)))
            scores.append(sim)
            binary.append(1)

# Negative pairs (same number)
pids = list(by_person.keys())
neg_count = len(scores)

for _ in range(neg_count):
    a, b = random.sample(pids, 2)
    f1 = random.choice(by_person[a])
    f2 = random.choice(by_person[b])
    v1 = np.load(os.path.join(EMB_DIR, f1))
    v2 = np.load(os.path.join(EMB_DIR, f2))
    sim = float(np.dot(v1, v2) /
                (np.linalg.norm(v1) * np.linalg.norm(v2)))
    scores.append(sim)
    binary.append(0)

fpr, tpr, _ = roc_curve(binary, scores)
roc_auc = auc(fpr, tpr)

print("\nROC AUC:", roc_auc)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig("roc_curve.png", dpi=200)
plt.show()
