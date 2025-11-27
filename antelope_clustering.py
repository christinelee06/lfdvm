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
EMB_DIR = "/home/christine/lfd/embeddings/antelope"  # point to Antelope embeddings
D = 512  # embedding dimension
THRESH = 0.62  # FAISS cosine similarity threshold for clustering

# ===============================
# Helper to extract person ID
# ===============================
def person_id(fname):
    return "_".join(fname.split("_")[:-1])

# ===============================
# Load embeddings
# ===============================
print("Loading embeddings...")
emb_files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy")]

embs = []
labels = []
filelist = []

for f in emb_files:
    path = os.path.join(EMB_DIR, f)
    if os.path.getsize(path) < 1024:
        continue
    vec = np.load(path)
    embs.append(vec)
    labels.append(person_id(f))
    filelist.append(f)

embs = np.array(embs).astype("float32")
labels = np.array(labels)

print("Loaded embeddings:", len(embs))
print("Unique identities:", len(set(labels)))

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embs)

# ===============================
# FAISS Clustering
# ===============================
print("\nBuilding FAISS index...")
index = faiss.IndexFlatIP(D)
index.add(embs)

print("Searching nearest neighbors...")
k_neighbors = 50
Dmat, Imat = index.search(embs, k_neighbors)

visited = set()
cluster_ids = np.full(len(embs), -1, dtype=int)
cid = 0

print("Clustering...")
for i in range(len(embs)):
    if i in visited:
        continue
    cluster_ids[i] = cid
    visited.add(i)
    for j, score in zip(Imat[i], Dmat[i]):
        if score > THRESH and j not in visited:
            cluster_ids[j] = cid
            visited.add(j)
    cid += 1

num_clusters = cid
print("Clusters formed:", num_clusters)

# ===============================
# Evaluate clustering
# ===============================
print("\nEvaluating clustering...")
cluster_map = defaultdict(list)
for cid_val, pid in zip(cluster_ids, labels):
    cluster_map[cid_val].append(pid)

TP = np.int64(0)
FP = np.int64(0)

for cid_val, lbls in cluster_map.items():
    counts = Counter(lbls)
    tp_cluster = sum(comb(k, 2) for k in counts.values())
    total_pairs = comb(len(lbls), 2)
    TP += tp_cluster
    FP += total_pairs - tp_cluster

person_counts = Counter(labels)
total_pos_pairs = sum(comb(k, 2) for k in person_counts.values())
FN = total_pos_pairs - TP

N = len(labels)
total_pairs = comb(N, 2)
total_neg_pairs = total_pairs - total_pos_pairs
TN = total_neg_pairs - FP

print("\n=== Confusion Counts ===")
print("TP:", TP)
print("FP:", FP)
print("FN:", FN)
print("TN:", TN)

precision = TP / (TP + FP) if TP + FP > 0 else 0
recall = TP / (TP + FN) if TP + FN > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
accuracy = (TP + TN) / (TP + FP + FN + TN)

print("\n=== Cluster Accuracy Metrics ===")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

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
for fname, pid in zip(filelist, labels):
    by_person[pid].append(fname)

# Positive pairs
for pid, files in by_person.items():
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            v1 = np.load(os.path.join(EMB_DIR, files[i]))
            v2 = np.load(os.path.join(EMB_DIR, files[j]))
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            scores.append(sim)
            binary.append(1)

# Negative pairs (same number as positives)
neg_samples = len(scores)
pids = list(by_person.keys())
for _ in range(neg_samples):
    p1, p2 = random.sample(pids, 2)
    f1 = random.choice(by_person[p1])
    f2 = random.choice(by_person[p2])
    v1 = np.load(os.path.join(EMB_DIR, f1))
    v2 = np.load(os.path.join(EMB_DIR, f2))
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    scores.append(sim)
    binary.append(0)

fpr, tpr, thr = roc_curve(binary, scores)
roc_auc = auc(fpr, tpr)

print("\nROC AUC:", roc_auc)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve (cosine similarity)")
plt.legend()
plt.grid(True)
plt.savefig("antelope_roc_curve.png", dpi=200)
plt.show()
