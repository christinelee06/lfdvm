import os
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import combinations

EMB_DIR = "/home/christine/lfd/embeddings"

def person_id(fname):
    return "_".join(fname.split("_")[:-1])

# -------------------------------
# Load all embeddings
# -------------------------------
print("Loading embeddings from:", EMB_DIR)
emb_files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy")]
total_embs = len(emb_files)
print(f"Found {total_embs} embedding files.")

embs = {}
for i, f in enumerate(emb_files, 1):
    path = os.path.join(EMB_DIR, f)

    # Skip tiny/incomplete files
    if os.path.getsize(path) < 1024:
        print(f"Skipping incomplete file: {f}")
        continue

    embs[f] = np.load(path)

    if i % 5000 == 0 or i == total_embs:
        print(f" Loaded {i}/{total_embs} files...")

# -------------------------------
# Group by person
# -------------------------------
by_person = {}
for fname in embs.keys():
    pid = person_id(fname)
    by_person.setdefault(pid, []).append(fname)

print(f"Identities found: {len(by_person)}")

scores = []
labels = []


# -------------------------------
# Positive Pairs (same person)
# -------------------------------
print("\nGenerating positive pairs (same-person)...")
pos_tasks = sum(len(files) * (len(files) - 1) // 2 for files in by_person.values())
print(f"Total positive comparisons: {pos_tasks}")

pos_done = 0
for pid, files in by_person.items():
    for f1, f2 in combinations(files, 2):
        sim = float(
            np.dot(embs[f1], embs[f2]) /
            (np.linalg.norm(embs[f1]) * np.linalg.norm(embs[f2]))
        )
        scores.append(sim)
        labels.append(1)

        pos_done += 1
        if pos_done % 10000 == 0 or pos_done == pos_tasks:
            pct = 100 * pos_done / pos_tasks
            print(f"  Positive pairs: {pos_done}/{pos_tasks} ({pct:.1f}%)")

# -------------------------------
# Negative Pairs (different people)
# -------------------------------
print("\nGenerating negative pairs (different-people)...")
pids = list(by_person.keys())

neg_tasks = sum(len(by_person[pid1]) for pid1, pid2 in combinations(pids, 2))
print(f"Total negative comparisons: {neg_tasks}")

neg_done = 0
for pid1, pid2 in combinations(pids, 2):
    for f1 in by_person[pid1]:
        f2 = by_person[pid2][0]  # sample one negative
        sim = float(
            np.dot(embs[f1], embs[f2]) /
            (np.linalg.norm(embs[f1]) * np.linalg.norm(embs[f2]))
        )
        scores.append(sim)
        labels.append(0)

        neg_done += 1
        if neg_done % 10000 == 0 or neg_done == neg_tasks:
            pct = 100 * neg_done / neg_tasks
            print(f"  Negative pairs: {neg_done}/{neg_tasks} ({pct:.1f}%)")


# -------------------------------
# ROC Curve
# -------------------------------
print("\nComputing ROC curve...")
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

print("\n=== RESULTS ===")
print("AUC:", roc_auc)

print("\nThreshold    FNIR      FPR")
print("--------------------------------")
for thr, t, f in zip(thresholds, tpr, fpr):
    fnir = 1 - t
    print(f"{thr:.4f}    {fnir:.4f}    {f:.4f}")
