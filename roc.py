import os
import numpy as np
from sklearn.metrics import roc_curve, auc
import random
import matplotlib.pyplot as plt

EMB_DIR = "/home/christine/lfd/embeddings"

def person_id(fname):
    return "_".join(fname.split("_")[:-1])

# -------------------------------
# Load embeddings
# -------------------------------
print("Loading embeddings...")
emb_files = [f for f in os.listdir(EMB_DIR) if f.endswith(".npy")]
embs = {}

for f in emb_files:
    path = os.path.join(EMB_DIR, f)
    if os.path.getsize(path) < 1024:
        continue
    embs[f] = np.load(path)

print("Loaded:", len(embs))

# -------------------------------
# Group by person
# -------------------------------
by_person = {}
for fname in embs:
    pid = person_id(fname)
    by_person.setdefault(pid, []).append(fname)

pids = list(by_person.keys())
print("Identities:", len(pids))

scores = []
labels = []

# -------------------------------
# Positive Pairs (all)
# -------------------------------
print("\nGenerating positive pairs...")
pos_tasks = sum(len(files)*(len(files)-1)//2 for files in by_person.values())
pos_done = 0

for pid, files in by_person.items():
    for i in range(len(files)):
        for j in range(i+1, len(files)):
            f1, f2 = files[i], files[j]
            v1, v2 = embs[f1], embs[f2]
            sim = float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))

            scores.append(sim)
            labels.append(1)

            pos_done += 1
            if pos_done % 10000 == 0:
                print(f"  Positives: {pos_done}/{pos_tasks} ({100*pos_done/pos_tasks:.1f}%)")

print("Positive pairs done:", pos_done)

# -------------------------------
# Negative Sampling (accurate)
# -------------------------------
neg_samples = pos_done * 5
print(f"\nSampling {neg_samples} negative pairs...")
neg_done = 0

for i in range(neg_samples):
    pid1, pid2 = random.sample(pids, 2)
    f1 = random.choice(by_person[pid1])
    f2 = random.choice(by_person[pid2])
    v1, v2 = embs[f1], embs[f2]
    sim = float(np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))

    scores.append(sim)
    labels.append(0)

    neg_done += 1
    if neg_done % 10000 == 0:
        print(f"  Negatives: {neg_done}/{neg_samples} ({100*neg_done/neg_samples:.1f}%)")

print("Negative sampling done.")

# -------------------------------
# ROC Curve
# -------------------------------
print("\nComputing ROC curve...")
fpr, tpr, thresholds = roc_curve(labels, scores)
roc_auc = auc(fpr, tpr)

# Compute FNIR
fnir = 1 - tpr

# Find optimal threshold (closest point to (0,0))
dist = np.sqrt(fpr**2 + fnir**2)
opt_idx = np.argmin(dist)
opt_threshold = thresholds[opt_idx]
opt_fnir = fnir[opt_idx]
opt_fpr = fpr[opt_idx]

print("\n=== RESULTS ===")
print("AUC:", roc_auc)
print(f"Optimal Threshold: {opt_threshold:.4f}, FNIR: {opt_fnir:.4f}, FPR: {opt_fpr:.4f}")

print("\nThreshold    FNIR      FPR")
print("--------------------------------")
for thr, fn, fp in zip(thresholds, fnir, fpr):
    print(f"{thr:.4f}    {fn:.4f}    {fp:.4f}")

# -------------------------------
# Plot ROC curve
# -------------------------------
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue')
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.scatter(opt_fpr, 1-opt_fnir, color='red', label=f'Optimal threshold {opt_threshold:.4f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)
plt.show()
