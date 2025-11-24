import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from identify_1vN import load_embedding, get_person_id, cosine_sim
import random

def identify_1vN(query, gallery_dir, top_k=5, sample_size=None):
    gallery_files = [f for f in os.listdir(gallery_dir) if f.endswith(".npy")]
    if sample_size:
        gallery_files = random.sample(gallery_files, min(sample_size, len(gallery_files)))

    scores = []
    for f in gallery_files:
        emb = load_embedding(os.path.join(gallery_dir, f))
        pid = get_person_id(f)
        sim = cosine_sim(query, emb)
        scores.append((pid, f, sim))

    scores.sort(key=lambda x: x[2], reverse=True)
    return scores[:top_k]  # top-K matches

def layered_verification(query_path, gallery_dir, top_k=5, sample_size=None):
    query = load_embedding(query_path)
    top_matches = identify_1vN(query, gallery_dir, top_k=top_k, sample_size=sample_size)

    print(f"\nQuery: {os.path.basename(query_path)}")
    print("Top-K 1vN matches:")
    for i, (pid, fname, sim) in enumerate(top_matches, 1):
        print(f"{i}. {pid}   ({fname})   sim={sim:.4f}")

    # Optional: verify best match with strict cosine similarity threshold
    best_pid, best_file, best_sim = top_matches[0]
    print(f"\n1v1 Verification (best match): {best_file}   sim={best_sim:.4f}")
    threshold = 0.8  # adjust based on your ROC/AUC
    if best_sim >= threshold:
        print(f"Verified as same person (sim >= {threshold})")
    else:
        print(f"Not verified (sim < {threshold})")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python layered_identification.py query.npy gallery_dir [top_k] [sample_size]")
        sys.exit(1)

    query_path = sys.argv[1]
    gallery_dir = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    sample_size = int(sys.argv[4]) if len(sys.argv) >= 5 else None

    layered_verification(query_path, gallery_dir, top_k=top_k, sample_size=sample_size)
