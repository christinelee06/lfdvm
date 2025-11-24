import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_embedding(path):
    return np.load(path)

def get_person_id(filename):
    return "_".join(filename.split("_")[:-1])

def cosine_sim(a, b):
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python identify_1vN.py query.npy gallery_dir [top_k] [sample_size]")
        sys.exit(1)

    query_path = sys.argv[1]
    gallery_dir = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
    sample_size = int(sys.argv[4]) if len(sys.argv) >= 5 else None  # optional sampling

    query = load_embedding(query_path)

    # Load gallery embeddings
    gallery_files = [f for f in os.listdir(gallery_dir) if f.endswith(".npy") and f != os.path.basename(query_path)]
    if sample_size:
        gallery_files = random.sample(gallery_files, min(sample_size, len(gallery_files)))

    scores = []
    for f in gallery_files:
        emb = load_embedding(os.path.join(gallery_dir, f))
        pid = get_person_id(f)
        sim = cosine_sim(query, emb)
        scores.append((pid, f, sim))

    # Sort by similarity
    scores.sort(key=lambda x: x[2], reverse=True)

    print("\nTop matches:")
    for i, (pid, fname, sim) in enumerate(scores[:top_k], 1):
        print(f"{i}. {pid}   ({fname})   sim={sim:.4f}")

    best = scores[0]
    print("\nBest Match:", best)
