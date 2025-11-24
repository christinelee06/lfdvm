import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

def load_embedding(path):
    return np.load(path)

def get_person_id(filename):
    return "_".join(filename.split("_")[:-1])

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 experiment_1vN.py gallery_dir top_k1,top_k2,... [sample_size]")
        sys.exit(1)

    gallery_dir = sys.argv[1]
    top_k_list = [int(x) for x in sys.argv[2].split(",")]
    sample_size = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    # Load embeddings
    gallery_files = [f for f in os.listdir(gallery_dir) if f.endswith(".npy")]
    if sample_size:
        gallery_files = random.sample(gallery_files, min(sample_size, len(gallery_files)))

    embs = {f: load_embedding(os.path.join(gallery_dir, f)) for f in gallery_files}
    for f in embs:
        embs[f] = embs[f] / np.linalg.norm(embs[f])  # normalize

    # Run experiment
    results = {k: 0 for k in top_k_list}
    total_queries = 0

    for query_file in gallery_files:
        query_emb = embs[query_file]
        query_pid = get_person_id(query_file)

        # Compute similarities
        scores = []
        for f, emb in embs.items():
            if f == query_file:
                continue
            scores.append((get_person_id(f), f, cosine_sim(query_emb, emb)))

        scores.sort(key=lambda x: x[2], reverse=True)
        top_pids = [pid for pid, _, _ in scores]

        total_queries += 1
